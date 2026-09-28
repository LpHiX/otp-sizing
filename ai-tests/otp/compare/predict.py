"""Run a model at a measured point's own boundary conditions, and score it.

The rule the whole comparison rests on: the model is given exactly the
boundary conditions the rig had -- measured speed, measured inlet state,
measured gas state -- and is asked for the quantity that was also measured.
Nothing is tuned per point.  A model that had to be told the answer to
reproduce it has not been validated.
"""

from typing import Callable, Optional

import numpy as np

from ..core.component import component, q
from ..experiments.records import ObservedPoint


@component
class Residual:
    field: str = q("", "quantity compared")
    observed: float = q("", "measured value")
    predicted: float = q("", "model value")
    units: str = q("", "units of both")

    @property
    def absolute(self) -> float:
        return self.predicted - self.observed

    @property
    def relative(self) -> float:
        return self.absolute / self.observed if self.observed else float("nan")


@component
class Comparison:
    point: ObservedPoint
    model: str = q("", "which model produced the prediction")
    residuals: tuple = q("", "one Residual per comparable quantity")
    warnings: tuple = q("", "anything the model flagged", default=())
    skipped: tuple = q("", "quantities the run did not measure", default=())

    def table(self) -> str:
        head = (f"{self.point.run_id}  [{self.model}]  "
                f"{self.point.t0:.2f}-{self.point.t1:.2f}s")
        rows = [f"  {'quantity':<14}{'measured':>12}{'model':>12}{'error':>10}  units"]
        for r in self.residuals:
            rows.append(f"  {r.field:<14}{r.observed:>12.4g}{r.predicted:>12.4g}"
                        f"{r.relative * 100:>9.1f}%  {r.units}")
        if self.skipped:
            rows.append(f"  not measured: {', '.join(self.skipped)}")
        for w in self.warnings:
            rows.append(f"  ! {w}")
        return head + "\n" + "\n".join(rows)


_UNITS = {"H_total": "m", "dp_pump": "bar", "P_shaft_est": "W", "Q": "m^3/s",
          "mdot_gg": "kg/s", "rpm": "RPM"}


def compare_pump(point: ObservedPoint, pump_point: Callable,
                 model: str = "pump correlation") -> Comparison:
    """Predict pump head and power at the measured speed, flow and inlet state."""
    missing = [f for f in ("rpm", "Q", "liquid") if getattr(point, f) is None]
    if missing:
        return Comparison(point=point, model=model, residuals=(),
                          skipped=tuple(missing),
                          warnings=("pump comparison needs rpm, Q and liquid",))

    pp = pump_point(Q=point.Q, rpm=point.rpm, inlet=point.liquid)
    residuals, skipped, warn = [], [], []

    if point.H_total is not None:
        residuals.append(Residual("H_total", point.H_total, pp.H_total_real, "m"))
    else:
        skipped.append("H_total")
    if point.dp_pump is not None:
        residuals.append(Residual("dp_pump", point.dp_pump, pp.dp, "bar"))
    else:
        skipped.append("dp_pump")
    if point.P_shaft_est is not None:
        # measured side is USEFUL power; compare like with like
        residuals.append(Residual("P_useful", point.P_shaft_est, pp.P_useful, "W"))
        warn.append("P_useful compared, not shaft power: the rig has no torque "
                    "measurement, so mechanical and disc losses are unobserved")
    else:
        skipped.append("P_shaft_est")

    if getattr(pp, "cavitating", False):
        warn.append("model reports head collapse at this point")
    if "similarity" in pp.method and "[" in pp.method:
        warn.append(pp.method.split("[", 1)[1].rstrip("]"))

    return Comparison(point=point, model=model, residuals=tuple(residuals),
                      warnings=tuple(warn), skipped=tuple(skipped))


def compare_turbine(point: ObservedPoint, turbine, p_exit_bar: Optional[float] = None,
                    rpm: Optional[float] = None,
                    model: str = "turbine mean line") -> Comparison:
    """Compare the turbine's GAS MASS FLOW against measurement.

    Power is deliberately not compared.  Neither OT rig has a torque
    measurement, so the only power number on the measured side is the pump's
    useful hydraulic power -- a different quantity from turbine shaft power by
    the whole of the pump's internal losses, the bearing and seal drag, and,
    during a spin-up, the rotor inertia.  Scoring the model against it would
    produce a large error that says nothing about the model.

    Mass flow IS a fair test: it is directly measured, and for a choked nozzle
    it depends only on the inlet gas state and the throat, not on speed.  That
    makes it a clean check on half the model with the loss half held out.
    """
    p_exit = p_exit_bar if p_exit_bar is not None else point.p_turbine_exit
    speed = rpm if rpm is not None else point.rpm
    if point.gas is None or p_exit is None:
        return Comparison(point=point, model=model, residuals=(),
                          skipped=("gas", "p_turbine_exit"),
                          warnings=("turbine comparison needs a gas state and a "
                                    "backpressure",))
    if speed is None:
        # Choked flow does not depend on speed; use zero so the flow test can run
        # and let the velocity-triangle outputs be meaningless.
        speed = 0.0

    tp = turbine.point_performance(rpm=speed, gas=point.gas, p_exit_bar=p_exit)
    residuals, skipped, warn = [], [], list(tp.warnings)

    mdot_gg = None
    if point.mdot_gg_ox is not None and point.mdot_gg_fuel is not None:
        mdot_gg = point.mdot_gg_ox + point.mdot_gg_fuel
    if mdot_gg is not None and mdot_gg > 0:
        residuals.append(Residual("mdot_gg", mdot_gg, tp.mdot, "kg/s"))
    else:
        skipped.append("mdot_gg (not measured, or not positive in this window)")

    skipped.append("P_shaft (no torque measurement on this rig)")
    return Comparison(point=point, model=model, residuals=tuple(residuals),
                      warnings=tuple(dict.fromkeys(warn)), skipped=tuple(skipped))


def summarise(comparisons) -> str:
    """RMS relative error per quantity, across a set of comparisons."""
    by_field: dict[str, list[float]] = {}
    for c in comparisons:
        for r in c.residuals:
            if np.isfinite(r.relative):
                by_field.setdefault(r.field, []).append(r.relative)
    lines = [f"  {'quantity':<14}{'n':>4}{'RMS rel err':>14}{'bias':>10}"]
    for field, vals in sorted(by_field.items()):
        a = np.array(vals)
        lines.append(f"  {field:<14}{a.size:>4}{np.sqrt((a ** 2).mean()) * 100:>13.1f}%"
                     f"{a.mean() * 100:>9.1f}%")
    return "\n".join(lines)
