"""Fit calibration coefficients to measured points.

The point of the whole architecture: hardware geometry is fixed by what was
built, boundary conditions are fixed by what was measured, and the only things
free to move are the coefficients in a `*Calibration` record.  Fitting anything
else would be fitting the rig rather than the model.

Guard rails that matter more than the fit itself:

  * a fit is only meaningful if there are more independent measurements than
    free coefficients -- `fit` refuses otherwise rather than returning an exact
    but meaningless solution;
  * the result carries the residual it achieved and the points it used, and
    stamps `source`, so a calibrated coefficient can never be mistaken for a
    literature value later.
"""

from dataclasses import replace
from typing import Callable, Sequence

import numpy as np
import scipy.optimize as opt

from ..core.component import component, q


@component
class CalibrationFit:
    calibration: object = q("", "the fitted *Calibration record")
    names: tuple = q("", "coefficients that were free")
    values: tuple = q("", "fitted values")
    bounds: tuple = q("", "bounds they were held inside")
    rms_relative: float = q("-", "RMS relative residual achieved")
    n_points: int = q("-", "measurements used")
    n_free: int = q("-", "coefficients fitted")
    at_bound: tuple = q("", "coefficients that ran into a bound", default=())
    run_ids: tuple = q("", "which runs the points came from", default=())
    message: str = q("", "optimiser message", default="")

    def report(self) -> str:
        lines = [f"fit: {self.n_free} coefficients to {self.n_points} measurements, "
                 f"RMS relative residual {self.rms_relative * 100:.1f}%",
                 f"runs: {', '.join(self.run_ids)}"]
        for n, v, b in zip(self.names, self.values, self.bounds):
            flag = "  <-- AT BOUND" if n in self.at_bound else ""
            lines.append(f"  {n:<24}{v:>10.4g}   bounds {b[0]:g} to {b[1]:g}{flag}")
        if self.at_bound:
            lines.append("  a coefficient at a bound means the fit wanted to go further; "
                         "the model is probably missing a term, not a better number")
        return "\n".join(lines)


def fit(calibration, free: dict[str, tuple[float, float]],
        points: Sequence, residual_fn: Callable,
        source: str | None = None) -> CalibrationFit:
    """Least-squares fit of named coefficients on a `*Calibration` record.

    `free`        coefficient name -> (lower, upper) bound
    `residual_fn` (calibration, point) -> sequence of RELATIVE residuals, or an
                  empty sequence if that point cannot be scored.
    """
    names = tuple(free)
    lo = np.array([free[n][0] for n in names], float)
    hi = np.array([free[n][1] for n in names], float)
    x0 = np.array([getattr(calibration, n) for n in names], float)
    x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)

    def build(x):
        return replace(calibration, **{n: float(v) for n, v in zip(names, x)})

    def all_residuals(x):
        cal = build(x)
        out = []
        for p in points:
            out.extend(residual_fn(cal, p))
        return np.array(out, float) if out else np.array([0.0])

    n_res = all_residuals(x0).size
    if n_res <= len(names):
        raise ValueError(
            f"{n_res} usable residuals for {len(names)} free coefficients: the fit "
            "would be underdetermined. Free fewer coefficients, or reduce more points.")

    sol = opt.least_squares(all_residuals, x0, bounds=(lo, hi),
                            xtol=1e-12, ftol=1e-12)
    res = all_residuals(sol.x)
    at_bound = tuple(n for n, v, a, b in zip(names, sol.x, lo, hi)
                     if abs(v - a) < 1e-9 or abs(v - b) < 1e-9)

    fitted = build(sol.x)
    run_ids = tuple(dict.fromkeys(getattr(p, "run_id", "?") for p in points))
    if hasattr(fitted, "source"):
        fitted = replace(fitted, source=source or
                         f"fitted to {len(points)} points from {', '.join(run_ids)}")

    return CalibrationFit(
        calibration=fitted, names=names, values=tuple(float(v) for v in sol.x),
        bounds=tuple(free[n] for n in names),
        rms_relative=float(np.sqrt((res ** 2).mean())), n_points=n_res,
        n_free=len(names), at_bound=at_bound, run_ids=run_ids, message=sol.message)


def pump_head_residual(pump_factory: Callable) -> Callable:
    """Relative head residual, for `fit`.  `pump_factory(calibration) -> Pump`."""
    def residual(calibration, point):
        if None in (point.rpm, point.Q, point.H_total, point.liquid) or not point.H_total:
            return ()
        pump = pump_factory(calibration)
        pp = pump.point_performance(Q=point.Q, rpm=point.rpm, inlet=point.liquid)
        if not np.isfinite(pp.H_total_real):
            return ()
        return ((pp.H_total_real - point.H_total) / point.H_total,)
    return residual


def turbine_flow_residual(turbine_factory: Callable) -> Callable:
    """Relative gas mass flow residual, for `fit`."""
    def residual(calibration, point):
        if point.gas is None or point.rpm is None or point.p_turbine_exit is None:
            return ()
        if point.mdot_gg_ox is None or point.mdot_gg_fuel is None:
            return ()
        measured = point.mdot_gg_ox + point.mdot_gg_fuel
        if not measured:
            return ()
        tp = turbine_factory(calibration).point_performance(
            rpm=point.rpm, gas=point.gas, p_exit_bar=point.p_turbine_exit)
        return ((tp.mdot - measured) / measured,)
    return residual
