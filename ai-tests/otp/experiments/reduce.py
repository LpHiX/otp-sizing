"""Run + steady window -> ObservedPoint.

This is where a time history becomes a single number, and where every assumption
that turns raw volts into a physical quantity has to be stated.  Nothing here
invents a value: a channel the rig did not record produces None, and the point
then simply supports fewer residuals.
"""

from typing import Optional

import numpy as np

from ..core.states import GasState, LiquidState
from ..core.units import BAR, g
from .records import ObservedPoint, Run, SteadyWindow


def find_steady_window(run: Run, channel: str, t0: float, t1: float, width: float,
                       require_above: Optional[float] = None,
                       metric: str = "std") -> SteadyWindow:
    """Pick the flattest `width`-second window of `channel` inside [t0, t1].

    `require_above` is not optional in practice.  Without it the flattest window
    in a firing is the quiescent period AFTER shutdown -- perfectly steady, and
    not the operating point anyone wanted.  Pass a threshold that says "the rig
    was actually running", e.g. chamber pressure above some bar.

    A blunt tool, and deliberately so: it proposes a window, the analyst keeps
    or overrides it.  Automatic steady-state detection that nobody looks at is
    how a spin-up transient ends up averaged into a "steady" point.
    """
    ch = run[channel]
    m = (ch.t >= t0) & (ch.t <= t1) & np.isfinite(ch.v)
    t, v = ch.t[m], ch.v[m]
    if t.size == 0:
        raise ValueError(f"{channel} has no finite samples in [{t0}, {t1}]")

    starts = np.arange(t[0], t[-1] - width, width / 10)
    best, best_score = None, np.inf
    rejected = 0
    for s in starts:
        w = (t >= s) & (t <= s + width)
        if w.sum() < 10:
            continue
        seg = v[w]
        if require_above is not None and float(seg.min()) < require_above:
            rejected += 1
            continue
        score = float(seg.std()) if metric == "std" else float(
            abs(np.polyfit(t[w], seg, 1)[0]))
        if np.isfinite(score) and score < best_score:
            best, best_score = s, score
    if best is None:
        raise ValueError(
            f"no {width}s window in [{t0}, {t1}] has {channel} everywhere above "
            f"{require_above} ({rejected} candidate windows rejected on that test)")
    reason = f"minimum {metric} of {channel} = {best_score:.4g}"
    if require_above is not None:
        reason += f", with {channel} > {require_above:g} throughout"
    return SteadyWindow(run_id=run.run_id, label=f"flattest {channel}",
                        t0=float(best), t1=float(best + width), reason=reason)


def reduce_point(run: Run, window: SteadyWindow, liquid: LiquidState,
                 gg_R: Optional[float] = None, gg_gamma: Optional[float] = None,
                 gg_T0_K: Optional[float] = None,
                 gas_source: str = "", notes: str = "") -> ObservedPoint:
    """Average a window into one point.

    `liquid` is the pumped fluid state -- density and vapour pressure come from
    the analyst, not from a pressure trace.  `gg_R`, `gg_gamma`, `gg_T0_K` are
    the gas property assumptions for the turbine inlet; supply all three or the
    point carries no gas state and cannot drive a turbine comparison.
    """
    t0, t1 = window.t0, window.t1
    mean = lambda k: run[k].mean(t0, t1) if k in run.channels else None

    p_in = mean("p_pump_in")
    p_out = mean("p_pump_out")
    Q = mean("Q_pump")
    mdot_pump = mean("mdot_pump")
    rpm = mean("rpm")
    rpm_sd = run["rpm"].std(t0, t1) if "rpm" in run.channels else None
    rpm_source = run["rpm"].desc if "rpm" in run.channels else ""

    if Q is None and mdot_pump is not None:
        Q = mdot_pump / liquid.rho
    if mdot_pump is None and Q is not None:
        mdot_pump = Q * liquid.rho

    dp = (p_out - p_in) if (p_out is not None and p_in is not None) else None
    H = dp * BAR / (liquid.rho * g) if dp is not None else None

    gas = None
    p_gg = mean("p_gg_chamber")
    T_gg = gg_T0_K if gg_T0_K is not None else mean("T_gg_chamber")
    if None not in (p_gg, T_gg, gg_R, gg_gamma):
        gas = GasState(p0_bar=p_gg, T0_K=T_gg, R=gg_R, gamma=gg_gamma,
                       source=gas_source or f"{run.run_id} window {t0:.2f}-{t1:.2f}s")

    mdot_ox, mdot_fu = mean("mdot_gg_ox"), mean("mdot_gg_fuel")
    MR = (mdot_ox / mdot_fu) if (mdot_ox and mdot_fu) else None

    # Shaft power inferred from the measurement.  This is USEFUL hydraulic power
    # only -- it excludes disc friction, seal and bearing drag, so it is a lower
    # bound on what the turbine actually delivered, never a shaft-power reading.
    P_shaft_est = (mdot_pump * g * H) if (mdot_pump is not None and H is not None) else None
    extra = ("P_shaft_est is useful hydraulic power (mdot*g*H); "
             "mechanical and disc losses are NOT included.")

    return ObservedPoint(
        run_id=run.run_id, campaign=run.campaign, t0=t0, t1=t1,
        rpm=rpm, rpm_sd=rpm_sd, rpm_source=rpm_source,
        Q=Q, mdot_pump=mdot_pump, p_pump_in=p_in, p_pump_out=p_out,
        dp_pump=dp, H_total=H, gas=gas, p_turbine_exit=mean("p_turbine_exit"),
        mdot_gg_ox=mdot_ox, mdot_gg_fuel=mdot_fu, MR_gg=MR,
        liquid=liquid, P_shaft_est=P_shaft_est,
        notes=(notes + " " + extra).strip())


def reduce_series(run: Run, liquid: LiquidState, t0: float, t1: float, n: int = 20,
                  **kwargs) -> list[ObservedPoint]:
    """Chop [t0, t1] into n consecutive windows -- a spin-up becomes a curve.

    Each window is still averaged, so this is a sequence of quasi-steady points,
    not a transient model.  On a fast spin-up the rotor inertia means the shaft
    power balance does not hold within a window; label such points accordingly.
    """
    edges = np.linspace(t0, t1, n + 1)
    return [reduce_point(run,
                         SteadyWindow(run_id=run.run_id, label=f"slice {i}",
                                      t0=float(a), t1=float(b),
                                      reason="uniform slice of a sweep"),
                         liquid, **kwargs)
            for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))]
