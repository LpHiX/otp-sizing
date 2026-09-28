"""Steady operating point of a fixed turbopump on a fixed gas supply.

Two unknowns, two balances:

    power:  P_turbine(rpm)            = P_pump(Q, rpm) + P_extra(rpm)
    head:   H_pump(Q, rpm)            = H_system(Q)

A root of these is a CANDIDATE steady point.  It is not proof of dynamic
stability, not proof of uniqueness, and says nothing about whether the pump is
cavitating there -- check `warnings` and the returned pump point for that.

What this deliberately does not do: it takes the gas supply as prescribed.  If
gas-generator pressure responds to pump discharge (it does, once the pump feeds
the engine that feeds the GG), this solve becomes the inner loop of a larger
one.  Prescribing inlet pressure, temperature, a choked throat AND a demanded
turbine power at the same time over-specifies the problem; only three of those
are independent.

`pump_point` is any callable with the package's liquid signature:

    pump_point(Q, rpm, inlet) -> PumpPerformance

so the Lock/Barske correlation, a similarity-scaled measured curve, and a
pump+inducer assembly all drop in without the solver changing.
"""

from typing import Callable, Optional

import numpy as np
import scipy.optimize as opt

from ..core.component import component, q
from ..core.states import GasState, LiquidState
from ..core.units import BAR, g


@component
class HydraulicSystem:
    """What the plumbing downstream demands, as a total pressure rise.

    `required_dp_pa(Q, liquid) -> Pa` must span the SAME two stations as the
    pump model's rise, and must already include tank/supply pressure, downstream
    pressure, line and valve losses, elevation and any static-to-total step.
    """
    required_dp_pa: Callable = q("", "f(Q, liquid) -> Pa")
    label: str = q("", "what this system is", default="")

    def required_head_m(self, Q: float, liquid: LiquidState) -> float:
        return float(self.required_dp_pa(Q, liquid)) / (liquid.rho * g)

    @staticmethod
    def orifice(K_pa_per_m6s2: float, dp_static_pa: float = 0.0,
                label: str = "quadratic resistance") -> "HydraulicSystem":
        """dp = dp_static + K * Q^2 -- a fixed restriction, e.g. a test orifice."""
        return HydraulicSystem(
            required_dp_pa=lambda Q, liquid: dp_static_pa + K_pa_per_m6s2 * Q ** 2,
            label=label)

    @staticmethod
    def injector(CdA: float, p_chamber_bar: float, p_supply_bar: float,
                 label: str = "injector into a fixed chamber") -> "HydraulicSystem":
        """An injector feeding a chamber held at a PRESCRIBED pressure.

        The prescribed chamber pressure makes any answer conditional on it.  To
        predict chamber pressure as well, the system needs a fixed-engine
        closure (injector areas on both sides plus a mass/energy balance), or
        the solver needs another unknown and another equation.
        """
        def dp(Q, liquid):
            mdot = Q * liquid.rho
            dp_inj = (mdot / CdA) ** 2 / (2 * liquid.rho)
            return (p_chamber_bar - p_supply_bar) * BAR + dp_inj
        return HydraulicSystem(required_dp_pa=dp, label=label)


@component
class SteadyState:
    converged: bool = q("-", "both residuals met the tolerance")
    rpm: float = q("RPM", "solved shaft speed")
    Q: float = q("m^3/s", "solved pump flow", alt=("L/s", lambda x: x * 1000))
    head_m: float = q("m", "pump total head at the solution")
    pump_power_W: float = q("W", "pump shaft power", alt=("kW", lambda x: x / 1000))
    turbine_power_W: float = q("W", "turbine net shaft power", alt=("kW", lambda x: x / 1000))
    head_residual_m: float = q("m", "pump head minus system demand")
    power_residual_W: float = q("W", "turbine power minus total shaft demand")
    roots: tuple = q("", "every distinct (rpm, turbine power) root found", default=())
    pump: object = q("", "the PumpPerformance at the solution", default=None)
    turbine: object = q("", "the TurbinePerformance at the solution", default=None)
    warnings: tuple = q("", "everything the solver wants flagged", default=())
    message: str = q("", "solver message", default="")


def solve_steady_state(turbine, pump_point: Callable, liquid: LiquidState,
                       gas: GasState, p_exit_bar: float, system: HydraulicSystem,
                       rpm_bounds: tuple[float, float],
                       Q_bounds: tuple[float, float],
                       extra_shaft_power: Callable | float = 0.0,
                       head_tol_m: float = 1e-3,
                       power_tol_W: float = 1e-1) -> SteadyState:
    extra = extra_shaft_power if callable(extra_shaft_power) else (lambda rpm: extra_shaft_power)

    lo_rpm, hi_rpm = rpm_bounds
    lo_Q, hi_Q = Q_bounds
    if not (lo_rpm > 0 and lo_Q > 0):
        raise ValueError("bounds must be strictly positive; the solve works in log space")

    def unpack(x):
        # log space keeps the two unknowns comparably scaled and strictly positive
        return float(np.exp(x[0])), float(np.exp(x[1]))

    def residuals(x):
        rpm, Q = unpack(x)
        try:
            pp = pump_point(Q=Q, rpm=rpm, inlet=liquid)
            tp = turbine.point_performance(rpm=rpm, gas=gas, p_exit_bar=p_exit_bar)
        except Exception:
            return [1e6, 1e6]
        H_dem = system.required_head_m(Q, liquid)
        r_head = pp.H_total_real - H_dem
        r_power = tp.P_shaft - (pp.P_shaft + extra(rpm))
        if not np.isfinite(r_head) or not np.isfinite(r_power):
            return [1e6, 1e6]
        # scale the power residual so neither balance dominates the least squares
        return [r_head, r_power / 1e3]

    # Multi-start, because this system genuinely has more than one root.  A
    # partial-admission turbine carries a scavenging loss that does not scale
    # with blade speed, so at low speed the net power crosses zero -- and a
    # near-zero turbine power balanced against a near-zero pump power is a
    # perfectly good root of both equations, and a useless answer.  Starting
    # from one guess lands on whichever root is nearest, which is usually that
    # one.  Collect every root found, then pick.
    starts = [(r, q_) for r in np.geomspace(lo_rpm, hi_rpm, 6)
              for q_ in np.geomspace(lo_Q, hi_Q, 4)]
    found = []
    for r0, q0 in starts:
        s = opt.least_squares(
            residuals, [np.log(r0), np.log(q0)],
            bounds=([np.log(lo_rpm), np.log(lo_Q)],
                    [np.log(hi_rpm), np.log(hi_Q)]),
            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        rr, qq = unpack(s.x)
        try:
            pp_ = pump_point(Q=qq, rpm=rr, inlet=liquid)
            tp_ = turbine.point_performance(rpm=rr, gas=gas, p_exit_bar=p_exit_bar)
        except Exception:
            continue
        rh = pp_.H_total_real - system.required_head_m(qq, liquid)
        rp = tp_.P_shaft - (pp_.P_shaft + extra(rr))
        if abs(rh) < head_tol_m and abs(rp) < power_tol_W:
            found.append((rr, qq, tp_.P_shaft, s))

    if found:
        # Distinct roots only, then the one that actually delivers power: a
        # solution sitting at zero turbine power is a stall, not an operating
        # point, and reporting it as "the" answer would be misleading.
        found.sort(key=lambda f: f[0])
        distinct = []
        for f in found:
            if not distinct or abs(np.log(f[0] / distinct[-1][0])) > 1e-3:
                distinct.append(f)
        sol = max(distinct, key=lambda f: f[2])[3]
        multi = [(f[0], f[2]) for f in distinct]
    else:
        sol = opt.least_squares(
            residuals, [np.log(np.sqrt(lo_rpm * hi_rpm)), np.log(np.sqrt(lo_Q * hi_Q))],
            bounds=([np.log(lo_rpm), np.log(lo_Q)],
                    [np.log(hi_rpm), np.log(hi_Q)]),
            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        multi = []

    rpm, Q = unpack(sol.x)
    pp = pump_point(Q=Q, rpm=rpm, inlet=liquid)
    tp = turbine.point_performance(rpm=rpm, gas=gas, p_exit_bar=p_exit_bar)
    r_head = pp.H_total_real - system.required_head_m(Q, liquid)
    r_power = tp.P_shaft - (pp.P_shaft + extra(rpm))

    warn = list(tp.warnings)
    if getattr(pp, "cavitating", False):
        warn.append("the pump correlation reports head collapse at this point")
    for edge, name, val in ((lo_rpm, "rpm", rpm), (hi_rpm, "rpm", rpm),
                            (lo_Q, "Q", Q), (hi_Q, "Q", Q)):
        if abs(np.log(val) - np.log(edge)) < 1e-6:
            warn.append(f"solution sits on the {name} bound {edge:g}; widen the bracket")
    converged = abs(r_head) < head_tol_m and abs(r_power) < power_tol_W
    if not converged:
        warn.append("residuals did not meet tolerance; this is not a steady point")
    if len(multi) > 1:
        others = ", ".join(f"{r:.0f} rpm ({p / 1000:.2f} kW)" for r, p in multi)
        warn.append(f"{len(multi)} roots satisfy both balances -- {others}. The one "
                    "delivering the most power is reported; the others are stall "
                    "or low-power branches. Which one the rig actually settles on "
                    "is a startup and stability question this solve cannot answer.")
    if tp.P_shaft <= 0:
        warn.append("the reported root delivers no net turbine power: this is a "
                    "stalled solution, not an operating point")

    return SteadyState(
        converged=converged, rpm=rpm, Q=Q, head_m=pp.H_total_real,
        pump_power_W=pp.P_shaft, turbine_power_W=tp.P_shaft,
        head_residual_m=r_head, power_residual_W=r_power, roots=tuple(multi),
        pump=pp, turbine=tp, warnings=tuple(dict.fromkeys(warn)), message=sol.message)
