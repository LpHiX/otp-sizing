"""Affinity-law scaling of a measured pump curve.

A prediction route that is independent of the Lock/Barske correlations, so that
agreeing with it means something.  For the same pump at corresponding points:

    Q  ~ N          H ~ N^2         P_fluid ~ rho * N^3

Two limits worth stating rather than discovering later:

  * Affinity assumes geometric and kinematic similarity.  Water -> IPA changes
    viscosity by about a factor of 2.5 and vapour pressure by more, so Reynolds
    number and suction margin both move.  Scaled head is not corrected for
    either; `warnings` flags a large Reynolds change.
  * Mechanical drag -- bearings, seals -- does not scale with density.  Fluid
    power is scaled here and mechanical power is added separately, which is why
    `ReferenceCurve.P_fluid` must have mechanical losses ALREADY removed.

No silent extrapolation: a flow outside the reference range returns NaN.
"""

from typing import Callable, Optional

import numpy as np

from ..core.component import component, q
from ..core.states import LiquidState
from ..core.units import BAR, g, rpm_to_rad
from ..pump.records import PumpPerformance


@component
class ReferenceCurve:
    """A measured pump curve at one speed and one fluid."""
    rpm: float = q("RPM", "speed the curve was measured at")
    rho: float = q("kg/m^3", "density of the fluid it was measured with")
    Q: np.ndarray = q("m^3/s", "flow points")
    H_total: np.ndarray = q("m", "total head at those flows")
    P_fluid: np.ndarray = q("W", "shaft power MINUS separately-modelled mechanical drag")
    nu: Optional[float] = q("m^2/s", "kinematic viscosity of the reference fluid",
                            default=None)
    source: str = q("", "where this curve came from", default="")

    def __post_init__(self):
        self.Q = np.asarray(self.Q, float)
        self.H_total = np.asarray(self.H_total, float)
        self.P_fluid = np.asarray(self.P_fluid, float)
        order = np.argsort(self.Q)
        self.Q, self.H_total, self.P_fluid = (self.Q[order], self.H_total[order],
                                              self.P_fluid[order])

    @staticmethod
    def from_observed(points, mechanical_power: float = 0.0,
                      source: str = "measured") -> "ReferenceCurve":
        """Build a reference from reduced measurements, all at one speed."""
        pts = [p for p in points if None not in (p.Q, p.H_total, p.rpm)]
        if not pts:
            raise ValueError("no reduced points carry Q, H_total and rpm")
        speeds = np.array([p.rpm for p in pts])
        if speeds.std() / speeds.mean() > 0.05:
            raise ValueError(f"points span {speeds.min():.0f}-{speeds.max():.0f} rpm; "
                             "a reference curve must be at one speed. For a spin-up, "
                             "use from_observed_affinity instead.")
        rho = pts[0].liquid.rho
        return ReferenceCurve(
            rpm=float(speeds.mean()), rho=rho,
            Q=[p.Q for p in pts], H_total=[p.H_total for p in pts],
            P_fluid=[p.mdot_pump * g * p.H_total - mechanical_power for p in pts],
            nu=pts[0].liquid.nu, source=source)

    @staticmethod
    def from_observed_affinity(points, rpm_ref: float, mechanical_power: float = 0.0,
                               source: str = "affinity-collapsed") -> "ReferenceCurve":
        """Collapse a VARIABLE-SPEED sweep onto one reference speed.

        A spin-up never holds a constant speed, so there is no single-speed curve
        to build.  Affinity says every point maps to the reference speed by

            Q -> Q * (N_ref/N)      H -> H * (N_ref/N)^2

        If the pump really is behaving self-similarly the mapped points fall on
        one curve, and the scatter about that curve IS the test of similarity.
        `collapse_residual` measures it.
        """
        pts = [p for p in points if None not in (p.Q, p.H_total, p.rpm) and p.rpm > 0]
        if len(pts) < 3:
            raise ValueError(f"{len(pts)} usable points; need at least 3")
        r = np.array([rpm_ref / p.rpm for p in pts])
        return ReferenceCurve(
            rpm=rpm_ref, rho=pts[0].liquid.rho,
            Q=[p.Q * ri for p, ri in zip(pts, r)],
            H_total=[p.H_total * ri ** 2 for p, ri in zip(pts, r)],
            P_fluid=[(p.mdot_pump * g * p.H_total - mechanical_power) * ri ** 3
                     for p, ri in zip(pts, r)],
            nu=pts[0].liquid.nu,
            source=f"{source} to {rpm_ref:.0f} rpm from {len(pts)} points spanning "
                   f"{min(p.rpm for p in pts):.0f}-{max(p.rpm for p in pts):.0f} rpm")

    def collapse_residual(self) -> float:
        """RMS relative scatter of the collapsed head about a smooth fit.

        Small means the points really do lie on one curve, i.e. affinity holds
        across the speeds used.  Large means it does not, and scaling a measured
        curve to a new speed will be wrong by about this much.
        """
        if self.Q.size < 4:
            return float("nan")
        coeffs = np.polyfit(self.Q, self.H_total, min(2, self.Q.size - 2))
        fit = np.polyval(coeffs, self.Q)
        nz = fit != 0
        return float(np.sqrt((((self.H_total[nz] - fit[nz]) / fit[nz]) ** 2).mean()))


@component
class SimilarityPump:
    """Predicts a point by scaling a measured curve, not by a correlation."""
    reference: ReferenceCurve
    mechanical_power: Callable = q("", "f(rpm) -> W of bearing/seal drag",
                                   default=lambda rpm: 0.0)
    re_change_warn: float = q("-", "flag a Reynolds change beyond this factor",
                              default=2.0)

    def point_performance(self, Q: float, rpm: float,
                          inlet: LiquidState) -> PumpPerformance:
        ref = self.reference
        speed_ratio = rpm / ref.rpm
        Q_ref = Q / speed_ratio                       # corresponding point on the reference

        if not (ref.Q[0] <= Q_ref <= ref.Q[-1]):
            H = P_fluid = np.nan
        else:
            H = float(np.interp(Q_ref, ref.Q, ref.H_total)) * speed_ratio ** 2
            P_fluid = (float(np.interp(Q_ref, ref.Q, ref.P_fluid))
                       * (inlet.rho / ref.rho) * speed_ratio ** 3)

        P_mech = float(self.mechanical_power(rpm))
        P_shaft = P_fluid + P_mech
        mdot = Q * inlet.rho
        omega = rpm_to_rad(rpm)
        dp = H * inlet.rho * g / BAR
        P_useful = mdot * g * H

        warn = []
        if np.isnan(H):
            warn.append(f"Q/N = {Q_ref * 1e3:.3f} L/s at reference speed is outside the "
                        f"measured range {ref.Q[0] * 1e3:.3f}-{ref.Q[-1] * 1e3:.3f} L/s")
        if ref.nu and inlet.nu and not (1 / self.re_change_warn
                                        <= inlet.nu / ref.nu <= self.re_change_warn):
            warn.append(f"kinematic viscosity changed by x{inlet.nu / ref.nu:.2f} from the "
                        "reference fluid; affinity scaling is uncorrected for Reynolds")

        nan = float("nan")
        return PumpPerformance(
            liquid=inlet, Q=Q, rpm=rpm, p_upstream=inlet.p0_bar, T_upstream=inlet.T_K,
            omega=omega, mdot=mdot, dp=dp,
            flow_coeff_inlet=nan, flow_coeff_outlet=nan, head_coeff=nan,
            H_total_real=H, H_static_real=nan, H_loss_diff=nan, eta_hydraulic=nan,
            P_shaft=P_shaft, P_hydraulic=P_fluid, P_disc_friction=nan,
            P_useful=P_useful, eta_power=P_useful / P_shaft if P_shaft else nan,
            torque=P_shaft / omega if omega else nan,
            v_inlet=nan, u_1=nan, w_1=nan, v_1=nan, u_2=nan, w_2=nan, v_2=nan,
            v_throat=nan, v_outlet=nan, p_inlet=inlet.p0_bar, p_1=nan, p_2=nan,
            p_2_total=nan, p_throat=nan, p_outlet=inlet.p0_bar + dp,
            p_outlet_static=nan, npsh_a_upstream=inlet.npsha_m,
            npsh_r_inlet=nan, npsh_r_throat=nan, npsh_r_ai_low=nan, npsh_r_ai_high=nan,
            suction_specific_speed=nan,
            method="similarity" + ("  [" + "; ".join(warn) + "]" if warn else ""),
            cavitating=False)
