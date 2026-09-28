"""Fixed-geometry turbine forward analysis.

This is the calculation the current hardware question needs and the notebook did
not have.  Given geometry, an inlet gas state, a backpressure and a shaft speed,
it returns mass flow, net shaft power and torque.  It does NOT take a required
power: required power is a sizing input, not a boundary condition, and letting a
forward model consume it is how a model ends up unable to disagree with you.

The FYP `partload_rpm` routine works the other way round -- it prescribes speed
and a cubic pump load and solves for the gas supply needed.  That cannot answer
"what speed does this turbopump settle at on a given gas supply", which is why
this function exists rather than a call into that one.

Mean-line, single stage, impulse, constant gas properties.  Each loss is a
separate named term with its own coefficient so that calibration can move one
without moving the others.  Coefficient defaults are placeholders -- see
`TurbineCalibration`.
"""

import numpy as np

from ..core.component import component, q
from ..core.states import GasState
from ..core.units import BAR, rpm_to_rad
from .records import TurbineCalibration, TurbineGeometry, TurbinePerformance


def nozzle_mass_flow(gas: GasState, A_throat: float, p_exit_bar: float,
                     cd: float) -> tuple[float, bool]:
    """Total nozzle mass flow, choked or subsonic."""
    gam, R, T0 = gas.gamma, gas.R, gas.T0_K
    p0_pa, pe_pa = gas.p0_bar * BAR, p_exit_bar * BAR
    critical = ((gam + 1) / 2) ** (gam / (gam - 1))
    choked = (p0_pa / pe_pa) >= critical
    if choked:
        mdot = (cd * A_throat * p0_pa * np.sqrt(gam / (R * T0))
                * (2 / (gam + 1)) ** ((gam + 1) / (2 * (gam - 1))))
    else:
        pr = pe_pa / p0_pa
        mdot = (cd * A_throat * p0_pa / np.sqrt(R * T0)
                * np.sqrt(2 * gam / (gam - 1) * (pr ** (2 / gam) - pr ** ((gam + 1) / gam))))
    return float(mdot), bool(choked)


def spouting_velocity(gas: GasState, p_exit_bar: float) -> float:
    """Isentropic velocity available in expanding from p0 to p_exit."""
    gam = gas.gamma
    pr = min(p_exit_bar / gas.p0_bar, 1.0)
    return float(np.sqrt(2 * gas.cp * gas.T0_K * (1 - pr ** ((gam - 1) / gam))))


@component
class Turbine:
    geom: TurbineGeometry
    calib: TurbineCalibration = q("", "loss coefficients",
                                  default_factory=TurbineCalibration)

    def point_performance(self, rpm: float, gas: GasState,
                          p_exit_bar: float) -> TurbinePerformance:
        g_, c = self.geom, self.calib
        warn = []

        if p_exit_bar >= gas.p0_bar:
            raise ValueError(f"backpressure {p_exit_bar} bar is not below inlet total "
                             f"{gas.p0_bar} bar; the turbine cannot expand")
        if not 0 < g_.admission_fraction <= 1:
            raise ValueError(f"admission_fraction must be in (0, 1], got "
                             f"{g_.admission_fraction}")

        mdot, choked = nozzle_mass_flow(gas, g_.A_throat, p_exit_bar, c.nozzle_cd)
        c_is = spouting_velocity(gas, p_exit_bar)
        dh_is = 0.5 * c_is ** 2

        c_1 = c.nozzle_velocity_coeff * c_is
        omega = rpm_to_rad(rpm)
        u = omega * g_.d_mean / 2

        a1, b2 = np.radians(g_.alpha_1_deg), np.radians(g_.beta_2_deg)
        c_u1, c_m1 = c_1 * np.cos(a1), c_1 * np.sin(a1)
        w_u1 = c_u1 - u
        w_1 = float(np.hypot(w_u1, c_m1))
        beta_1 = float(np.degrees(np.arctan2(c_m1, w_u1)))

        w_2 = c.rotor_velocity_coeff * w_1
        w_u2 = -w_2 * np.cos(b2)                     # turned back across the rotor
        c_u2 = u + w_u2
        c_2 = float(np.hypot(c_u2, w_2 * np.sin(b2)))

        dh_euler = float(u * (c_u1 - c_u2))
        P_euler = mdot * dh_euler

        # Exit state, used only by the loss terms that need a density.
        T_exit = gas.T0_K - dh_euler / gas.cp
        if T_exit <= 0:
            raise ValueError("Euler work exceeds the gas total enthalpy; check the "
                             "angle convention and the velocity coefficients")
        rho_exit = p_exit_bar * BAR / (gas.R * T_exit)

        # --- loss terms, each independently calibratable ------------------------
        # Tip leakage: gas that goes over the blade tips does no work.  On both
        # OT rigs this is the largest single loss, so it leads.
        leak_fraction = min(c.k_leakage * g_.tip_clearance / g_.blade_height, 1.0)
        P_leakage = leak_fraction * P_euler
        # Inactive-arc windage: the blades outside the admitted sector pump gas.
        P_windage = (c.k_windage * (1 - g_.admission_fraction) * rho_exit * u ** 3
                     * g_.d_mean * g_.blade_height)
        # Sector filling/emptying: kinetic energy spent re-accelerating the passage
        # gas each time a blade enters the admitted arc (Stenning-type scaling).
        zeta_sector = (c.k_sector * g_.blade_chord
                       / (np.pi * g_.d_mean * g_.admission_fraction))
        P_sector = zeta_sector * mdot * c_1 ** 2 / 2
        # Disc friction on the rotor faces.
        P_disc = c.k_disc_friction * rho_exit * omega ** 3 * (g_.d_mean / 2) ** 5

        P_shaft = P_euler - P_leakage - P_windage - P_sector - P_disc - c.mechanical_power

        if not choked:
            warn.append("nozzle not choked; the supersonic nozzle is running off-design")
        if leak_fraction > 0.3:
            warn.append(f"tip leakage is {leak_fraction * 100:.0f}% of Euler work; "
                        "the gap/height ratio dominates this design")
        if rpm == 0:
            # Called purely for the choked mass flow, which does not need a speed.
            warn.append("rpm is zero: the mass flow is still valid, but every "
                        "velocity-triangle and power output in this record is not")
        else:
            if u / c_is > 0.5:
                warn.append(f"u/c_is = {u / c_is:.2f} is past the impulse optimum (~0.5)")
            if u / c_is < 0.2:
                warn.append(f"u/c_is = {u / c_is:.3f} is far below the impulse optimum "
                            "(~0.5); most of the jet kinetic energy leaves unused, "
                            "which is why the isentropic efficiency is low")
            if beta_1 < 0:
                warn.append(f"beta_1 = {beta_1:.1f} deg: blade speed exceeds the "
                            "tangential gas velocity, the rotor is being dragged")
            if P_shaft <= 0:
                warn.append("net shaft power is not positive at this speed")

        return TurbinePerformance(
            gas=gas, rpm=rpm, p_exit_bar=p_exit_bar, omega=omega, mdot=mdot,
            choked=choked, u=u, c_is=c_is, c_1=c_1, w_1=w_1, w_2=w_2, c_2=c_2,
            beta_1_deg=beta_1, u_over_c_is=u / c_is if c_is else np.nan,
            dh_euler=dh_euler, dh_is=dh_is, P_euler=P_euler, P_leakage=P_leakage,
            P_windage=P_windage,
            P_sector=P_sector, P_disc_friction=P_disc, P_mechanical=c.mechanical_power,
            P_shaft=P_shaft, torque=P_shaft / omega if omega else np.nan,
            eta_isentropic=P_shaft / (mdot * dh_is) if mdot * dh_is else np.nan,
            eta_blade=P_euler / (mdot * dh_is) if mdot * dh_is else np.nan,
            T_exit_K=T_exit, rho_exit=rho_exit,
            blade_passing_hz=g_.n_blades * rpm / 60.0,
            warnings=tuple(warn))
