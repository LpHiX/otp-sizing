"""Turbine records.

Angle convention, fixed once here because every mean-line source picks a
different one and mixing them is the usual way this calculation goes wrong:

    alpha_1_deg and beta_2_deg are measured FROM THE TANGENTIAL DIRECTION,
    in the plane of rotation.  A nozzle that points nearly tangentially (what a
    partial-admission impulse stage wants) therefore has a SMALL alpha_1.

turboRocket's turbine_in.yaml uses `alpha: 70` measured from the axis of flow,
i.e. 20 degrees from tangential.  campaigns/ot2.py does that conversion once,
where it can be checked, rather than burying it in the physics.
"""

from ..core.component import component, q
from ..core.states import GasState
from ..core.units import K_TO_C


@component
class TurbineRequirements:
    P_req: float = q("W", "net shaft power the turbine must deliver",
                     alt=("kW", lambda x: x / 1000))
    rpm: float = q("RPM", "shaft speed at which that power is required")
    source: str = q("", "what set this requirement", default="unspecified")

    @staticmethod
    def from_pump_perfs(pump_perfs, rpm: float | None = None,
                        extra_shaft_power_W: float = 0.0,
                        source: str = "sum of pump shaft powers") -> "TurbineRequirements":
        """Demand comes from SHAFT power, not hydraulic power.

        `extra_shaft_power_W` is every other load on the same shaft -- bearings,
        seals, inducer -- and defaults to zero rather than to a guess.
        """
        perfs = list(pump_perfs)
        if rpm is None:
            speeds = {round(p.rpm, 6) for p in perfs}
            if len(speeds) != 1:
                raise ValueError(f"pump points are at different speeds: {speeds}; "
                                 "pass rpm= explicitly")
            rpm = perfs[0].rpm
        return TurbineRequirements(
            P_req=sum(p.P_shaft for p in perfs) + extra_shaft_power_W,
            rpm=rpm, source=source)


@component
class TurbineChoices:
    d_mean: float = q("m", "mean (pitch) diameter", alt=("mm", lambda x: x * 1000))
    n_nozzles: int = q("-", "number of stator nozzles")
    n_blades: int = q("-", "number of rotor blades")
    alpha_1_deg: float = q("deg", "nozzle absolute exit angle from TANGENTIAL")
    beta_2_deg: float = q("deg", "rotor relative exit angle from TANGENTIAL")
    blade_height: float = q("m", "rotor blade height", alt=("mm", lambda x: x * 1000))
    admission_fraction: float = q("-", "fraction of the annulus fed by nozzles")
    tip_clearance: float = q("m", "rotor tip gap", alt=("mm", lambda x: x * 1000),
                             default=2e-3)
    blade_chord: float = q("m", "rotor blade chord", alt=("mm", lambda x: x * 1000),
                           default=5e-3)


@component
class TurbineGeometry:
    """Hardware only.  Unchanged when speed, gas or backpressure change."""
    d_mean: float = q("m", "mean (pitch) diameter", alt=("mm", lambda x: x * 1000))
    A_throat: float = q("m^2", "TOTAL nozzle throat area, all nozzles",
                        alt=("mm^2", lambda x: x * 1e6))
    A_exit: float = q("m^2", "TOTAL nozzle exit area, all nozzles",
                      alt=("mm^2", lambda x: x * 1e6))
    n_nozzles: int = q("-", "number of stator nozzles")
    n_blades: int = q("-", "number of rotor blades")
    alpha_1_deg: float = q("deg", "nozzle absolute exit angle from TANGENTIAL")
    beta_2_deg: float = q("deg", "rotor relative exit angle from TANGENTIAL")
    blade_height: float = q("m", "rotor blade height", alt=("mm", lambda x: x * 1000))
    blade_chord: float = q("m", "rotor blade chord", alt=("mm", lambda x: x * 1000))
    admission_fraction: float = q("-", "fraction of the annulus fed by nozzles")
    tip_clearance: float = q("m", "rotor tip gap", alt=("mm", lambda x: x * 1000))
    source: str = q("", "measured, or sized by which run", default="unspecified")

    @property
    def blade_passing_events_per_rev(self) -> int:
        """What an audio blade-passing tone divides by to give shaft speed.

        The rotor blade count.  Verified against OT-1, whose R2S_2025 notebook
        used `N = f * 60 / 18` with an 18-blade rotor.
        """
        return self.n_blades


@component
class TurbineCalibration:
    """Every coefficient the forward model does not derive from geometry.

    The defaults are LITERATURE-RANGE PLACEHOLDERS, not fitted values.  Nothing
    in this package treats them as validated; `otp.compare.calibrate` exists to
    replace them with numbers fitted to OT-1/OT-2 and to stamp `source`.
    """
    nozzle_cd: float = q("-", "nozzle throat discharge coefficient", default=0.97)
    nozzle_velocity_coeff: float = q("-", "phi_n, actual/ideal nozzle exit velocity",
                                     default=0.85)
    rotor_velocity_coeff: float = q("-", "psi_r, w_2/w_1 across the rotor", default=0.80)
    # Tip leakage is the largest single loss on both OT rigs.  The classic result
    # is that the leaked fraction is approximately the gap/height ratio, so
    # k_leakage = 1.0 reproduces turboRocket's phi_l = 0.199 at OT-2's
    # delta_r/h = 2.0/11.18 = 0.179.
    k_leakage: float = q("-", "leakage loss per unit tip-gap/blade-height", default=1.0)
    # Stodola-family windage on the unadmitted arc; the coefficient is O(0.01).
    k_windage: float = q("-", "inactive-arc windage coefficient", default=0.0105)
    # Stenning sector filling/emptying; the coefficient is O(0.05).
    k_sector: float = q("-", "sector filling/emptying loss coefficient", default=0.05)
    k_disc_friction: float = q("-", "disc friction coefficient", default=0.02)
    mechanical_power: float = q("W", "bearing + seal drag removed from shaft power",
                                default=0.0)
    source: str = q("", "where these coefficients came from",
                    default="uncalibrated literature-range placeholders")


@component
class TurbinePerformance:
    gas: GasState
    rpm: float = q("RPM", "shaft speed")
    p_exit_bar: float = q("bar", "static pressure the nozzle expands to")
    omega: float = q("rad/s", "shaft speed")
    mdot: float = q("kg/s", "gas mass flow through the nozzles")
    choked: bool = q("-", "nozzle throat is choked")
    u: float = q("m/s", "mean blade speed")
    c_is: float = q("m/s", "isentropic spouting velocity")
    c_1: float = q("m/s", "nozzle absolute exit velocity")
    w_1: float = q("m/s", "rotor inlet relative velocity")
    w_2: float = q("m/s", "rotor exit relative velocity")
    c_2: float = q("m/s", "rotor absolute exit velocity")
    beta_1_deg: float = q("deg", "rotor inlet relative flow angle from tangential")
    u_over_c_is: float = q("-", "blade speed ratio")
    dh_euler: float = q("J/kg", "Euler specific work")
    dh_is: float = q("J/kg", "isentropic available work")
    P_euler: float = q("W", "gross Euler power", alt=("kW", lambda x: x / 1000))
    P_leakage: float = q("W", "tip clearance leakage loss", alt=("kW", lambda x: x / 1000))
    P_windage: float = q("W", "inactive-arc windage loss", alt=("kW", lambda x: x / 1000))
    P_sector: float = q("W", "sector filling/emptying loss", alt=("kW", lambda x: x / 1000))
    P_disc_friction: float = q("W", "disc friction loss", alt=("kW", lambda x: x / 1000))
    P_mechanical: float = q("W", "bearing/seal drag", alt=("kW", lambda x: x / 1000))
    P_shaft: float = q("W", "NET shaft power delivered", alt=("kW", lambda x: x / 1000))
    torque: float = q("N*m", "net shaft torque")
    eta_isentropic: float = q("-", "net shaft power / isentropic available power")
    eta_blade: float = q("-", "Euler power / isentropic available power")
    T_exit_K: float = q("K", "estimated static exit temperature",
                        alt=("C", lambda x: x + K_TO_C))
    rho_exit: float = q("kg/m^3", "estimated static exit density")
    blade_passing_hz: float = q("Hz", "rotor blade passing frequency at this speed")
    warnings: tuple = q("", "regime checks that did not pass", default=())
