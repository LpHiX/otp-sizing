import numpy as np

from ..core.component import component, q, group
from ..propellants.propellant import Propellant
from ..engine.records import EnginePerformance
from ..core.units import g, K_TO_C

@component
class PumpRequirements:
    propellant: Propellant
    Q_req: float = q("m^3/s", "required pump flow rate", alt=("L/s", lambda x: x * 1000))
    H_req: float = q("m", "required pump head")

    @staticmethod
    def from_engine_perf(engine_perf: EnginePerformance, side: str) -> PumpRequirements:
        if side == "ox":
            propellant = engine_perf.oxidizer
            mdot = engine_perf.mdot_ox
            p_discharge_bar = engine_perf.p_ox_preinj
        elif side == "fuel":
            propellant = engine_perf.fuel
            mdot = engine_perf.mdot_fuel
            p_discharge_bar = engine_perf.p_fuel_preinj
        else:
            raise ValueError(f"Invalid pump side: {side}")

        Q_req = mdot / propellant.tank_fluid.density
        H_req = (p_discharge_bar - propellant.p_tank) * 1e5 / (propellant.tank_fluid.density * g)

        return PumpRequirements(propellant=propellant, Q_req=Q_req, H_req=H_req)

@component
class PumpChoices:
    rpm: float = q("RPM", "pump rotational speed")
    n_blades: int = q("-", "number of impeller blades")
    v_inlet: float = q("m/s", "pump inlet velocity", default=1.0)
    d1_over_d0: float = q("-", "impeller inlet diameter over pump inlet diameter", default=1.1)
    flow_coeff_throat: float = q("-", "flow coefficient at pump throat", default=0.8)
    coeff_p: float = q("-", "Barske's phi coefficient, only used for Barske sizing method", default=0.2)

@component
class PumpGeometry:
    d_inlet: float = q("m", "Pump inlet diameter", alt=("mm", lambda x: x * 1000))
    d_1: float = q("m", "Impeller inlet diameter", alt=("mm", lambda x: x * 1000))
    d_2: float = q("m", "Impeller outlet diameter", alt=("mm", lambda x: x * 1000))
    d_3: float = q("m", "Casing diameter", alt=("mm", lambda x: x * 1000))
    b_1: float = q("m", "Impeller inlet blade height", alt=("mm", lambda x: x * 1000))
    b_2: float = q("m", "Impeller outlet blade height", alt=("mm", lambda x: x * 1000))
    b_3: float = q("m", "Casing height", alt=("mm", lambda x: x * 1000))
    s_ax: float = q("m", "Axial spacing between impeller and casing", alt=("mm", lambda x: x * 1000))
    d_throat: float = q("m", "Diffuser throat diameter", alt=("mm", lambda x: x * 1000))
    d_outlet: float = q("m", "Pump outlet diameter", alt=("mm", lambda x: x * 1000))

@component
class PumpPerformance:
    with group("Inputs"):
        propellant: Propellant = q("pumped propellant")
        Q: float = q("m^3/s", "pump flow rate", alt=("L/s", lambda x: x * 1000))
        rpm: float = q("RPM", "pump rotational speed")
        p_upstream: float = q("bar", "pump upstream pressure")
        T_upstream: float = q("K", "pump upstream temperature", alt=("C", lambda x: x + K_TO_C))
    with group("Outputs"):
        omega: float = q("rad/s", "pump rotational speed in rad/s")
        mdot: float = q("kg/s", "pump mass flow rate")
        dp: float = q("bar", "pump pressure rise")

    flow_coeff_inlet: float = q("-", "impeller inlet flow coefficient")
    flow_coeff_outlet: float = q("-", "impeller outlet flow coefficient")
    head_coeff: float = q("-", "pump head coefficient")
    with group("Head and Power"):
        H_total_real: float = q("m", "pump head")
        H_static_real: float = q("m", "pump ideal head")
        H_loss_diff: float = q("m", "pump diffuser head loss")
        eta_hydraulic: float = q("-", "pump hydraulic efficiency")
        P_shaft: float = q("W", "pump shaft power", alt=("kW", lambda x: x / 1000))
        P_hydraulic: float = q("W", "pump hydraulic power", alt=("kW", lambda x: x / 1000))
        P_disc_friction: float = q("W", "pump disc friction power", alt=("kW", lambda x: x / 1000))
        P_useful: float = q("W", "pump useful power", alt=("kW", lambda x: x / 1000))
        eta_power: float = q("-", "pump power efficiency")
        torque: float = q("N*m", "pump torque")
    with group("Velocities"):
        v_inlet: float = q("m/s", "pump inlet velocity")
        u_1: float = q("m/s", "impeller inlet blade speed")
        w_1: float = q("m/s", "impeller inlet relative velocity")
        v_1: float = q("m/s", "impeller inlet velocity")
        u_2: float = q("m/s", "impeller outlet blade speed")
        w_2: float = q("m/s", "impeller outlet relative velocity")
        v_2: float = q("m/s", "impeller outlet velocity")
        v_throat: float = q("m/s", "diffuser throat velocity")
        v_outlet: float = q("m/s", "pump outlet velocity")
    with group("Pressures"):
        p_inlet: float = q("bar", "pump inlet static pressure")
        p_1: float = q("bar", "impeller inlet static pressure")
        p_2: float = q("bar", "impeller outlet static pressure")
        p_2_total: float = q("bar", "impeller outlet total pressure")
        p_throat: float = q("bar", "diffuser throat static pressure")
        p_outlet: float = q("bar", "pump outlet total pressure")
        p_outlet_static: float = q("bar", "pump outlet static pressure")
    with group("NPSH"):
        npsh_a_upstream: float = q("m", "pump upstream NPSH available")
        npsh_r_inlet: float = q("m", "pump inlet NPSH required")
        npsh_r_throat: float = q("m", "pump throat NPSH required")
        npsh_r_ai_low: float = q("m", "pump inlet NPSH required (low lam_w)")
        npsh_r_ai_high: float = q("m", "pump inlet NPSH required (high lam_w)")
        suction_specific_speed: float = q("-", "pump suction specific speed")