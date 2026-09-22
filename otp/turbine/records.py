import numpy as np
from typing import List

from ..core.component import component, q, group
from ..pump.records import PumpPerformance
from ..engine.records import PropCEA

@component
class TurbineRequirements:
    P_req: float = q("W", "required shaft power", alt=("kW", lambda x: x / 1000))
    rpm: float = q("RPM", "shaft rotational speed")

    @staticmethod
    def from_pump_perfs(pump_perfs: List[PumpPerformance]) -> TurbineRequirements:
        first_rpm = pump_perfs[0].rpm
        for p in pump_perfs:
            if p.rpm != first_rpm:
                raise ValueError("Input pumps do not have the same RPM")
        return TurbineRequirements(sum(p.P_shaft for p in pump_perfs), pump_perfs[0].rpm)


@component
class TurbineChoices:
    d_mean: float = q("m", "mean turbine diameter", alt=("mm", lambda x: x * 1000))
    beta_deg: float = q("degree", "nozzle angle to the circumferential direction")
    doa: float = q("-", "arc degree of admission")
    n_nozzles: float = q("-", "number of stator nozzles")

@component
class TurbineGeometry:
    d_mean: float = q("m", "mean turbine diameter", alt=("mm", lambda x: x * 1000))
    beta_deg: float = q("degree", "nozzle angle to the circumferential direction")
    doa: float = q("-", "arc degree of admission")
    n_nozzles: float = q("-", "number of stator nozzles")
    blade_height: float = q("m", "blade height", alt=("mm", lambda x: x * 1000))
    A_throat_total: float = q("m^2", "total area of stator nozzle throat")
    A_3_total: float = q("m^2", "total area of stator nozzle exit")
    nozzle_throat_length: float = q("m", "length of stator nozzle throat")
    nozzle_exit_length: float = q("m", "length of stator nozzle exit")

@component
class TurbineInletGas:
    p01: float = q("Pa", "stagnation pressure", alt=("bar", lambda x: x * 1e-5))
    T01: float = q("K", "stagnation temperature")
    R: float = q("J/kgK", "gas constant")
    gamma: float = q("-", "heat capacity ratio")
    model: PropCEA | None
    OF: float = q("-", "OF ratio if applicable")

@component
class TurbinePerformance:
    with group("Inputs"):
        gas: TurbineInletGas = q("inlet gas state")
        rpm: float = q("RPM", "shaft rotational speed")
        omega: float = q("rad/s", "shaft rotational speed in rad/s")
        u: float = q("m/s", "mean blade speed")
        mdot: float = q("kg/s", "turbine mass flow")
        p_amb: float = q("Pa", "turbine exhaust ambient pressure", alt=("bar", lambda x: x * 1e-5))

    with group("Thermodynamics"):
        eps: float = q("-", "nozzle exit over throat area ratio")
        p_ratio: float = q("-", "nozzle stator pressure ratio")
        p01: float = q("Pa", "stagnation pressure at turbine inlet", alt=("bar", lambda x: x * 1e-5))
        T01: float = q("K", "stagnation temperature at turbine inlet")
        p_throat: float = q("Pa", "stagnation pressure at nozzle throat", alt=("bar", lambda x: x * 1e-5))
        T_throat: float = q("K", "stagnation temperature at nozzle throat")
        a_throat: float = q("m/s", "sonic velocity at nozzle throat")
        rho_throat: float = q("kg/m^3", "density at nozzle throat")
        p3: float = q("Pa", "stagnation pressure at nozzle exit", alt=("bar", lambda x: x * 1e-5))
        T3: float = q("K", "stagnation temperature at nozzle exit")
        M3: float = q("-", "nozzle exit Mach number")
        Mr: float = q("-", "rotor inlet relative Mach number")
        a_3: float = q("m/s", "sonic velocity at nozzle exit")
        rho_3: float = q("kg/m^3", "density at nozzle exit")

    with group ("Velocity Triangles"):
        phi_n: float = q("-", "nozzle velocity coefficient")
        phi_r: float = q("-", "rotor velocity coefficient")
        c3_ideal: float = q("m/s", "isentropic nozzle exit velocity")
        c3_real: float = q("m/s", "real nozzle exit velocity")
        c3u_real: float = q("m/s", "nozzle exit tangential velocity")
        c3m_real: float = q("m/s", "nozzle exit meridional velocity")
        w3_real: float = q("m/s", "rotor inlet relative velocity")
        w3u_real: float = q("m/s", "rotor inlet relative tangential velocity")
        w4_real: float = q("m/s", "rotor exit relative velocity")
        w4u_real: float = q("m/s", "rotor exit relative tangential velocity")
        c4u_real: float = q("m/s", "rotor exit absolute tangential velocity")

    with group ("Work and Power"):
        deltah_useful: float = q("J/kg", "useful specific work", alt=("kJ/kg", lambda x: x / 1000))
        deltah_is_ta: float = q("J/kg", "isentropic total-to-ambient specific work", alt=("kJ/kg", lambda x: x / 1000))
        c_0: float = q("m/s", "isentropic spouting velocity")
        u_over_c0: float = q("-", "blade-jet speed ratio")
        eta_is_ta: float = q("-", "total-to-ambient isentropic efficiency")
        P_euler: float = q("W", "Euler power before windage", alt=("kW", lambda x: x / 1000))
        P_windage: float = q("W", "partial-admission windage loss", alt=("kW", lambda x: x / 1000))
        P_shaft: float = q("W", "turbine shaft power", alt=("kW", lambda x: x / 1000))
        torque: float = q("N*m", "turbine torque")

    # Consistency check between the choked throat and station 3 continuity
    mdot_continuity: float = q("kg/s", "mass flow implied by blade height and the station 3 state")
    continuity_ratio: float = q("-", "mdot_continuity over mdot, 1.0 if geometry is self-consistent")