import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
from pyfluids import Fluid, FluidsList, Input


from ..core.component import component, q, group
from ..propellants.propellant import Propellant
from ..core.units import BAR, K_TO_C, P_A, g

@component
class EngineRequirements:
    F:  float = q("N", "thrust")
    p_a: float = q("Pa", "ambient pressure", alt=("bar", lambda x: x * 1e-5))

@component
class EngineChoices:
    OF: float = q("-", "mixture ratio")
    p_c: float = q("Pa", "chamber pressure", alt=("bar", lambda x: x * 1e-5))
    p_e: float = q("Pa", "exit pressure", alt=("bar", lambda x: x * 1e-5))
    perc_film: float = q("-", "percentage of film cooling (of core total core mdot)")
    fuel_name: str = q("-", "fuel name", default="RP-1")
    oxidizer_name: str = q("-", "oxidizer name", default="LOX")

@component
class InjectorChoices:
    inj_dp: float = q("Pa", "injector pressure drop", alt=("bar", lambda x: x * 1e-5))
    fuel_cd: float = q("-", "fuel injector discharge coefficient")
    ox_cd: float = q("-", "oxidizer injector discharge coefficient")

@component
class ChamberGeometry:
    eps: float = q("-", "area ratio")
    d_t: float = q("m", "throat diameter", alt=("mm", lambda x: x * 1000))
    d_e: float = q("m", "exit diameter", alt=("mm", lambda x: x * 1000))

@component
class InjectorGeometry:
    A_fuelinj: float = q("m^2", "fuel injector area", alt=("mm^2", lambda x: x * 1e6))
    A_oxinj: float = q("m^2", "oxidizer injector area", alt=("mm^2", lambda x: x * 1e6))
    fuel_cd: float = q("-", "fuel injector discharge coefficient")
    ox_cd: float = q("-", "oxidizer injector discharge coefficient")


@component
class EngineCalibration:
    eff_cstar: float = q("-", "c* efficiency", default=0.95)

@component
class EnginePerformance:
    with group("Inputs"):
        fuel: Propellant = q("fuel")
        oxidizer: Propellant = q("oxidizer")
        p_c: float = q("Pa", "chamber pressure", alt=("bar", lambda x: x * 1e-5))
        p_a: float = q("Pa", "ambient pressure", alt=("bar", lambda x: x * 1e-5))
        OF: float = q("-", "mixture ratio")
    with group("Thrust and ISP"):
        F_amb: float = q("N", "delivered thrust", alt=("kN", lambda x: x / 1000))
        F_vac: float = q("N", "delivered thrust in vacuum", alt=("kN", lambda x: x / 1000))
        F_sl: float = q("N", "delivered thrust at sea level", alt=("kN", lambda x: x / 1000))
        isp_amb: float = q("s", "specific impulse at ambient pressure")
        isp_vac: float = q("s", "specific impulse")
        isp_sl: float = q("s", "specific impulse at sea level")
    with group("Mass flow rates"):
        cstar: float = q("m/s", "characteristic velocity")
        mdot: float = q("kg/s", "total mass flow")
        mdot_ox: float = q("kg/s", "oxidiser mass flow")
        mdot_fuel: float = q("kg/s", "fuel mass flow")
        Cf_amb: float = q("-", "thrust coefficient at ambient pressure")
        Cf_vac: float = q("-", "thrust coefficient in vacuum")
        Cf_sl: float = q("-", "thrust coefficient at sea level")
    with group("Injector pressures"):
        p_fuel_injdp: float = q("Pa", "fuel injector pressure drop", alt=("bar", lambda x: x * 1e-5))
        p_ox_injdp: float = q("Pa", "oxidizer injector pressure drop", alt=("bar", lambda x: x * 1e-5))
        p_fuel_preinj: float = q("Pa", "fuel pre-injector pressure", alt=("bar", lambda x: x * 1e-5))
        p_ox_preinj: float = q("Pa", "oxidizer pre-injector pressure", alt=("bar", lambda x: x * 1e-5))
        stiffness_fuel: float = q("-", "fuel injector stiffness")
        stiffness_ox: float = q("-", "oxidizer injector stiffness")
    with group("Gas properties"):
        T_c: float = q("K", "chamber temperature", alt=("C", lambda x: x + K_TO_C))
        mw_exit: float = q("g/mol", "molecular weight of exhaust")
        gam_exit: float = q("-", "ratio of specific heats of exhaust")
        R_exit: float = q("J/kg-K", "specific gas constant of exhaust")

@component
class PropCEA:
    oxidizer: Propellant
    fuel: Propellant

    def __post_init__(self):
        self.cea = CEA_Obj(
            oxName = self.oxidizer.cea_name,
            fuelName = self.fuel.cea_name,
            isp_units='sec',
            cstar_units = 'm/s',
            pressure_units='Pa',
            temperature_units='K',
            sonic_velocity_units='m/s',
            enthalpy_units='J/g',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise', # stored value in pa-s
            thermal_cond_units='W/cm-degC', # stored value in W/m-K
            # fac_CR=self.cr,
            make_debug_prints=False)
