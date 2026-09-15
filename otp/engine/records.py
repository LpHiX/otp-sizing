import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
from pyfluids import Fluid, FluidsList, Input


from ..core.component import component, q
from ..propellants.propellant import Propellant
from ..core.units import BAR, K_TO_C, P_A_BAR, g

@component
class EngineRequirements:
    F:  float = q("N", "thrust")
    p_a_bar: float = q("bar", "ambient pressure")

@component
class EngineChoices:
    OF: float = q("-", "mixture ratio")
    p_c_bar: float = q("bar", "chamber pressure")
    p_e_bar: float = q("bar", "exit pressure")
    perc_film: float = q("-", "percentage of film cooling (of core total core mdot)")
    fuel_name: str = q("-", "fuel name", default="RP-1")
    oxidizer_name: str = q("-", "oxidizer name", default="LOX")

@component
class InjectorChoices:
    inj_dp: float = q("bar", "injector pressure drop")
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
    fuel: Propellant
    oxidizer: Propellant
    p_c_bar: float = q("bar", "chamber pressure")
    p_a_bar: float = q("bar", "ambient pressure")
    OF: float = q("-", "mixture ratio")
    F_amb: float = q("N", "delivered thrust", alt=("kN", lambda x: x / 1000))
    F_vac: float = q("N", "delivered thrust in vacuum", alt=("kN", lambda x: x / 1000))
    F_sl: float = q("N", "delivered thrust at sea level", alt=("kN", lambda x: x / 1000))
    isp_amb: float = q("s", "specific impulse at ambient pressure")
    isp_vac: float = q("s", "specific impulse")
    isp_sl: float = q("s", "specific impulse at sea level")
    cstar: float = q("m/s", "characteristic velocity")
    T_c: float = q("K", "chamber temperature", alt=("C", lambda x: x + K_TO_C))
    mdot: float = q("kg/s", "total mass flow")
    mdot_ox: float = q("kg/s", "oxidiser mass flow")
    mdot_fuel: float = q("kg/s", "fuel mass flow")
    Cf_amb: float = q("-", "thrust coefficient at ambient pressure")
    Cf_vac: float = q("-", "thrust coefficient in vacuum")
    Cf_sl: float = q("-", "thrust coefficient at sea level")
    p_fuel_injdp: float = q("bar", "fuel injector pressure drop")
    p_ox_injdp: float = q("bar", "oxidizer injector pressure drop")
    p_fuel_preinj: float = q("bar", "fuel pre-injector pressure")
    p_ox_preinj: float = q("bar", "oxidizer pre-injector pressure")
    stiffness_fuel: float = q("-", "fuel injector stiffness")
    stiffness_ox: float = q("-", "oxidizer injector stiffness")
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
            pressure_units='Bar',
            temperature_units='K',
            sonic_velocity_units='m/s',
            enthalpy_units='J/g',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise', # stored value in pa-s
            thermal_cond_units='W/cm-degC', # stored value in W/m-K
            # fac_CR=self.cr,
            make_debug_prints=False)
