"""Engine records -- ported unchanged from exploration/inducer_prelim.ipynb."""

import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj

from ..core.component import component, q
from ..core.states import Propellant
from ..core.units import BAR, K_TO_C, P_A_BAR, g


@component
class EngineRequirements:
    F: float = q("N", "thrust")
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


@component
class PropCEA:
    oxidizer: Propellant
    fuel: Propellant

    def __post_init__(self):
        self.cea = CEA_Obj(
            oxName=self.oxidizer.cea_name,
            fuelName=self.fuel.cea_name,
            isp_units="sec",
            cstar_units="m/s",
            pressure_units="Bar",
            temperature_units="K",
            sonic_velocity_units="m/s",
            enthalpy_units="J/g",
            density_units="kg/m^3",
            specific_heat_units="J/kg-K",
            viscosity_units="centipoise",
            thermal_cond_units="W/cm-degC",
            make_debug_prints=False)


@component
class Engine:
    chamber: ChamberGeometry
    injector: InjectorGeometry
    propcea: PropCEA
    calib: EngineCalibration

    def point_performance(self, p_c_bar: float, p_a_bar: float, OF: float) -> EnginePerformance:
        cea = self.propcea.cea
        isp_vac, cstar_theory, t_c = cea.get_IvacCstrTc(
            Pc=p_c_bar, MR=OF, eps=self.chamber.eps, frozen=1, frozenAtThroat=1)
        isp_sl = cea.estimate_Ambient_Isp(
            Pc=p_c_bar, MR=OF, eps=self.chamber.eps,
            Pamb=P_A_BAR)[0] * self.calib.eff_cstar
        isp_amb = cea.estimate_Ambient_Isp(
            Pc=p_c_bar, MR=OF, eps=self.chamber.eps,
            Pamb=p_a_bar)[0] * self.calib.eff_cstar

        cstar = cstar_theory * self.calib.eff_cstar
        A_t = self.chamber.d_t ** 2 * np.pi / 4
        mdot = A_t * p_c_bar * BAR / cstar
        mdot_ox = mdot * OF / (1 + OF)
        mdot_fuel = mdot / (1 + OF)

        F_amb, F_vac, F_sl = (mdot * isp * g for isp in (isp_amb, isp_vac, isp_sl))
        Cf_amb, Cf_vac, Cf_sl = (F / (p_c_bar * BAR * A_t) for F in (F_amb, F_vac, F_sl))

        # Guess parameters for the first loop; add an iteration if these are not good enough.
        fuel_preinj = self.propcea.fuel.at(p_c_bar, self.propcea.fuel.t_tank)
        ox_preinj = self.propcea.oxidizer.at(p_c_bar, self.propcea.oxidizer.t_tank)

        p_fuel_injdp = ((mdot_fuel / self.injector.A_fuelinj / self.injector.fuel_cd) ** 2
                        / (2 * fuel_preinj.density) / BAR)
        p_ox_injdp = ((mdot_ox / self.injector.A_oxinj / self.injector.ox_cd) ** 2
                      / (2 * ox_preinj.density) / BAR)

        return EnginePerformance(
            fuel=self.propcea.fuel, oxidizer=self.propcea.oxidizer,
            p_c_bar=p_c_bar, p_a_bar=p_a_bar, OF=OF,
            F_amb=F_amb, F_vac=F_vac, F_sl=F_sl,
            isp_amb=isp_amb, isp_vac=isp_vac, isp_sl=isp_sl,
            cstar=cstar, T_c=t_c, mdot=mdot, mdot_ox=mdot_ox, mdot_fuel=mdot_fuel,
            Cf_amb=Cf_amb, Cf_vac=Cf_vac, Cf_sl=Cf_sl,
            p_fuel_injdp=p_fuel_injdp, p_ox_injdp=p_ox_injdp,
            p_fuel_preinj=p_c_bar + p_fuel_injdp, p_ox_preinj=p_c_bar + p_ox_injdp,
            stiffness_fuel=p_fuel_injdp / p_c_bar, stiffness_ox=p_ox_injdp / p_c_bar)
