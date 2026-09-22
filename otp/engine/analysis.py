import numpy as np
from pyfluids import Input

from ..core.component import component
from ..core.units import P_A, K_TO_C, R_GAS, g
from .records import ChamberGeometry, EnginePerformance, InjectorGeometry, PropCEA, EngineCalibration

@component
class Engine:
    chamber: ChamberGeometry
    injector: InjectorGeometry
    propcea: PropCEA
    calib: EngineCalibration

    def point_performance(self, p_c: float,  p_a: float, OF: float) -> EnginePerformance:
        isp_vac, cstar_theory, T_c, mw_exit, gam_exit = self.propcea.cea.get_IvacCstrTc_exitMwGam(Pc=p_c, MR=OF, eps=self.chamber.eps, frozen=1, frozenAtThroat=1)
        isp_sl = self.propcea.cea.estimate_Ambient_Isp(Pc=p_c, MR=OF, eps=self.chamber.eps, Pamb=P_A)[0] * self.calib.eff_cstar
        isp_amb = self.propcea.cea.estimate_Ambient_Isp(Pc=p_c, MR=OF, eps=self.chamber.eps, Pamb=p_a)[0] * self.calib.eff_cstar

        R_exit = R_GAS / mw_exit

        cstar = cstar_theory * self.calib.eff_cstar
        mdot = self.chamber.d_t**2 * np.pi / 4 * p_c / cstar

        mdot_ox = mdot * OF / (1 + OF)
        mdot_fuel = mdot / (1 + OF)

        F_amb = mdot * isp_amb * g
        F_vac = mdot * isp_vac * g
        F_sl = mdot * isp_sl * g

        Cf_amb = F_amb / (p_c * self.chamber.d_t**2 * np.pi / 4)
        Cf_vac = F_vac / (p_c * self.chamber.d_t**2 * np.pi / 4)
        Cf_sl = F_sl / (p_c * self.chamber.d_t**2 * np.pi / 4)

        # Guess parameters for first loop, if they aren't good enough, add a loop to iterate to a better solution

        fuel_preinj = self.propcea.fuel.fluid.with_state(Input.pressure(p_c), Input.temperature(self.propcea.fuel.t_tank + K_TO_C))
        ox_preinj = self.propcea.oxidizer.fluid.with_state(Input.pressure(p_c), Input.temperature(self.propcea.oxidizer.t_tank + K_TO_C))
        
        p_fuel_injdp = (mdot_fuel / self.injector.A_fuelinj / self.injector.fuel_cd)**2 / (2 * fuel_preinj.density)
        p_ox_injdp = (mdot_ox / self.injector.A_oxinj / self.injector.ox_cd)**2 / (2 * ox_preinj.density)

        p_fuel_preinj = p_c + p_fuel_injdp
        p_ox_preinj = p_c + p_ox_injdp

        stiffness_fuel = p_fuel_injdp / p_c
        stiffness_ox = p_ox_injdp / p_c

        

        return EnginePerformance(
            fuel=self.propcea.fuel,
            oxidizer=self.propcea.oxidizer,
            p_c=p_c,
            p_a=p_a,
            OF=OF,
            F_amb=F_amb,
            F_vac=F_vac,
            F_sl=F_sl,
            isp_amb=isp_amb,
            isp_vac=isp_vac,
            isp_sl=isp_sl,
            cstar=cstar,
            mdot=mdot,
            mdot_ox=mdot_ox,
            mdot_fuel=mdot_fuel,
            Cf_amb=Cf_amb,
            Cf_vac=Cf_vac,
            Cf_sl=Cf_sl,
            p_fuel_injdp=p_fuel_injdp,
            p_ox_injdp=p_ox_injdp,
            p_fuel_preinj=p_fuel_preinj,
            p_ox_preinj=p_ox_preinj,
            stiffness_fuel=stiffness_fuel,
            stiffness_ox=stiffness_ox,
            T_c=T_c,
            mw_exit=mw_exit,
            gam_exit=gam_exit,
            R_exit=R_exit)