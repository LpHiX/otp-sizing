import numpy as np
from pyfluids import Fluid, FluidsList, Input

from .records import ChamberGeometry, EngineRequirements, EngineChoices, InjectorChoices, EngineCalibration, InjectorGeometry, PropCEA
from ..core.units import BAR, g, K_TO_C

def size_engine(engine_req: EngineRequirements, engine_choices: EngineChoices, injector_choices: InjectorChoices, engine_calib: EngineCalibration, propcea: PropCEA) -> tuple[ChamberGeometry, InjectorGeometry]:

    # With low pressure engines, I will hereon assume frozen at throat properties. Refer to notes on non equilibrium flows if this assumption is not valid.
    eps = propcea.cea.get_eps_at_PcOvPe(Pc=engine_choices.p_c, MR=engine_choices.OF, PcOvPe=engine_choices.p_c/engine_choices.p_e, frozen=1, frozenAtThroat=1)
    cstar = propcea.cea.get_Cstar(Pc=engine_choices.p_c, MR=engine_choices.OF) * engine_calib.eff_cstar
    isp_ideal = propcea.cea.estimate_Ambient_Isp(Pc=engine_choices.p_c, MR=engine_choices.OF, eps=eps, Pamb=engine_req.p_a)[0] * engine_calib.eff_cstar

    # mdot for at the design point
    mdot = engine_req.F / (isp_ideal * g)
    A_t = mdot * cstar / (engine_choices.p_c)
    d_t = np.sqrt(4 * A_t / np.pi)  
    d_e = d_t * np.sqrt(eps) 

    chamber = ChamberGeometry(eps=eps, d_t=d_t, d_e=d_e)

    mdot_fuel = mdot / (1 + engine_choices.OF)
    mdot_ox = mdot * engine_choices.OF / (1 + engine_choices.OF)


    p_preinj = engine_choices.p_c + injector_choices.inj_dp
    fuel_preinj = propcea.fuel.fluid.with_state(Input.pressure(p_preinj), Input.temperature(propcea.fuel.t_tank + K_TO_C))
    ox_preinj = propcea.oxidizer.fluid.with_state(Input.pressure(p_preinj), Input.temperature(propcea.oxidizer.t_tank + K_TO_C))

    A_fuelinj = mdot_fuel / (injector_choices.fuel_cd * np.sqrt(2 * injector_choices.inj_dp * fuel_preinj.density))
    A_oxinj = mdot_ox / (injector_choices.ox_cd * np.sqrt(2 * injector_choices.inj_dp * ox_preinj.density))

    injector = InjectorGeometry(A_fuelinj=A_fuelinj, A_oxinj=A_oxinj, fuel_cd=injector_choices.fuel_cd, ox_cd=injector_choices.ox_cd)

    return chamber, injector