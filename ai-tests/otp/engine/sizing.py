"""Engine sizing -- ported unchanged from exploration/inducer_prelim.ipynb."""

import numpy as np

from ..core.units import BAR, g
from .records import (ChamberGeometry, EngineCalibration, EngineChoices,
                      EngineRequirements, InjectorChoices, InjectorGeometry, PropCEA)


def size_engine(engine_req: EngineRequirements, engine_choices: EngineChoices,
                injector_choices: InjectorChoices, engine_calib: EngineCalibration,
                propcea: PropCEA) -> tuple[ChamberGeometry, InjectorGeometry]:
    # For low pressure engines, frozen-at-throat properties are assumed hereon.
    # Refer to notes on non-equilibrium flows if this assumption is not valid.
    cea = propcea.cea
    eps = cea.get_eps_at_PcOvPe(
        Pc=engine_choices.p_c_bar, MR=engine_choices.OF,
        PcOvPe=engine_choices.p_c_bar / engine_choices.p_e_bar, frozen=1, frozenAtThroat=1)
    cstar = cea.get_Cstar(Pc=engine_choices.p_c_bar,
                          MR=engine_choices.OF) * engine_calib.eff_cstar
    isp_ideal = cea.estimate_Ambient_Isp(
        Pc=engine_choices.p_c_bar, MR=engine_choices.OF, eps=eps,
        Pamb=engine_req.p_a_bar)[0] * engine_calib.eff_cstar

    mdot = engine_req.F / (isp_ideal * g)               # at the design point
    A_t = mdot * cstar / (engine_choices.p_c_bar * BAR)
    d_t = np.sqrt(4 * A_t / np.pi)
    chamber = ChamberGeometry(eps=eps, d_t=d_t, d_e=d_t * np.sqrt(eps))

    mdot_fuel = mdot / (1 + engine_choices.OF)
    mdot_ox = mdot * engine_choices.OF / (1 + engine_choices.OF)

    p_preinj = engine_choices.p_c_bar + injector_choices.inj_dp
    fuel_preinj = propcea.fuel.at(p_preinj, propcea.fuel.t_tank)
    ox_preinj = propcea.oxidizer.at(p_preinj, propcea.oxidizer.t_tank)

    injector = InjectorGeometry(
        A_fuelinj=mdot_fuel / (injector_choices.fuel_cd * np.sqrt(
            2 * BAR * injector_choices.inj_dp * fuel_preinj.density)),
        A_oxinj=mdot_ox / (injector_choices.ox_cd * np.sqrt(
            2 * BAR * injector_choices.inj_dp * ox_preinj.density)),
        fuel_cd=injector_choices.fuel_cd, ox_cd=injector_choices.ox_cd)

    return chamber, injector
