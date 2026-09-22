from pyfluids import Fluid, FluidsList, Input
from dataclasses import field

from ..core.component import component, q
from ..core.units import K_TO_C

@component
class Propellant:
    cea_name: str = q("", "RocketCEA name") 
    # E.g. LOX / RP-1 / Isopropanol / CH4 / Kerosene / NitrousOxide
    # https://rocketcea.readthedocs.io/en/latest/propellants.html
    coolprop_fluid: FluidsList
    t_tank: float = q("K", "tank temperature", alt=("C", lambda x: x + K_TO_C))
    p_tank: float = q("Pa", "tank pressure", alt=("bar", lambda x: x * 1e-5))
    fluid: Fluid = field(init=False)

    def __post_init__(self):
        self.fluid = Fluid(self.coolprop_fluid)
        self.tank_fluid = self.fluid.with_state(Input.pressure(self.p_tank), Input.temperature(self.t_tank+K_TO_C))
