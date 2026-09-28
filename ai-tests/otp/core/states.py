"""Boundary-condition records.

Three separate types, because they answer three different questions:

  Propellant   a tank of something, with a CEA name -- what the SIZING side wants.
  LiquidState  the liquid at one station, already reduced to numbers -- what a
               pump/inducer forward calculation and a measured point both want.
  GasState     the gas at the turbine inlet, ditto.

`LiquidState` and `GasState` deliberately hold plain floats rather than a live
CoolProp object.  A measured point has a density and a vapour pressure and
nothing else; forcing it through an equation of state it never came from would
invent precision.  `LiquidState.from_propellant` builds one when you do have an
equation of state.
"""

from pyfluids import Fluid, FluidsList, Input

from .component import component, q
from .units import BAR, K_TO_C


@component
class Propellant:
    cea_name: str = q("", "RocketCEA name")
    # E.g. LOX / RP-1 / Isopropanol / CH4 / Kerosene / NitrousOxide
    # https://rocketcea.readthedocs.io/en/latest/propellants.html
    coolprop_fluid: FluidsList = q("", "pyfluids fluid")
    t_tank: float = q("K", "tank temperature", alt=("C", lambda x: x + K_TO_C))
    p_tank: float = q("bar", "tank pressure")

    def __post_init__(self):
        self.fluid = Fluid(self.coolprop_fluid)
        self.tank_fluid = self.fluid.with_state(
            Input.pressure(self.p_tank * BAR), Input.temperature(self.t_tank + K_TO_C))

    def at(self, p_bar: float, T_K: float):
        return self.fluid.with_state(Input.pressure(p_bar * BAR),
                                     Input.temperature(T_K + K_TO_C))

    def p_sat_bar(self, T_K: float) -> float:
        return self.fluid.with_state(Input.temperature(T_K + K_TO_C),
                                     Input.quality(0)).pressure / BAR


@component
class LiquidState:
    """Liquid at one station.  `p0_bar` is TOTAL absolute pressure at that station."""
    name: str = q("", "fluid label")
    rho: float = q("kg/m^3", "density")
    nu: float = q("m^2/s", "kinematic viscosity")
    p_vap_bar: float = q("bar", "vapour pressure at T")
    p0_bar: float = q("bar", "total absolute pressure at the station")
    T_K: float = q("K", "temperature", alt=("C", lambda x: x + K_TO_C))
    source: str = q("", "where these numbers came from", default="unspecified")

    @staticmethod
    def from_propellant(prop: Propellant, p0_bar: float, T_K: float,
                        source: str = "pyfluids") -> "LiquidState":
        st = prop.at(p0_bar, T_K)
        return LiquidState(name=prop.cea_name, rho=st.density,
                           nu=st.kinematic_viscosity, p_vap_bar=prop.p_sat_bar(T_K),
                           p0_bar=p0_bar, T_K=T_K, source=source)

    @property
    def npsha_m(self) -> float:
        from .units import g
        return (self.p0_bar - self.p_vap_bar) * BAR / (self.rho * g)


@component
class GasState:
    """Gas at the TURBINE INLET -- i.e. after any gas-generator delivery losses.

    Constant-property record on purpose: a measured GG point gives p0, T0 and a
    mixture, and the property fit is a separate, stated assumption.  A reacting
    backend can produce this same record without the rest of the package caring.
    """
    p0_bar: float = q("bar", "total absolute inlet pressure")
    T0_K: float = q("K", "total inlet temperature", alt=("C", lambda x: x + K_TO_C))
    R: float = q("J/kg/K", "specific gas constant")
    gamma: float = q("-", "ratio of specific heats")
    source: str = q("", "where these numbers came from", default="unspecified")

    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)

    @property
    def rho0(self) -> float:
        return self.p0_bar * BAR / (self.R * self.T0_K)
