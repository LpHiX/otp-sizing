"""Constants and unit helpers shared by every module.

Convention used throughout the package, chosen to match the existing notebook:
  * pressures passed around as `*_bar` are ABSOLUTE bar unless the name says
    `_barg`; SI pressures are Pa.
  * temperatures in kelvin unless the name says `_C`.
  * everything else SI.
"""

g = 9.80665          # m/s^2
K_TO_C = -273.15     # add to K to get degC
P_A_BAR = 1.01325    # standard sea-level ambient, bar(a)
BAR = 1e5            # Pa per bar


def barg_to_bara(p_barg: float, p_atm_bar: float = P_A_BAR) -> float:
    return p_barg + p_atm_bar


def rpm_to_rad(rpm: float) -> float:
    return rpm * 2.0 * 3.141592653589793 / 60.0


def rad_to_rpm(omega: float) -> float:
    return omega * 60.0 / (2.0 * 3.141592653589793)
