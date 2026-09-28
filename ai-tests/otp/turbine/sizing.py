"""Turbine sizing, verified by the forward model it will later be analysed with.

The free variable is the total nozzle throat area: diameter, speed, angles and
admission are choices the designer has already made, and throat area is what
actually buys power once they are fixed.  The solve is a root find on the same
`Turbine.point_performance` used everywhere else, so a sized turbine and an
analysed turbine can never disagree.

Sizing does not adjust the inlet pressure or the shaft speed to make the numbers
work.  If the requested power is unreachable with the given choices it raises.
"""

import numpy as np
import scipy.optimize as opt

from ..core.states import GasState
from .analysis import Turbine
from .records import (TurbineCalibration, TurbineChoices, TurbineGeometry,
                      TurbineRequirements)


class TurbineSizingError(RuntimeError):
    """The requested power cannot be produced with the given choices."""


def size_turbine(req: TurbineRequirements, choices: TurbineChoices,
                 calib: TurbineCalibration, gas: GasState, p_exit_bar: float,
                 area_ratio: float = 6.0,
                 A_throat_bounds: tuple[float, float] = (1e-7, 5e-3)) -> TurbineGeometry:
    """`area_ratio` is the nozzle exit/throat area ratio the designer has chosen."""

    def geometry_from_A(A_throat: float) -> TurbineGeometry:
        return TurbineGeometry(
            d_mean=choices.d_mean, A_throat=A_throat, A_exit=A_throat * area_ratio,
            n_nozzles=choices.n_nozzles, n_blades=choices.n_blades,
            alpha_1_deg=choices.alpha_1_deg, beta_2_deg=choices.beta_2_deg,
            blade_height=choices.blade_height, blade_chord=choices.blade_chord,
            admission_fraction=choices.admission_fraction,
            tip_clearance=choices.tip_clearance,
            source=f"sized for {req.P_req:.0f} W at {req.rpm:.0f} rpm")

    def residual(A_throat: float) -> float:
        perf = Turbine(geometry_from_A(A_throat), calib).point_performance(
            rpm=req.rpm, gas=gas, p_exit_bar=p_exit_bar)
        return perf.P_shaft - req.P_req

    lo, hi = A_throat_bounds
    r_lo, r_hi = residual(lo), residual(hi)
    if r_lo > 0:
        raise TurbineSizingError(
            f"even the smallest throat in the bracket ({lo * 1e6:.3f} mm^2) overshoots "
            f"{req.P_req:.0f} W; lower the requirement or the inlet pressure")
    if r_hi < 0:
        raise TurbineSizingError(
            f"{req.P_req:.0f} W at {req.rpm:.0f} rpm is unreachable with these choices: "
            f"even a {hi * 1e6:.0f} mm^2 throat gives only "
            f"{r_hi + req.P_req:.0f} W. Losses may exceed the Euler work at this "
            f"speed -- check u/c_is and the admission fraction")

    A = opt.brentq(residual, lo, hi, xtol=1e-12, rtol=1e-10)
    return geometry_from_A(A)


def admission_fraction_from_nozzles(n_nozzles: int, nozzle_width: float,
                                    d_mean: float) -> float:
    """Fraction of the annulus circumference actually fed by nozzles."""
    return n_nozzles * nozzle_width / (np.pi * d_mean)
