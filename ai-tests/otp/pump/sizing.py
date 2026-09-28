"""Pump sizing -- ported from exploration/inducer_prelim.ipynb.

Behaviour change: the notebook printed a message and then fell through with
`d_2` unbound when the root solve failed, raising UnboundLocalError from a line
that had nothing to do with the real problem.  It now raises PumpSizingError.
"""

import numpy as np
import scipy.optimize as opt

from ..core.units import g, rpm_to_rad
from .curves import barske_head, lock_head
from .records import PumpChoices, PumpGeometry, PumpRequirements


class PumpSizingError(RuntimeError):
    """The requested head could not be reached with the given choices."""


def size_pump(pump_req: PumpRequirements, pump_choices: PumpChoices,
              sizing_method: str) -> PumpGeometry:
    d_inlet = np.sqrt(4 * pump_req.Q_req / (np.pi * pump_choices.v_inlet))
    d_1 = d_inlet * pump_choices.d1_over_d0
    omega = rpm_to_rad(pump_choices.rpm)

    def geometry_from_d2(d_2) -> PumpGeometry:
        u_2 = omega * d_2 / 2
        v_3 = pump_choices.flow_coeff_throat * u_2
        d_throat = np.sqrt(4 * pump_req.Q_req / (np.pi * v_3))
        d_outlet = 2 * d_throat  # Arbitrary expansion ratio, does not matter too much
        s_ax = min(0.00075, d_2 / 100)
        b_1 = d_1 / 4
        b_2 = max(b_1 * d_1 / d_2, s_ax)
        b_3 = b_2 + 2 * s_ax
        d_3 = d_2 + max(b_3, 0.0025)
        return PumpGeometry(d_inlet=d_inlet, d_1=d_1, d_2=d_2, d_3=d_3, b_1=b_1,
                            b_2=b_2, b_3=b_3, s_ax=s_ax, d_throat=d_throat,
                            d_outlet=d_outlet, n_blades=pump_choices.n_blades)

    # Head coefficient ~1.4 gives the bracket a sane upper end.
    d_2_estimate = (2 / omega) * np.sqrt(2 * g * pump_req.H_req / 1.4)

    if sizing_method == "barske_simple":
        # Closed form; neglects the outlet dynamic head.
        d_2 = np.sqrt((8 * g * pump_req.H_req / omega ** 2 + d_1 ** 2)
                      / (1 + pump_choices.coeff_p))
        return geometry_from_d2(d_2)

    if sizing_method == "barske":
        def residual(x):
            return pump_req.H_req - barske_head(
                geometry_from_d2(x), pump_req.Q_req, omega, pump_choices.coeff_p)[0]
    elif sizing_method == "lock":
        def residual(x):
            return pump_req.H_req - lock_head(
                geometry_from_d2(x), pump_req.Q_req, omega, pump_choices.n_blades)[0]
    else:
        raise ValueError(f"Invalid sizing method: {sizing_method}")

    try:
        d_2 = opt.toms748(residual, 1.1 * d_1, 4 * d_2_estimate)   # size to TOTAL head
    except Exception as exc:
        raise PumpSizingError(
            f"{sizing_method}: no impeller diameter in "
            f"[{1.1 * d_1 * 1e3:.1f}, {4 * d_2_estimate * 1e3:.1f}] mm delivers "
            f"H_req={pump_req.H_req:.1f} m at {pump_choices.rpm:.0f} rpm and "
            f"Q={pump_req.Q_req * 1e3:.3f} L/s (usually Q too high for the speed)") from exc

    return geometry_from_d2(d_2)
