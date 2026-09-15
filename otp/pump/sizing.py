import numpy as np
from .records import PumpRequirements, PumpChoices, PumpGeometry
from .analysis import barske_head, lock_head
from ..core.units import g

import scipy.optimize as opt

def size_pump(pump_req: PumpRequirements, pump_choices: PumpChoices, sizing_method: str) -> PumpGeometry:
    d_inlet = np.sqrt(4 * pump_req.Q_req / (np.pi * pump_choices.v_inlet))
    d_1 = d_inlet * pump_choices.d1_over_d0
    omega = pump_choices.rpm * 2 * np.pi / 60

    d_2 = 0 # To stop stupid unbound comment

    def geometry_from_d2(d_2) -> PumpGeometry:
        u_2 = omega * d_2 / 2
        v_3 = pump_choices.flow_coeff_throat * u_2
        d_throat = np.sqrt(4 * pump_req.Q_req / (np.pi * v_3))
        d_outlet = 2 * d_throat  # Arbitrary expansion ratio, doesn't matter too much
        s_ax = min(0.00075, d_2 / 100)
        b_1 = d_1 / 4
        b_2 = max(b_1 * d_1 / d_2, s_ax)
        b_3 = b_2 + 2 * s_ax
        d_3 = d_2 + max(b_3, 0.0025)
        return PumpGeometry(d_inlet=d_inlet, d_1=d_1, d_2=d_2, d_3=d_3, b_1=b_1, b_2=b_2, b_3=b_3,
                            s_ax=s_ax, d_throat=d_throat, d_outlet=d_outlet)
    if sizing_method == "barske":
        # This form neglects outlet dynamic head, so a modified version is used
        # d_2 = np.sqrt((8 * g * pump_req.H_req / omega**2 + d_1**2) / (1 + pump_choices.coeff_p))
        d_2_estimate = (2 / omega) * np.sqrt(2 * g * pump_req.H_req / 1.4)  # head coefficient ~1.4
        try:
            d_2 = opt.toms748(lambda x: pump_req.H_req - barske_head(geometry_from_d2(x), pump_req.Q_req, omega, pump_choices.coeff_p)[0], 1.1 * d_1, 4 * d_2_estimate) # Size to total head
        except:
            print("Failed to converge on d_2 for barske method, (Probably Q too high)")
    elif sizing_method == "lock":
        d_2_estimate = (2 / omega) * np.sqrt(2 * g * pump_req.H_req / 1.4)  # head coefficient ~1.4
        try:
            d_2 = opt.toms748(lambda x: pump_req.H_req - lock_head(geometry_from_d2(x), pump_req.Q_req, omega, pump_choices.n_blades)[0], 1.1 * d_1, 4 * d_2_estimate) # Size to total head
        except:
            print("Failed to converge on d_2 for lock method, (Probably Q too high)")
    elif sizing_method == "barske_simple":
        d_2 = np.sqrt((8 * g * pump_req.H_req / omega**2 + d_1**2) / (1 + pump_choices.coeff_p))
    else:
        raise ValueError(f"Invalid sizing method: {sizing_method}")

    return geometry_from_d2(d_2)