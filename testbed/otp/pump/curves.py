"""Digitised Lock chart data, and the two head correlations built on it.

Both `lock_head` and `barske_head` are pure functions of geometry + operating
point.  Sizing and forward analysis call the same function, so a sized pump and
an analysed pump can never disagree about what the correlation says.

Ported from exploration/inducer_prelim.ipynb.  The digitised numbers are Lock
Fig 10a (h_0 slip factor) and Fig 10b (C_h vs blade spacing / length).
"""

import numpy as np
import scipy.interpolate as intrp

from ..core.units import g


class _LockCurves:
    def __init__(self):
        self.eta_losses = 0.194
        self.K_factor = 0.17
        r_ratio = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        n_blades = np.array([2, 4, 8, 16])
        h0 = np.transpose([
            [0.498, 0.482, 0.435, 0.334, 0.190, 0.0],   # 2 blades
            [0.636, 0.636, 0.606, 0.517, 0.332, 0.0],   # 4 blades
            [0.768, 0.768, 0.768, 0.731, 0.518, 0.0],   # 8 blades
            [0.864, 0.864, 0.864, 0.864, 0.753, 0.0]])  # 16 blades
        h0_2D = intrp.RegularGridInterpolator((r_ratio, n_blades), h0, method="pchip")
        h0_16 = intrp.PchipInterpolator(r_ratio, [0.864, 0.864, 0.864, 0.864, 0.753, 0.0])
        self.h_0_grid = lambda r, n: (float(h0_2D([[r, n]])[0]) if n <= 16
                                      else float(h0_16(r)))
        bsl = [0.506, 0.606, 0.704, 0.804, 0.904, 1.003, 1.104, 1.202, 1.301, 1.401,
               1.501, 1.603, 1.704, 1.803, 1.903, 2.003, 2.104, 2.206, 2.307, 2.406, 2.505]
        Ch = [0.995, 0.994, 0.988, 0.980, 0.967, 0.955, 0.942, 0.926, 0.908, 0.890,
              0.872, 0.854, 0.837, 0.819, 0.803, 0.787, 0.772, 0.757, 0.743, 0.729, 0.717]
        self.bsl_range = (bsl[0], bsl[-1])
        Ch_i = intrp.interp1d(bsl, Ch, kind="linear", bounds_error=False,
                              fill_value=(Ch[0], Ch[-1]))
        self.C_h_grid = lambda s: float(Ch_i(s))

    def h_0(self, r_ratio: float, n_blades: int) -> float:
        return self.h_0_grid(r_ratio, n_blades)

    def C_h(self, blade_spacing_over_length: float) -> float:
        return self.C_h_grid(blade_spacing_over_length)


_LOCK = _LockCurves()


def lock_head(geom, Q: float, omega: float, n_blades: int):
    """Lock head developed at flow Q.  Shared by size_pump and pump analysis."""
    u_2 = omega * geom.d_2 / 2
    h_0 = _LOCK.h_0(geom.d_1 / geom.d_2, n_blades)
    C_h = _LOCK.C_h((geom.d_1 * np.pi / n_blades) / ((geom.d_2 - geom.d_1) / 2))

    dummy_1a = h_0 * u_2 ** 2 / g
    dummy_1b = (_LOCK.eta_losses * 24 / (g * geom.d_throat ** 4 * np.pi ** 2)
                * (1 - (geom.d_throat / geom.d_outlet) ** 2) ** 2)
    dummy_1c = 24 * (geom.d_outlet ** -4 - geom.d_inlet ** -4) / (np.pi ** 2 * g)
    # Negative when the outlet is wider than the inlet enough to swamp the
    # diffuser term -- a geometry the correlation does not describe.  Return NaN
    # rather than a warning, so the sizing bracket search stays quiet and the
    # caller sees an unusable point instead of a misleading number.
    ratio = dummy_1a / (dummy_1b + dummy_1c)
    if not np.isfinite(ratio) or ratio < 0:
        nan = float("nan")
        return (nan, nan, nan, h_0, C_h, dummy_1a, dummy_1b, dummy_1c, nan)
    Q_ops = np.sqrt(ratio)

    v_r_ratio = (geom.b_1 * geom.d_1) / (geom.b_2 * geom.d_2)
    H_total_wo_diff_loss = dummy_1a - (C_h * _LOCK.K_factor * v_r_ratio * (u_2 ** 2 / g)
                                       * (geom.d_1 / geom.d_2) * (1 - Q / Q_ops))
    H_static_wo_diff_loss = H_total_wo_diff_loss - dummy_1c / 3 * Q ** 2
    H_loss_diff = dummy_1b / 3 * Q ** 2
    H_total_real = H_total_wo_diff_loss - H_loss_diff
    H_static_real = H_static_wo_diff_loss - H_loss_diff
    return (H_total_real, H_static_real, H_loss_diff, h_0, C_h,
            dummy_1a, dummy_1b, dummy_1c, Q_ops)


def barske_head(geom, Q: float, omega: float, coeff_p: float) -> tuple[float, float]:
    v_inlet = Q / (np.pi * geom.d_inlet ** 2 / 4)
    u_2 = omega * geom.d_2 / 2
    u_1 = omega * geom.d_1 / 2
    v_out = Q / (np.pi * geom.d_outlet ** 2 / 4)
    H_total_real = 0.5 / g * ((1 + coeff_p) * u_2 ** 2 - u_1 ** 2 + (1 - coeff_p) * v_out ** 2)
    H_static = H_total_real - 0.5 / g * (v_out ** 2 - v_inlet ** 2)
    return H_total_real, H_static
