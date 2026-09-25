import numpy as np
from pyfluids import Input

from ..core.component import component
from ..propellants.propellant import Propellant
from .records import PumpRequirements, PumpChoices, PumpGeometry, PumpPerformance
from ..core.units import g, K_TO_C

import scipy.interpolate as intrp

class _LockCurves:
    """Digitized Lock Fig 10a (h_0 slip factor) and 10b (C_h vs blade spacing / length)."""
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
        self.h_0_grid = lambda r, n: float(h0_2D([[r, n]])[0]) if n <= 16 else float(h0_16(r))
        bsl = [0.506, 0.606, 0.704, 0.804, 0.904, 1.003, 1.104, 1.202, 1.301, 1.401,
               1.501, 1.603, 1.704, 1.803, 1.903, 2.003, 2.104, 2.206, 2.307, 2.406, 2.505]
        Ch = [0.995, 0.994, 0.988, 0.980, 0.967, 0.955, 0.942, 0.926, 0.908, 0.890,
              0.872, 0.854, 0.837, 0.819, 0.803, 0.787, 0.772, 0.757, 0.743, 0.729, 0.717]
        self.bsl_range = (bsl[0], bsl[-1])
        Ch_i = intrp.interp1d(bsl, Ch, kind="linear", bounds_error=False, fill_value=(Ch[0], Ch[-1]))
        self.C_h_grid = lambda s: float(Ch_i(s))

    def h_0(self, r_ratio: float, n_blades: int) -> float:
        return self.h_0_grid(r_ratio, n_blades)

    def C_h(self, blade_spacing_over_length: float) -> float:
        return self.C_h_grid(blade_spacing_over_length)

_LOCK = _LockCurves()

def lock_head(geom: PumpGeometry, Q: float, omega: float, n_blades: int) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Lock static head developed at flow Q. Shared by size_pump and pump analysis."""
    u_2 = omega * geom.d_2 / 2
    h_0 = _LOCK.h_0(geom.d_1 / geom.d_2, n_blades)
    C_h = _LOCK.C_h((geom.d_1 * np.pi / n_blades) / ((geom.d_2 - geom.d_1) / 2))

    dummy_1a = h_0 * u_2**2 / g
    dummy_1b = _LOCK.eta_losses * 24 / (g * geom.d_throat**4 * np.pi**2) * (1 - (geom.d_throat / geom.d_outlet)**2)**2
    dummy_1c = 24 * (geom.d_outlet**-4 - geom.d_inlet**-4) / (np.pi**2 * g)
    Q_ops = np.sqrt(dummy_1a / (dummy_1b + dummy_1c))

    v_r_ratio = (geom.b_1 * geom.d_1) / (geom.b_2 * geom.d_2)
    H_total_wo_diff_loss = dummy_1a - C_h * _LOCK.K_factor * v_r_ratio * (u_2**2 / g) * (geom.d_1 / geom.d_2) * (1 - Q / Q_ops) 
    H_static_wo_diff_loss = H_total_wo_diff_loss - dummy_1c / 3 * Q**2
    H_loss_diff = dummy_1b / 3 * Q**2
    H_total_real = H_total_wo_diff_loss - H_loss_diff
    H_static_real = H_static_wo_diff_loss - H_loss_diff
    return H_total_real, H_static_real, H_loss_diff, h_0, C_h, dummy_1a, dummy_1b, dummy_1c, Q_ops

def barske_head(geom: PumpGeometry, Q: float, omega: float, coeff_p: float) -> tuple[float, float]:
    v_inlet = Q / (np.pi * geom.d_inlet**2 / 4)
    u_2 = omega * geom.d_2 / 2
    u_1 = omega * geom.d_1 / 2
    v_out = Q / (np.pi * geom.d_outlet**2 / 4)
    H_total_real = 0.5 / g * ((1 + coeff_p) * u_2**2 - u_1**2 + (1 - coeff_p) * v_out**2)
    H_static = H_total_real - 0.5 / g * (v_out**2 - v_inlet**2)
    return H_total_real, H_static

@component
class Pump:
    req: PumpRequirements
    choices: PumpChoices
    geom: PumpGeometry

    def point_performance(self, propellant: Propellant, Q: float, rpm: float, p_upstream: float, T_upstream: float, method: str) -> PumpPerformance:

        # This analysis assumes incompressible with density at upstream conditions for now.

        upstream = propellant.fluid.with_state(Input.pressure(p_upstream),Input.temperature(T_upstream + K_TO_C))
        omega = rpm * 2 * np.pi / 60
        mdot = Q * upstream.density
        v_inlet = Q / (np.pi * self.geom.d_inlet**2 / 4)
        u_1 = omega * self.geom.d_1 / 2
        w_1 = Q / (np.pi * self.geom.d_1**2 / 4)
        v_1 = np.sqrt(w_1**2 + u_1**2)
        u_2 = omega * self.geom.d_2 / 2
        w_2 = Q / (np.pi * self.geom.d_2**2 / 4)
        v_2 = np.sqrt(w_2**2 + u_2**2)
        v_throat = Q / (np.pi * self.geom.d_throat**2 / 4)
        v_outlet = Q / (np.pi * self.geom.d_outlet**2 / 4)
        flow_coeff_inlet = w_1 / u_1
        flow_coeff_outlet = w_2 / u_2
        if upstream.kinematic_viscosity is not None:
            P_disc_friction = 1956 * upstream.density * upstream.kinematic_viscosity ** 0.2 * (rpm/1000)**2.8 * (self.geom.d_2**4.6 + 4.6 * self.geom.d_1**3.6 * self.geom.b_1)
        else:
            raise ValueError("Kinematic viscosity is None, cannot calculate disc friction power.")

        def return_performance(H_total_real, H_static_real, H_loss_diff):
            H_euler = H_total_real + H_loss_diff
            P_useful = mdot * g * H_total_real
            P_hydraulic = mdot * g * H_euler
            P_shaft = P_hydraulic + P_disc_friction
            eta_hydraulic = H_total_real / H_euler
            eta_power = P_useful / P_shaft
            torque = P_shaft / omega

            head_coeff = 2 * H_total_real * g / u_2**2
            p_inlet = p_upstream - 0.5 * upstream.density * v_inlet**2
            p_1 = p_upstream - 0.5 * upstream.density * v_1**2
            p_2_total = p_upstream + H_euler * upstream.density * g
            p_2 = p_2_total - 0.5 * upstream.density * v_2**2
            p_throat = p_2_total - 0.5 * upstream.density * v_throat**2
            p_outlet_static = p_2_total - H_loss_diff * upstream.density * g - 0.5 * upstream.density * v_outlet**2
            p_outlet = p_upstream + H_total_real * upstream.density * g
            dp = p_outlet - p_upstream

            p_sat = propellant.fluid.with_state(Input.temperature(T_upstream + K_TO_C), Input.quality(0)).pressure
            npsh_a_upstream = (p_upstream - p_sat) / (upstream.density * g)
            npsh_r_inlet = (p_upstream - p_1) / (upstream.density * g)
            npsh_r_throat = (p_upstream - p_throat) / (upstream.density * g)

            lam_c = 1.1
            lam_w_low = 0.1
            lam_w_high = 0.3
            npsh_r_ai_low = lam_c * w_1**2/(2*g) + lam_w_low * v_1**2/(2*g)   # lam_c≈1.1, lam_w≈0.1–0.3
            npsh_r_ai_high = lam_c * w_1**2/(2*g) + lam_w_high * v_1**2/(2*g)   # lam_c≈1.1, lam_w≈0.1–0.3

            specific_speed = rpm * np.sqrt(Q) / (H_total_real)**0.75
            suction_specific_speed = rpm * np.sqrt(Q) / (npsh_a_upstream**0.75)

            return PumpPerformance(propellant=propellant, Q=Q, rpm=rpm, p_upstream=p_upstream, T_upstream=T_upstream, omega=omega, mdot=mdot, dp=dp, flow_coeff_inlet=flow_coeff_inlet, flow_coeff_outlet=flow_coeff_outlet, head_coeff=head_coeff, H_total_real=H_total_real, H_static_real=H_static_real, H_loss_diff=H_loss_diff, eta_hydraulic=eta_hydraulic, P_shaft=P_shaft, P_hydraulic=P_hydraulic, P_disc_friction=P_disc_friction, P_useful=P_useful, eta_power=eta_power, torque=torque, v_inlet=v_inlet, u_1=u_1, w_1=w_1, v_1=v_1, u_2=u_2, w_2=w_2, v_2=v_2, v_throat=v_throat, v_outlet=v_outlet, p_inlet=p_inlet, p_1=p_1, p_2=p_2, p_2_total=p_2_total, p_throat=p_throat, p_outlet=p_outlet, p_outlet_static=p_outlet_static, npsh_a_upstream=npsh_a_upstream, npsh_r_inlet=npsh_r_inlet, npsh_r_throat=npsh_r_throat, npsh_r_ai_low=npsh_r_ai_low, npsh_r_ai_high=npsh_r_ai_high, specific_speed=specific_speed, suction_specific_speed=suction_specific_speed )

        if method == "lock":
            H_total_real, H_static_real, H_loss_diff, h_0, C_h, dummy_1a, dummy_1b, dummy_1c, Q_ops = lock_head(self.geom, Q, omega, self.choices.n_blades)
            H_throat = H_static_real + H_loss_diff - 8 * Q**2 * (self.geom.d_throat**-4 - self.geom.d_outlet**-4) / (np.pi**2 * g)
            p_sat = propellant.fluid.with_state(Input.temperature(T_upstream + K_TO_C), Input.quality(0)).pressure
            npsh_a_upstream = (p_upstream - p_sat) / (upstream.density * g)

            if -npsh_a_upstream > H_throat:
                H_loss_diff = H_total_real + H_loss_diff
                H_total_real = 0
                H_static_real = (-dummy_1c / 3 * Q**2)

        elif method == "barske":

            H_total_real, H_static_real = barske_head(self.geom, Q, omega, self.choices.coeff_p)
            H_euler = (2 * u_2**2 - u_1**2) / (2 * g)
            H_loss_diff = H_euler - H_total_real

            if v_throat > 1.4 * u_2:
                H_loss_diff = H_euler - H_total_real
                H_total_real = 0
                H_static_real = 0

        else:
            raise ValueError(f"Invalid method: {method}")

        return return_performance(H_total_real, H_static_real, H_loss_diff)