"""Fixed-geometry pump forward analysis.

Signature change from the notebook, and the reason for it:

    notebook:  pump.point_performance(propellant, Q, rpm, p_upstream, T_upstream, method)
    here:      pump.point_performance(Q, rpm, inlet)          inlet: LiquidState

Every liquid component in the package -- pump, inducer, a similarity-scaled
measured curve, a pump+inducer assembly -- answers exactly this call and returns
a `PumpPerformance`.  That uniformity is what lets the steady-state solver take
any of them without knowing which it has.  The head correlation is chosen once,
when the Pump is built, and can still be overridden per call for sweeps.
"""

import numpy as np

from ..core.component import component, q
from ..core.states import LiquidState
from ..core.units import BAR, g, rpm_to_rad
from .curves import barske_head, lock_head
from .records import PumpCalibration, PumpGeometry, PumpPerformance


@component
class Pump:
    geom: PumpGeometry
    calib: PumpCalibration = q("", "loss/head coefficients",
                               default_factory=PumpCalibration)
    method: str = q("", "head correlation: lock | barske", default="lock")

    def point_performance(self, Q: float, rpm: float, inlet: LiquidState,
                          method: str | None = None) -> PumpPerformance:
        # Incompressible, density taken at inlet conditions.
        method = method or self.method
        geom, rho = self.geom, inlet.rho
        omega = rpm_to_rad(rpm)
        p_upstream = inlet.p0_bar

        mdot = Q * rho
        v_inlet = Q / (np.pi * geom.d_inlet ** 2 / 4)
        u_1 = omega * geom.d_1 / 2
        w_1 = Q / (np.pi * geom.d_1 ** 2 / 4)
        v_1 = np.hypot(w_1, u_1)
        u_2 = omega * geom.d_2 / 2
        w_2 = Q / (np.pi * geom.d_2 ** 2 / 4)
        v_2 = np.hypot(w_2, u_2)
        v_throat = Q / (np.pi * geom.d_throat ** 2 / 4)
        v_outlet = Q / (np.pi * geom.d_outlet ** 2 / 4)

        if inlet.nu is None:
            raise ValueError("LiquidState.nu is None; disc friction cannot be evaluated.")
        P_disc_friction = (1956 * rho * inlet.nu ** 0.2 * (rpm / 1000) ** 2.8
                           * (geom.d_2 ** 4.6 + 4.6 * geom.d_1 ** 3.6 * geom.b_1)
                           * self.calib.disc_friction_multiplier)

        npsh_a_upstream = (p_upstream - inlet.p_vap_bar) * BAR / (rho * g)
        cavitating = False

        if method == "lock":
            (H_total_real, H_static_real, H_loss_diff, _h0, _Ch,
             _a, _b, dummy_1c, _Qops) = lock_head(geom, Q, omega, geom.n_blades)
            H_throat = (H_static_real + H_loss_diff
                        - 8 * Q ** 2 * (geom.d_throat ** -4 - geom.d_outlet ** -4)
                        / (np.pi ** 2 * g))
            if -npsh_a_upstream > H_throat:
                cavitating = True
                H_loss_diff = H_total_real + H_loss_diff
                H_total_real = 0.0
                H_static_real = -dummy_1c / 3 * Q ** 2
        elif method == "barske":
            H_total_real, H_static_real = barske_head(geom, Q, omega, self.calib.coeff_p)
            H_euler = (2 * u_2 ** 2 - u_1 ** 2) / (2 * g)
            H_loss_diff = H_euler - H_total_real
            if v_throat > 1.4 * u_2:
                cavitating = True
                H_total_real = 0.0
                H_static_real = 0.0
        else:
            raise ValueError(f"Invalid method: {method}")

        H_total_real *= self.calib.head_multiplier
        H_static_real *= self.calib.head_multiplier

        H_euler = H_total_real + H_loss_diff
        P_useful = mdot * g * H_total_real
        P_hydraulic = mdot * g * H_euler
        P_shaft = (P_hydraulic + P_disc_friction) * self.calib.power_multiplier
        eta_hydraulic = H_total_real / H_euler if H_euler else np.nan
        eta_power = P_useful / P_shaft if P_shaft else np.nan
        torque = P_shaft / omega if omega else np.nan
        head_coeff = 2 * H_total_real * g / u_2 ** 2 if u_2 else np.nan

        p_inlet = p_upstream - 0.5 * rho * v_inlet ** 2 / BAR
        p_1 = p_upstream - 0.5 * rho * v_1 ** 2 / BAR
        p_2_total = p_upstream + H_euler * rho * g / BAR
        p_2 = p_2_total - 0.5 * rho * v_2 ** 2 / BAR
        p_throat = p_2_total - 0.5 * rho * v_throat ** 2 / BAR
        p_outlet_static = (p_2_total - H_loss_diff * rho * g / BAR
                           - 0.5 * rho * v_outlet ** 2 / BAR)
        p_outlet = p_upstream + H_total_real * rho * g / BAR

        lam_c, lam_w_low, lam_w_high = 1.1, 0.1, 0.3   # lam_c ~1.1, lam_w ~0.1-0.3
        return PumpPerformance(
            liquid=inlet, Q=Q, rpm=rpm, p_upstream=p_upstream, T_upstream=inlet.T_K,
            omega=omega, mdot=mdot, dp=p_outlet - p_upstream,
            flow_coeff_inlet=w_1 / u_1 if u_1 else np.nan,
            flow_coeff_outlet=w_2 / u_2 if u_2 else np.nan,
            head_coeff=head_coeff,
            H_total_real=H_total_real, H_static_real=H_static_real, H_loss_diff=H_loss_diff,
            eta_hydraulic=eta_hydraulic, P_shaft=P_shaft, P_hydraulic=P_hydraulic,
            P_disc_friction=P_disc_friction, P_useful=P_useful, eta_power=eta_power,
            torque=torque, v_inlet=v_inlet, u_1=u_1, w_1=w_1, v_1=v_1, u_2=u_2, w_2=w_2,
            v_2=v_2, v_throat=v_throat, v_outlet=v_outlet, p_inlet=p_inlet, p_1=p_1,
            p_2=p_2, p_2_total=p_2_total, p_throat=p_throat, p_outlet=p_outlet,
            p_outlet_static=p_outlet_static, npsh_a_upstream=npsh_a_upstream,
            npsh_r_inlet=(p_upstream - p_1) * BAR / (rho * g),
            npsh_r_throat=(p_upstream - p_throat) * BAR / (rho * g),
            npsh_r_ai_low=lam_c * w_1 ** 2 / (2 * g) + lam_w_low * v_1 ** 2 / (2 * g),
            npsh_r_ai_high=lam_c * w_1 ** 2 / (2 * g) + lam_w_high * v_1 ** 2 / (2 * g),
            suction_specific_speed=(rpm * np.sqrt(Q) / npsh_a_upstream ** 0.75
                                    if npsh_a_upstream > 0 else np.nan),
            method=method, cavitating=cavitating)
