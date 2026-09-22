from typing import Tuple

import numpy as np

from .records import TurbineRequirements, TurbineChoices, TurbineGeometry, TurbineInletGas, TurbinePerformance
from .analysis import nozzle_velocity_coeff, rotor_velocity_coeff, windage_power, Turbine
import scipy.optimize as opt

def size_turbine(turbine_req: TurbineRequirements, turbine_choices: TurbineChoices, turbine_inlet_gas: TurbineInletGas, mdot: float, max_iter: int=20, tol:float = 1e-2) -> Tuple[TurbineGeometry, TurbinePerformance]:
    P_target = turbine_req.P_req
    rpm = turbine_req.rpm
    beta_deg = turbine_choices.beta_deg
    d_mean = turbine_choices.d_mean
    doa = turbine_choices.doa
    n_nozzles = turbine_choices.n_nozzles

    beta = np.deg2rad(beta_deg)


    # Angular velocity & blade speed
    w = rpm * 2 * np.pi / 60
    u = w * d_mean / 2

    stator = stator_thermal_gg if turbine_inlet_gas.model is not None else stator_thermal_ideal

    P_req = P_target

    for _ in range(max_iter):
        # Useful enthalpy drop
        deltah_useful = P_req / mdot
        
        # Velocity triangles
        c3u_ideal = deltah_useful / (2 * u) + u
        c3_ideal = c3u_ideal / np.cos(beta)

        a_throat, a_3, rho_throat, rho_3, eps, p_ratio, M3 = stator(turbine_inlet_gas, c3_ideal)

        phi_n = nozzle_velocity_coeff(M3)
        c3_real = phi_n * c3_ideal
        c3m_real = c3_real * np.sin(beta)
        c3u_real = c3_real * np.cos(beta)
        blade_height = mdot / (doa * rho_3 * c3m_real * d_mean * np.pi)

        dB = (180 - beta_deg * 2)
        w3u_real = c3u_real - u
        w3_real = np.sqrt(w3u_real**2 + c3m_real**2)
        Mr =  w3_real / a_3
        # print(f"dB={dB:.1f}, Mr={Mr:.2f}, c3u_real={c3u_real:.1f}, u={u:.1f}, w3_real={w3_real:.1f}, a_3={a_3:.1f}, c3_real={c3_real:.1f}")
        phi_r = rotor_velocity_coeff(dB, Mr)

        w4_real = w3_real * phi_r
        w4u_real = w4_real * np.cos(beta)
        c4u_real = u - w4u_real

        p_v = windage_power(doa, rho_3, rpm, d_mean, blade_height)

        P_euler = mdot * u * (c3u_real - c4u_real)
        P_shaft = P_euler - p_v
        if P_shaft <= 0:
            raise ValueError(
                f"No net power: windage {p_v:.0f} W exceeds the Euler work of {P_euler:.0f} W. "
                f"Lower the admission fraction or the speed.")

        if abs(P_shaft - P_target) <= tol * P_target:
            A_throat_total = mdot / (rho_throat * a_throat)
            nozzle_throat_length = A_throat_total / n_nozzles / blade_height

            A_3_total = A_throat_total * eps
            nozzle_exit_length = eps * A_throat_total / n_nozzles / blade_height

            geometry = TurbineGeometry(
                d_mean=d_mean,
                beta_deg=beta_deg,
                doa=doa,
                n_nozzles=n_nozzles,
                blade_height=blade_height,
                A_throat_total=A_throat_total,
                A_3_total=A_3_total,
                nozzle_throat_length=nozzle_throat_length,
                nozzle_exit_length=nozzle_exit_length
            )

            performance = Turbine(turbine_req, turbine_choices, geometry).point_performance(turbine_inlet_gas, rpm, turbine_inlet_gas.p01/p_ratio)
            return geometry, performance
        P_req *= P_target / P_shaft
    raise ValueError(
        f"Failed to converge on turbine geometry after {max_iter} iterations "
        f"(last P_shaft={P_shaft:.1f} W vs target {P_target:.1f} W)")

def stator_thermal_gg(turbine_inlet_gas: TurbineInletGas, c3_ideal: float, eps_bounds: tuple[float, float] = (1.0, 200.0)) -> tuple[float, float, float, float, float, float, float]:
    if turbine_inlet_gas.model is None:
        raise ValueError("No CEA model given to calculate turbine gas parameters")
    cea = turbine_inlet_gas.model.cea
    p01 = turbine_inlet_gas.p01
    OF = turbine_inlet_gas.OF


    def h_exit(eps: float) -> float:
        return cea.get_Enthalpies(Pc=p01, MR=OF, eps=eps, frozen=0)[2] * 1000.0  # J/g -> J/kg

    h01 = cea.get_Enthalpies(Pc=p01, MR=OF, eps=eps_bounds[0], frozen=0)[0] * 1000.0


    def c3_at(eps: float) -> float:
        """Exit velocity this area ratio delivers, from h01 = h3 + c3^2/2."""
        return np.sqrt(2.0 * (h01 - cea.get_Enthalpies(Pc=p01, MR=OF, eps=eps, frozen=0)[2] * 1000.0))

    c_lo, c_hi = c3_at(eps_bounds[0]), c3_at(eps_bounds[1])
    if c3_ideal < c_lo:
        raise ValueError(
            f"c3_ideal={c3_ideal:.1f} m/s is below the sonic exit velocity {c_lo:.1f} m/s -- "
            f"the stage is too lightly loaded for a converging-diverging nozzle at this speed")
    if c3_ideal > c_hi:
        raise ValueError(
            f"c3_ideal={c3_ideal:.1f} m/s exceeds {c_hi:.1f} m/s at eps={eps_bounds[1]:.0f}; "
            f"not enough total enthalpy in this gas")
    eps = opt.brentq(lambda e: c3_at(e) - c3_ideal, *eps_bounds)

    _, a_throat, a_3 = cea.get_SonicVelocities(Pc=p01, MR=OF, eps=eps)
    _, rho_throat, rho_3 = cea.get_Densities(Pc=p01, MR=OF, eps=eps)
    M3 = cea.get_MachNumber(Pc=p01, MR=OF, eps=eps)
    p_ratio = cea.get_PcOvPe(Pc=p01, MR=OF, eps=eps)
    return a_throat, a_3, rho_throat, rho_3, eps, p_ratio, M3

def stator_thermal_ideal(turbine_inlet_gas: TurbineInletGas, c3_ideal: float) -> tuple[float, float, float, float, float, float, float]:
    p01 = turbine_inlet_gas.p01
    T01 = turbine_inlet_gas.T01
    R = turbine_inlet_gas.R
    gamma = turbine_inlet_gas.gamma
    cp = gamma * R / (gamma - 1)

    T3 = T01 - 0.5 * c3_ideal**2 / cp
    if T3 <= 0:
        raise ValueError(f"c3_ideal={c3_ideal:.1f} m/s exceeds the total enthalpy of this gas")
    a_3 = np.sqrt(gamma * R * T3)
    M3 = c3_ideal / a_3
    p_ratio = (1 + 0.5 * (gamma - 1) * M3**2) ** (gamma / (gamma - 1))
    p3 = p01 / p_ratio
    rho_3 = p3 / (R * T3)

    T_throat = T01 / (1 + 0.5 * (gamma - 1))
    p_throat = p01 * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    a_throat = np.sqrt(gamma * R * T_throat)
    rho_throat = p_throat / (R * T_throat)

    eps = (0.5 * (gamma + 1))**(-(gamma + 1) / (2 * (gamma - 1))) * (1 + 0.5 * (gamma - 1) * M3**2)**((gamma + 1) / (2 * (gamma - 1))) / M3
    return a_throat, a_3, rho_throat, rho_3, eps, p_ratio, M3