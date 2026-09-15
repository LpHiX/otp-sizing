import numpy as np

from .records import TurbineRequirements, TurbineChoices, TurbineGeometry, TurbineInletGas
from ..engine.records import EnginePerformance

def size_turbine(turbine_req: TurbineRequirements, turbine_choices: TurbineChoices, turbine_inlet_gas: TurbineInletGas, mdot: float, max_iter: int) -> TurbineGeometry | None:
    P_target = turbine_req.P_req
    beta_deg = turbine_choices.beta_deg
    rpm = turbine_req.rpm
    d_mean = turbine_choices.d_mean
    p_ratio = turbine_choices.p_ratio
    doa = turbine_choices.doa
    n_nozzles = turbine_choices.n_nozzles

    beta = np.deg2rad(beta_deg)

    P_req = 2 * P_target # Arbitrary 2x iteration starting point

    # Angular velocity & blade speed
    w = rpm * 2 * np.pi / 60
    u = w * d_mean / 2

    for _ in range(max_iter):
        # Useful enthalpy drop
        deltah_useful = P_req / mdot
        
        # Velocity triangles
        c3u_ideal = deltah_useful / (2 * u) + u
        c3_ideal = c3u_ideal / np.cos(beta)
        c3m_ideal = c3_ideal * np.sin(beta)

        a_throat, rho_throat, eps, rho_3, M3 = 0,0,0,0,0 # To stop stupid unbound comment

        if turbine_inlet_gas.model is not None:
            a_throat, rho_throat, eps, rho_3, M3 = stator_thermal_gg(turbine_inlet_gas, p_ratio)
        else:   
            a_throat, rho_throat, eps, rho_3, M3 = stator_thermal_ideal(turbine_inlet_gas, p_ratio)

        

        phi_n = np.sqrt(
            1 - (0.0029 * M3**3 - 0.0502 * M3**2 + 0.2241 * M3 - 0.0877)
        )
        c3_real = phi_n * c3_ideal
        c3m_real = c3_real * np.sin(beta)
        c3u_real = c3_real * np.cos(beta)
        blade_height = mdot / (doa * rho_3 * c3m_real * d_mean * np.pi)

        dB = (180 - beta_deg * 2)
        w3u_real = c3u_real - u
        w3_real = np.sqrt(w3u_real**2 + c3m_real**2)
        Mr =  w3_real / a_throat
        phi_r = (
            0.957
            - 0.000362 * dB        - 0.0258 * Mr
            + 0.00000639 * dB**2   + 0.0674 * Mr**2
            - 0.0000000753 * dB**3 - 0.043 * Mr**3
            - 0.000238 * dB * Mr
            + 0.00000145 * dB**2 * Mr
            + 0.0000425 * dB * Mr**2
        )

        w4_real = w3_real * phi_r
        w4u_real = w4_real * np.cos(beta)
        c4u_real = u - w4u_real 

        p_v = (1.85 / 2) * (
            (1 - doa) * rho_3 * (rpm / 60)**3
            * d_mean**4 * 4.5 * blade_height
        )

        P_shaft = mdot * u * (c3u_real - c4u_real) - p_v
        P_req *= P_target / P_shaft


        # Total throat area
        A_throat_total = mdot / (rho_throat * a_throat)
        nozzle_throat_length = A_throat_total / n_nozzles / blade_height

        A_3_total = A_throat_total * eps
        nozzle_exit_length = eps * A_throat_total / n_nozzles / blade_height

        return TurbineGeometry(
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
    raise ValueError("Failed to converge on turbine geometry after max iterations")


def stator_thermal_gg(turbine_inlet_gas: TurbineInletGas, p_ratio: float) -> tuple[float, float, float, float, float]:
    if turbine_inlet_gas.model is None:
        raise ValueError("No CEA model given to calculate turbine gas parameters")
    cea = turbine_inlet_gas.model.cea
    p01 = turbine_inlet_gas.p01
    OF = turbine_inlet_gas.OF


    # Assume the long gas gen passage to allow full equilibrium
    eps =cea.get_eps_at_PcOvPe(Pc=p01, MR=OF, PcOvPe=p_ratio, frozen=0)

    T01, _, T3 = cea.get_Temperatures(Pc=p01, MR=OF, eps=eps)
    _ , a_throat, a3 = cea.get_SonicVelocities(Pc=p01, MR=OF, eps=eps)
    M3 = cea.get_MachNumber(Pc=p01, MR=OF, eps=eps)
    
    # Pressure & density at blade inlet
    # p3 = p01 / (1 + 0.5 * (gam - 1) * M3**2)**(gam/(gam-1))
    _, rho_throat, rho_3 = cea.get_Densities(Pc=p01, MR=OF, eps=eps)
    return a_throat, rho_throat, eps, rho_3, M3

def stator_thermal_ideal(turbine_inlet_gas: TurbineInletGas, p_ratio: float) -> tuple[float, float, float, float, float]:
    p01 = turbine_inlet_gas.p01
    T01 = turbine_inlet_gas.T01
    R = turbine_inlet_gas.R
    gamma = turbine_inlet_gas.gamma

    M3 = np.sqrt((2/(gamma-1)) * (p_ratio**((gamma-1)/gamma) - 1))
    T3 = T01 / (1 + 0.5 * (gamma - 1) * M3**2)
    a_throat = np.sqrt(gamma * R * T3)
    rho_throat = p01*1e5 / (R * T3)
    rho_3 = p01*1e5 / (R * T3)
    eps = (0.5 * (gamma + 1))**(-(gamma + 1) / (2 * (gamma - 1))) * (1 + 0.5 * (gamma - 1) * M3**2)**((gamma + 1) / (2 * (gamma - 1))) / M3
    return a_throat, rho_throat, eps, rho_3, M3