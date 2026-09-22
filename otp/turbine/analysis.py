import numpy as np
import scipy.optimize as opt

from ..core.component import component
from .records import (TurbineRequirements, TurbineChoices, TurbineGeometry,
                      TurbineInletGas, TurbinePerformance)



def gas_state_at_eps(gas: TurbineInletGas, eps: float) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    if gas.model is not None:
        cea, p01, OF = gas.model.cea, gas.p01, gas.OF
        _, a_throat, a_3 = cea.get_SonicVelocities(Pc=p01, MR=OF, eps=eps)
        _, rho_throat, rho_3 = cea.get_Densities(Pc=p01, MR=OF, eps=eps)
        M3 = cea.get_MachNumber(Pc=p01, MR=OF, eps=eps)
        p_ratio = cea.get_PcOvPe(Pc=p01, MR=OF, eps=eps)
        h01, _, h3 = [h * 1000.0 for h in cea.get_Enthalpies(Pc=p01, MR=OF, eps=eps, frozen=0)]
        c3_ideal = np.sqrt(2.0 * (h01 - h3))
        _, T_throat, T3 = cea.get_Temperatures(Pc=p01, MR=OF, eps=eps)
        p_throat = p01 / cea.get_Throat_PcOvPe(Pc=p01, MR=OF)
        p3 = p01 / p_ratio
        return a_throat, a_3, rho_throat, rho_3, p_ratio, M3, c3_ideal, p_throat, T_throat, p3, T3

    gamma, R, T01, p01 = gas.gamma, gas.R, gas.T01, gas.p01

    def area_ratio(M: float) -> float:
        return (1 / M) * ((2 / (gamma + 1)) * (1 + 0.5 * (gamma - 1) * M**2)) ** ((gamma + 1) / (2 * (gamma - 1)))

    M3 = opt.brentq(lambda M: area_ratio(M) - eps, 1.0, 50.0)

    T3 = T01 / (1 + 0.5 * (gamma - 1) * M3**2)
    a_3 = np.sqrt(gamma * R * T3)
    c3_ideal = M3 * a_3
    p_ratio = (1 + 0.5 * (gamma - 1) * M3**2) ** (gamma / (gamma - 1))
    rho_3 = (p01 / p_ratio) / (R * T3)
    p3 = p01 / p_ratio

    T_throat = T01 / (1 + 0.5 * (gamma - 1))
    p_throat = p01 * (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    a_throat = np.sqrt(gamma * R * T_throat)
    rho_throat = p_throat / (R * T_throat)
    return a_throat, a_3, rho_throat, rho_3, p_ratio, M3, c3_ideal, p_throat, T_throat, p3, T3


def deltah_isentropic_ta(gas: TurbineInletGas, p_amb: float) -> float:
    if gas.model is not None:
        cea, p01, OF = gas.model.cea, gas.p01, gas.OF
        eps_amb = cea.get_eps_at_PcOvPe(Pc=p01, MR=OF, PcOvPe=p01 / p_amb, frozen=0)
        h01, _, h_amb = [h * 1000.0 for h in cea.get_Enthalpies(Pc=p01, MR=OF, eps=eps_amb, frozen=0)]
        return h01 - h_amb

    cp = gas.gamma * gas.R / (gas.gamma - 1)
    T_amb_is = gas.T01 * (p_amb / gas.p01) ** ((gas.gamma - 1) / gas.gamma)
    return cp * (gas.T01 - T_amb_is)


def nozzle_velocity_coeff(M3: float) -> float:
    return float(np.sqrt(1 - (0.0029 * M3**3 - 0.0502 * M3**2 + 0.2241 * M3 - 0.0877)))


def rotor_velocity_coeff(dB: float, Mr: float) -> float:
    return float(
        0.957
        - 0.000362 * dB        - 0.0258 * Mr
        + 0.00000639 * dB**2   + 0.0674 * Mr**2
        - 0.0000000753 * dB**3 - 0.043 * Mr**3
        - 0.000238 * dB * Mr
        + 0.00000145 * dB**2 * Mr
        + 0.0000425 * dB * Mr**2
    )


def windage_power(doa: float, rho_3: float, rpm: float, d_mean: float, blade_height: float) -> float:
    return (1.85 / 2) * ((1 - doa) * rho_3 * (rpm / 60)**3 * d_mean**4 * 4.5 * blade_height)


@component
class Turbine:
    req: TurbineRequirements
    choices: TurbineChoices
    geom: TurbineGeometry

    def point_performance(self, gas: TurbineInletGas, rpm: float, p_amb: float) -> TurbinePerformance:
        beta = np.deg2rad(self.geom.beta_deg)
        omega = rpm * 2 * np.pi / 60
        u = omega * self.geom.d_mean / 2

        eps = self.geom.A_3_total / self.geom.A_throat_total
        a_throat, a_3, rho_throat, rho_3, p_ratio, M3, c3_ideal, p_throat, T_throat, p3, T3 = gas_state_at_eps(gas, eps)

        mdot = rho_throat * a_throat * self.geom.A_throat_total

        phi_n = nozzle_velocity_coeff(M3)
        c3_real = phi_n * c3_ideal
        c3m_real = c3_real * np.sin(beta)
        c3u_real = c3_real * np.cos(beta)

        dB = 180 - self.geom.beta_deg * 2
        w3u_real = c3u_real - u
        w3_real = np.sqrt(w3u_real**2 + c3m_real**2)
        Mr = w3_real / a_3
        phi_r = rotor_velocity_coeff(dB, Mr)

        w4_real = w3_real * phi_r
        w4u_real = w4_real * np.cos(beta)
        c4u_real = u - w4u_real

        P_euler = mdot * u * (c3u_real - c4u_real)
        P_windage = windage_power(self.geom.doa, rho_3, rpm, self.geom.d_mean, self.geom.blade_height)
        P_shaft = P_euler - P_windage
        torque = P_shaft / omega

        deltah_useful = P_shaft / mdot
        deltah_is_ta = deltah_isentropic_ta(gas, p_amb)
        c_0 = np.sqrt(2 * deltah_is_ta)

        # Does the blade height pass the flow the choked throat is delivering?
        mdot_continuity = self.geom.doa * rho_3 * c3m_real * self.geom.d_mean * np.pi * self.geom.blade_height

        return TurbinePerformance(
            gas=gas, rpm=rpm, omega=omega, u=u, mdot=mdot, p_amb=p_amb,
            eps=eps,p_ratio=p_ratio,p01=gas.p01,T01=gas.T01,p_throat=p_throat,T_throat=T_throat,a_throat=a_throat,rho_throat=rho_throat,p3=p3,T3=T3,M3=M3,Mr=Mr,a_3=a_3,rho_3=rho_3,
            phi_n=phi_n, phi_r=phi_r,
            c3_ideal=c3_ideal, c3_real=c3_real, c3u_real=c3u_real, c3m_real=c3m_real,
            w3_real=w3_real, w3u_real=w3u_real, w4_real=w4_real, w4u_real=w4u_real,
            c4u_real=c4u_real,
            deltah_useful=deltah_useful, deltah_is_ta=deltah_is_ta,
            c_0=c_0, u_over_c0=u / c_0, eta_is_ta=deltah_useful / deltah_is_ta,
            P_euler=P_euler, P_windage=P_windage, P_shaft=P_shaft, torque=torque,
            mdot_continuity=mdot_continuity, continuity_ratio=mdot_continuity / mdot,
        )
