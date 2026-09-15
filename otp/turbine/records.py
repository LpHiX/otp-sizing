import numpy as np
from typing import List

from ..core.component import component, q
from ..pump.records import PumpPerformance
from ..engine.records import PropCEA

@component
class TurbineRequirements:
    P_req: float = q("W", "required shaft power", alt=("kW", lambda x: x / 1000))
    rpm: float = q("RPM", "shaft rotational speed")

    @staticmethod
    def from_pump_perfs(pump_perfs: List[PumpPerformance]) -> TurbineRequirements:
        first_rpm = pump_perfs[0].rpm
        for p in pump_perfs:
            if p.rpm != first_rpm:
                raise ValueError("Input pumps do not have the same RPM")
        return TurbineRequirements(sum(p.P_shaft for p in pump_perfs), pump_perfs[0].rpm)


@component
class TurbineChoices:
    d_mean: float = q("m", "mean turbine diameter", alt=("mm", lambda x: x * 1000))
    beta_deg: float = q("degree", "nozzle angle to the circumferential direction")
    doa: float = q("-", "arc degree of admission")
    n_nozzles: float = q("-", "number of stator nozzles")
    p_ratio: float = q("-", "nozzle stator pressure ratio")

@component
class TurbineGeometry:
    d_mean: float = q("m", "mean turbine diameter", alt=("mm", lambda x: x * 1000))
    beta_deg: float = q("degree", "nozzle angle to the circumferential direction")
    doa: float = q("-", "arc degree of admission")
    n_nozzles: float = q("-", "number of stator nozzles")
    blade_height: float = q("m", "blade height", alt=("mm", lambda x: x * 1000))
    A_throat_total: float = q("m^2", "total area of stator nozzle throat")
    A_3_total: float = q("m^2", "total area of stator nozzle exit")
    nozzle_throat_length: float = q("m", "length of stator nozzle throat")
    nozzle_exit_length: float = q("m", "length of stator nozzle exit")

@component
class TurbineInletGas:
    p01: float = q("bar", "stagnation pressure")
    T01: float = q("K", "stagnation temperature")
    R: float = q("J/kgK", "gas constant")
    gamma: float = q("-", "heat capacity ratio")
    model: PropCEA | None
    OF: float = q("-" "OF ratio if applicable")

# @component
# class TurbinePerformance:
