"""The experiment side of the package, and the one type that joins it to the models.

The whole modularity claim rests on this file.  A new dataset -- OT-3, the FYP
rig, a table digitised out of a paper -- never touches the physics.  It supplies
a loader that produces a `Run`, and a `Rig` describing the hardware.  Everything
downstream consumes `ObservedPoint`, which carries the same quantities a model
prediction carries, in the same units, at the same stations.

    <any file format>  --loader-->  Run  --reduce-->  ObservedPoint
                                                          |
    Rig (geometry + calibration)  --predict-->  PredictedPoint
                                                          |
                                              compare / calibrate

`ObservedPoint` fields are Optional on purpose.  OT-1 measured pump flow with a
Coriolis meter; OT-2 did not fit one.  A missing measurement must stay missing
rather than become a default, so the comparison can drop that residual instead
of scoring a model against a number nobody read.
"""

from typing import Callable, Optional

import numpy as np

from ..core.component import component, q
from ..core.states import GasState, LiquidState
from ..core.units import K_TO_C


@component
class Channel:
    name: str = q("", "canonical channel name")
    t: np.ndarray = q("s", "time base")
    v: np.ndarray = q("", "values")
    units: str = q("", "units of v")
    source: str = q("", "raw channel id this came from", default="")
    desc: str = q("", "rig description of the channel", default="")

    def at(self, t: float | np.ndarray):
        return np.interp(t, self.t, self.v)

    def window(self, t0: float, t1: float) -> "Channel":
        m = (self.t >= t0) & (self.t <= t1)
        return Channel(self.name, self.t[m], self.v[m], self.units,
                       self.source, self.desc)

    def _finite(self, t0: float, t1: float) -> np.ndarray:
        m = (self.t >= t0) & (self.t <= t1)
        seg = self.v[m]
        return seg[np.isfinite(seg)]

    def mean(self, t0: float, t1: float) -> float:
        seg = self._finite(t0, t1)
        return float(seg.mean()) if seg.size else float("nan")

    def std(self, t0: float, t1: float) -> float:
        seg = self._finite(t0, t1)
        return float(seg.std()) if seg.size else float("nan")


@component
class Rig:
    """The hardware a campaign ran, plus what it pumped and what drove it."""
    name: str = q("", "rig / turbopump name")
    pump: object = q("", "an otp.pump.analysis.Pump, or None", default=None)
    turbine: object = q("", "an otp.turbine.analysis.Turbine, or None", default=None)
    pumped_fluid: str = q("", "what the pump moved", default="")
    gg_propellants: str = q("", "gas generator propellants", default="")
    notes: str = q("", "anything the numbers do not say", default="")


@component
class Run:
    """One test firing, channels renamed to canonical names."""
    run_id: str = q("", "e.g. 20260701-111")
    campaign: str = q("", "e.g. OT-2")
    rig: Rig
    channels: dict = q("", "canonical name -> Channel")
    meta: dict = q("", "raw file attributes", default_factory=dict)
    notes: str = q("", "what this run was", default="")

    def __getitem__(self, key: str) -> Channel:
        if key not in self.channels:
            raise KeyError(f"{self.run_id} has no channel {key!r}. "
                           f"Available: {sorted(self.channels)}")
        return self.channels[key]

    def has(self, *keys: str) -> bool:
        return all(k in self.channels for k in keys)

    def add(self, channel: Channel) -> "Run":
        self.channels[channel.name] = channel
        return self


@component
class ObservedPoint:
    """A measurement reduced to one steady operating point.

    Same quantities, units and stations as `PredictedPoint`.  Anything that was
    not measured stays None.
    """
    run_id: str = q("", "source run")
    campaign: str = q("", "source campaign")
    t0: float = q("s", "window start")
    t1: float = q("s", "window end")

    rpm: Optional[float] = q("RPM", "shaft speed", default=None)
    rpm_sd: Optional[float] = q("RPM", "scatter over the window", default=None)
    rpm_source: str = q("", "how speed was obtained", default="")

    Q: Optional[float] = q("m^3/s", "pump volume flow",
                           alt=("L/s", lambda x: x * 1000), default=None)
    mdot_pump: Optional[float] = q("kg/s", "pump mass flow", default=None)
    p_pump_in: Optional[float] = q("bar", "pump inlet total pressure, absolute", default=None)
    p_pump_out: Optional[float] = q("bar", "pump outlet pressure, absolute", default=None)
    dp_pump: Optional[float] = q("bar", "pump pressure rise", default=None)
    H_total: Optional[float] = q("m", "pump total head", default=None)

    gas: Optional[GasState] = q("", "turbine inlet gas state", default=None)
    p_turbine_exit: Optional[float] = q("bar", "turbine backpressure, absolute", default=None)
    mdot_gg_ox: Optional[float] = q("kg/s", "gas generator oxidiser flow", default=None)
    mdot_gg_fuel: Optional[float] = q("kg/s", "gas generator fuel flow", default=None)
    MR_gg: Optional[float] = q("-", "gas generator mixture ratio", default=None)

    liquid: Optional[LiquidState] = q("", "pumped fluid state at the pump inlet", default=None)
    P_shaft_est: Optional[float] = q("W", "shaft power inferred from measurement",
                                     alt=("kW", lambda x: x / 1000), default=None)
    notes: str = q("", "caveats on this point", default="")

    def residual_fields(self) -> tuple[str, ...]:
        """Which comparisons this point can actually support."""
        return tuple(f for f in ("rpm", "Q", "dp_pump", "H_total", "P_shaft_est")
                     if getattr(self, f) is not None)


@component
class SteadyWindow:
    """A named slice of a run that the analyst asserts is steady enough to average."""
    run_id: str = q("", "run this window belongs to")
    label: str = q("", "what this window is")
    t0: float = q("s", "start")
    t1: float = q("s", "end")
    reason: str = q("", "why this window was chosen", default="")
