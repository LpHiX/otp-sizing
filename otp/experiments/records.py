from typing import Callable, Optional

import numpy as np

from ..core.component import component, q
from ..engine.records import InjectorGeometry, ChamberGeometry
from ..pump.records import PumpGeometry
from ..turbine.records import TurbineGeometry

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

    def max(self, t0: float, t1: float) -> float:
        seg = self._finite(t0, t1)
        return float(seg.max()) if seg.size else float("nan")

    def min(self, t0: float, t1: float) -> float:
        seg = self._finite(t0, t1)
        return float(seg.min()) if seg.size else float("nan")

@component
class Run:
    run_id: str = q("", "e.g. 20260701-111")
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