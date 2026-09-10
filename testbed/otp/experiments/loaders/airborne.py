"""Loader for the Airborne HDF5 files the J1 test facility writes.

Both OT-1 (R2S 2025) and OT-2 (R2S 2026) use this format, but they DO NOT use
the same channel assignments.  Read from the files themselves, 2026-09-10:

    channel   OT-1 (2025)                OT-2 (2026)
    -------   ------------------------   ----------------------
    PTX101    Engine IPA Inlet           Pump Inlet
    PTX102    Engine IPA Injector        Pump Shaft Seal
    PTX103    Engine Chamber             GG Chamber
    PTX104    Pump Outlet                Pump Outlet
    PTX105    GG Chamber                 Orifice Inlet
    TCX104    GG Chamber                 (absent)
    TCX101/2  Engine Ox / IPA Injector   GG Chamber Fore / Aft

Anything that keys off `PTX105 == GG chamber` -- which is true for OT-1 and
false for OT-2 -- silently reads the wrong sensor.  That is exactly the mistake
a per-campaign ChannelMap prevents, and the reason this loader takes a map
rather than hard-coding names.

Pressure channels are recorded in bar(g).  They are converted to absolute here,
once, so that no downstream code has to remember which convention it holds.
"""

from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from ...core.component import component, q
from ...core.units import P_A_BAR
from ..records import Channel, Rig, Run


@component
class ChannelMap:
    """Canonical name -> raw channel id, plus what to do about units."""
    campaign: str = q("", "campaign this map describes")
    mapping: dict = q("", "canonical name -> raw channel id")
    gauge_to_absolute: tuple = q("", "canonical names recorded in bar(g)",
                                 default=())
    dead: tuple = q("", "raw ids known to be dead in this campaign", default=())
    derive: dict = q("", "canonical name -> f(Run) -> Channel", default_factory=dict)
    notes: str = q("", "", default="")


def _is_dead(v: np.ndarray, flat_rel: float = 1e-3) -> tuple[bool, str]:
    """Is this channel carrying information?

    Three ways a transducer says nothing, all seen in these files:
      * all NaN                     (TC140/TC141 on OT-2 run 111)
      * bit-exactly constant
      * flat to within `flat_rel` of its own level -- PT140 on OT-2 run 111 sits
        at -11.796 bar(g) with a 0.012 bar peak-to-peak wander, which is sensor
        noise on an unplugged channel, not a measurement.
    """
    finite = np.isfinite(v)
    if not finite.any():
        return True, "all NaN"
    vv = v[finite]
    span = float(vv.max() - vv.min())
    level = max(abs(float(np.median(vv))), 1e-12)
    if span == 0.0:
        return True, "constant"
    if span / level < flat_rel:
        return True, f"flat: {span:.3g} span on a level of {np.median(vv):.4g}"
    return False, ""


def _is_unphysical_absolute(v: np.ndarray) -> tuple[bool, str]:
    """An absolute pressure below vacuum is a miscalibrated or unplugged channel."""
    med = float(np.nanmedian(v))
    if med < 0.0:
        return True, f"median absolute pressure {med:.3g} bar is below vacuum"
    return False, ""


def load_run(path: str | Path, cmap: ChannelMap, rig: Rig,
             run_id: str | None = None, notes: str = "",
             p_atm_bar: float = P_A_BAR) -> Run:
    path = Path(path)
    channels: dict[str, Channel] = {}
    dead_found: list[str] = []

    with h5py.File(path, "r") as f:
        meta = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()
                if k in ("name", "start_datetime", "summary", "operator", "location")}
        raw = f["channels"]
        for canonical, raw_id in cmap.mapping.items():
            if raw_id not in raw:
                continue
            ds = raw[raw_id]
            v = np.asarray(ds["data"], dtype=float)
            t = np.asarray(ds["time"], dtype=float)
            attrs = dict(ds.attrs)
            units = str(attrs.get("units", ""))
            desc = str(attrs.get("name", raw_id))

            dead, why = _is_dead(v)
            if raw_id in cmap.dead:
                dead, why = True, "listed as dead for this campaign"

            if canonical in cmap.gauge_to_absolute:
                v = v + p_atm_bar
                units = "bar(a)"
                if not dead:
                    dead, why = _is_unphysical_absolute(v)

            if dead:
                dead_found.append(f"{canonical} [{raw_id} {desc}] -- {why}")
                continue

            channels[canonical] = Channel(name=canonical, t=t, v=v, units=units,
                                          source=raw_id, desc=desc)

    run = Run(run_id=run_id or path.stem, campaign=cmap.campaign, rig=rig,
              channels=channels, meta=meta, notes=notes)

    for canonical, fn in cmap.derive.items():
        try:
            derived = fn(run)
        except Exception as exc:                      # a derivation that cannot run
            run.notes += f"\n  could not derive {canonical}: {exc}"
            continue
        if derived is not None:
            run.add(derived)

    for entry in dead_found:
        run.notes += f"\n  dropped: {entry}"
    return run


def orifice_flow(dp_channel: str, Cd: float, A_throat: float, rho: float,
                 out_name: str = "Q_pump") -> Callable[[Run], Channel]:
    """Build a volume-flow channel from an orifice pressure drop.

    Incompressible sharp-edged orifice, beta correction folded into Cd:
        Q = Cd * A * sqrt(2 * dp / rho)
    Only meaningful while the orifice is actually flowing full and forward.
    """
    def derive(run: Run) -> Channel:
        ch = run[dp_channel]
        dp_pa = np.clip(ch.v, 0.0, None) * 1e5
        Q = Cd * A_throat * np.sqrt(2 * dp_pa / rho)
        return Channel(name=out_name, t=ch.t, v=Q, units="m^3/s",
                       source=f"derived from {dp_channel}",
                       desc=f"orifice flow, Cd={Cd}, A={A_throat * 1e6:.2f} mm^2, "
                            f"rho={rho:.0f} kg/m3")
    return derive


def jet_thrust_flow(thrust: str, dp_channel: str, rho: float,
                    gas_flow_channels: tuple = (), gas_exit_velocity: float = 0.0,
                    Cv: float = 0.98, out_name: str = "Q_pump",
                    min_dp_bar: float = 1.0) -> Callable[[Run], Channel]:
    """Volume flow from a load cell, when the flow discharges as a free jet.

    A stand that dumps its pumped liquid through an orifice to atmosphere and
    measures the reaction has, in effect, a flow meter -- and one that needs no
    orifice diameter, because the geometry cancels:

        F = mdot * V          and      mdot = rho * A_eff * V
        =>  F = rho * A_eff * V^2      =>  A_eff = F / (rho * V^2)
        with V = Cv * sqrt(2 * dP / rho)  from the orifice pressure drop.

    So `A_eff` is an OUTPUT, not an input, and `mdot = rho * A_eff * V` follows.
    Recovering a sensible discharge coefficient from it is then a check on the
    whole chain rather than an assumption fed into it.

    `gas_flow_channels` and `gas_exit_velocity` remove any other jet pushing on
    the same load cell -- on a turbopump stand the turbine exhaust does, and it
    is not a small term.  Leave them empty only if nothing else does.

    The gas correction is the weak link: it is a subtraction, so an error in
    `gas_exit_velocity` propagates straight into the water flow.  Roughly, a
    +/-20% error there moves the liquid flow by about -/+12% on OT-2 run 111.
    """
    def derive(run: Run) -> Channel:
        F = run[thrust]
        dp_pa = np.clip(run[dp_channel].at(F.t), 0.0, None) * 1e5
        V = Cv * np.sqrt(2 * dp_pa / rho)

        F_gas = np.zeros_like(F.v)
        for name in gas_flow_channels:
            F_gas = F_gas + run[name].at(F.t) * gas_exit_velocity
        F_liquid = F.v - F_gas

        with np.errstate(divide="ignore", invalid="ignore"):
            A_eff = F_liquid / (rho * V ** 2)
            Q = A_eff * V
        Q[dp_pa < min_dp_bar * 1e5] = np.nan      # no jet, no measurement
        Q[F_liquid <= 0] = np.nan

        gas_note = (f", minus {'+'.join(gas_flow_channels)} x "
                    f"{gas_exit_velocity:.0f} m/s" if gas_flow_channels else "")
        return Channel(name=out_name, t=F.t, v=Q, units="m^3/s",
                       source=f"derived from {thrust} and {dp_channel}",
                       desc=f"free-jet flow: F=rho*A*V^2 with V from {dp_channel}, "
                            f"Cv={Cv}{gas_note}")
    return derive


def difference(a: str, b: str, out_name: str, units: str,
               desc: str = "") -> Callable[[Run], Channel]:
    """out = a - b, on a's time base."""
    def derive(run: Run) -> Channel:
        ca, cb = run[a], run[b]
        return Channel(name=out_name, t=ca.t, v=ca.v - cb.at(ca.t), units=units,
                       source=f"{a} - {b}", desc=desc or f"{a} minus {b}")
    return derive
