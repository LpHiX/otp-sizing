from typing import Callable
import numpy as np
from pathlib import Path
import h5py


from otp.core.component import component, q
from otp.experiments.records import Channel, Run
from otp.core.units import P_A, BAR

@component
class ChannelMap:
    """Canonical name -> raw channel id, plus what to do about units."""
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

def load_run(path: str | Path, cmap: ChannelMap,
             run_id: str | None = None, notes: str = "",
             p_atm_bar: float = P_A / BAR) -> Run:
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


    run = Run(run_id=run_id or path.stem,
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

def orifice_flow(dp_channel: str, Cd: float, A_throat: float, rho: float, out_name: str = "Q_pump") -> Callable[[Run], Channel]:
    def derive(run: Run) -> Channel:
        ch = run[dp_channel]
        dp_pa = np.clip(ch.v, 0.0, None) * 1e5
        Q = Cd * A_throat * np.sqrt(2 * dp_pa / rho)
        return Channel(name=out_name, t=ch.t, v=Q, units="m^3/s",
                       source=f"derived from {dp_channel}",
                       desc=f"orifice flow, Cd={Cd}, A={A_throat * 1e6:.2f} mm^2, "
                            f"rho={rho:.0f} kg/m3")
    return derive


def difference(a: str, b: str, out_name: str, units: str, desc: str = "") -> Callable[[Run], Channel]:
    def derive(run: Run) -> Channel:
        ca, cb = run[a], run[b]
        return Channel(name=out_name, t=ca.t, v=ca.v - cb.at(ca.t), units=units,
                       source=f"{a} - {b}", desc=desc or f"{a} minus {b}")
    return derive

def csv_channel(path: str | Path, col: str, units: str, shift: float = 0.0,
                t_col: str = "time", out_name: str | None = None) -> Callable[[Run], Channel]:
    def derive(run: Run) -> Channel:
        d = np.genfromtxt(path, delimiter=",", names=True)
        return Channel(name=out_name or col, t=d[t_col] + shift, v=d[col], units=units,
                       source=Path(path).name, desc=f"{col} from {Path(path).name}, t shifted {shift:+g} s")
    return derive