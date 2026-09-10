"""Shaft speed from the audio track: blade-passing tone -> rpm.

The tone the microphone hears is the rotor blade passing frequency,

    f_bp = n_blades * rpm / 60      so      rpm = 60 * f_bp / n_blades

The divisor is the ROTOR BLADE COUNT, not the nozzle count.  OT-1's R2S_2025
notebook used `N = f * 60 / 18` and the OT-1 rotor has 18 blades; OT-2's rotor
has 35, so the same audio band means a very different speed on the two rigs.
Getting this wrong scales every derived power by the same factor, which is why
`events_per_rev` is a required argument with no default.

Two extraction modes, because both are needed:

  `rpm_from_breakpoints`  reproduces OT-1 exactly -- the analyst reads the ridge
      off the spectrogram by eye and types in (t, f) pairs.  It is the honest
      fallback when the tone is buried, and it is what the published OT-1
      numbers came from, so it has to stay reproducible.

  `track_ridge`  follows the strongest bin inside a frequency band automatically
      and reports the per-frame prominence, so a weak or absent tone shows up as
      a low-confidence sample instead of a confident wrong answer.

Time alignment: audio time is video time.  OT-1 aligned with `t_zero = 161/60`,
frame 161 at 60 fps.  Pass that as `t_offset`; the returned channel is on the
DAQ time base.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
from scipy.signal import spectrogram

from ..core.component import component, q
from .records import Channel


@component
class Spectrogram:
    f: np.ndarray = q("Hz", "frequency bins")
    t: np.ndarray = q("s", "frame times, already offset onto the DAQ base")
    Sxx: np.ndarray = q("", "magnitude, shape (len(f), len(t))")
    fs: int = q("Hz", "sample rate")
    source: str = q("", "file this came from")

    @property
    def db(self) -> np.ndarray:
        return 20 * np.log10(self.Sxx + 1e-12)


def load_audio(path: str | Path) -> tuple[int, np.ndarray]:
    """Mono float samples from a WAV.  Extract a WAV from video with ffmpeg first."""
    fs, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return int(fs), data


def compute_spectrogram(path: str | Path, t_offset: float = 0.0,
                        nperseg: int = 4096, noverlap: int | None = None,
                        nfft: int | None = None) -> Spectrogram:
    """Default resolution is set for a blade tone in the kHz range.

    At fs = 44.1 kHz, nperseg = 4096 gives ~10.8 Hz bins and ~93 ms frames.  For
    a 35-blade rotor that is ~18 rpm of frequency resolution, and fine enough in
    time for a multi-second burn.  A 1-second spin-up needs a shorter window.
    """
    fs, data = load_audio(path)
    noverlap = noverlap if noverlap is not None else nperseg * 3 // 4
    f, t, Sxx = spectrogram(data, fs=fs, window="hann", nperseg=nperseg,
                            noverlap=noverlap, nfft=nfft, scaling="density",
                            mode="magnitude")
    return Spectrogram(f=f, t=t - t_offset, Sxx=Sxx, fs=fs, source=str(path))


def track_ridge(spec: Spectrogram, band: tuple[float, float],
                min_prominence_db: float = 3.0,
                max_jump_hz: float | None = None,
                whiten: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow the blade tone inside `band`.

    Returns (t, f_peak, prominence_db).  Frames below `min_prominence_db` come
    back as NaN frequency rather than as a guess, and `max_jump_hz` rejects
    frames whose peak moves further from the last accepted one than a real rotor
    could accelerate between frames.

    `whiten` (default on) subtracts each frequency bin's own median level over
    the whole record before looking for the peak.  This matters more than any
    other setting here.  A rocket test stand is loud and its noise is strongly
    coloured: on OT-2 run 111 the broadband energy around 8-9 kHz sits about
    10 dB above the 17 kHz blade tone, so a plain argmax over raw level locks
    onto the noise floor and returns a confident wrong answer -- which is
    exactly what the first version of this function did.  Against each bin's own
    background the tone stands 15-25 dB clear, because the noise is steady and
    the tone is not.

    The background is a median over time, so it is only valid when the tone is
    present for less than half the record.  For a recording that is almost
    entirely burn, pass `whiten=False` and a tight band, or trim the audio.
    """
    lo, hi = band
    sel = (spec.f >= lo) & (spec.f <= hi)
    if not sel.any():
        raise ValueError(f"no spectrogram bins inside {band} Hz")

    db_all = spec.db
    if whiten:
        db_all = db_all - np.median(db_all, axis=1, keepdims=True)

    db = db_all[sel, :]
    fband = spec.f[sel]

    idx = np.argmax(db, axis=0)
    peak_db = db[idx, np.arange(db.shape[1])]
    # Whitened: how far the peak stands above its own bin's usual level.
    # Unwhitened: how far it stands above the rest of the band this frame.
    prominence = peak_db if whiten else peak_db - np.median(db, axis=0)
    f_peak = fband[idx].astype(float)

    f_peak[prominence < min_prominence_db] = np.nan

    if max_jump_hz is not None:
        f_peak = _enforce_continuity(f_peak, prominence, max_jump_hz)

    return spec.t, f_peak, prominence


def _enforce_continuity(f_peak: np.ndarray, prominence: np.ndarray,
                        max_jump_hz: float) -> np.ndarray:
    """Reject frames that jump further than the rotor could have moved.

    Anchored on the STRONGEST frame and grown outwards in both directions, not
    chained forward from the first surviving frame.  Chaining forward means the
    first frame to clear the prominence threshold defines the track, and if that
    frame is noise -- which it will be, since noise comes before ignition --
    every real frame afterwards is rejected for disagreeing with it.
    """
    out = np.full_like(f_peak, np.nan)
    valid = np.flatnonzero(np.isfinite(f_peak))
    if valid.size == 0:
        return out

    anchor = valid[np.argmax(prominence[valid])]
    out[anchor] = f_peak[anchor]

    for direction in (1, -1):
        last, gap = f_peak[anchor], 0
        i = anchor + direction
        while 0 <= i < f_peak.size:
            val = f_peak[i]
            if np.isfinite(val) and abs(val - last) <= max_jump_hz * (1 + gap):
                out[i] = val
                last, gap = val, 0
            else:
                # Tolerate a short dropout: the tone can be masked briefly
                # without the rotor having gone anywhere.
                gap += 1
                if gap > 20:
                    break
            i += direction
    return out


def rpm_from_ridge(spec: Spectrogram, band: tuple[float, float], events_per_rev: int,
                   min_prominence_db: float = 3.0, max_jump_hz: float | None = None,
                   whiten: bool = True, name: str = "rpm") -> tuple[Channel, Channel]:
    """Automatic extraction.  Returns (rpm channel, prominence channel)."""
    t, f_peak, prom = track_ridge(spec, band, min_prominence_db, max_jump_hz, whiten)
    rpm = 60.0 * f_peak / events_per_rev
    src = (f"blade-passing ridge in {band[0]:.0f}-{band[1]:.0f} Hz / "
           f"{events_per_rev} per rev" + (", background-whitened" if whiten else ""))
    return (Channel(name=name, t=t, v=rpm, units="RPM", source=spec.source, desc=src),
            Channel(name=name + "_prominence", t=t, v=prom, units="dB",
                    source=spec.source,
                    desc="peak above its own bin background" if whiten
                         else "peak above band median"))


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    """Start and stop indices of the longest contiguous True stretch."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, bool).view(np.int8), 0]))
    if edges.size == 0:
        raise ValueError("mask is entirely False")
    return max(zip(edges[::2], edges[1::2]), key=lambda r: r[1] - r[0])


def align_to_channel(rpm_channel: Channel, reference: Channel, level: float,
                     search: tuple[float, float] = (-30.0, 30.0),
                     step: float = 0.01) -> tuple[float, float]:
    """Find the audio-to-DAQ time offset by lining up two on/off envelopes.

    `reference` is a DAQ channel that is unambiguously on during the burn and off
    outside it -- gas generator chamber pressure is the obvious one.  `level` is
    the threshold that makes it a square wave.  Returns (offset, agreement),
    where `offset` is the number to put in the campaign's AUDIO_T_OFFSET and
    `agreement` is the fraction of audio frames whose tone-present state matches
    the DAQ burn state at that offset.

    Subtract the offset from audio time to get DAQ time -- which is exactly what
    `compute_spectrogram(..., t_offset=OFFSET)` does.

    Scored by intersection over union of the two "on" intervals, NOT by the
    fraction of frames that agree.  Agreement is maximised by sliding the burn
    entirely outside the recording, where both envelopes are off everywhere and
    the score approaches 1 while the alignment is meaningless.
    """
    tone_on = np.isfinite(rpm_channel.v)
    if not tone_on.any():
        raise ValueError("the rpm channel has no valid frames to align")
    daq_on = (reference.v > level).astype(float)
    if not (daq_on > 0.5).any():
        raise ValueError(f"{reference.name} never exceeds {level}")

    best = (0.0, -1.0)
    for off in np.arange(search[0], search[1], step):
        at_audio = np.interp(rpm_channel.t - off, reference.t, daq_on,
                             left=0.0, right=0.0) > 0.5
        union = np.count_nonzero(at_audio | tone_on)
        if union == 0:
            continue
        score = np.count_nonzero(at_audio & tone_on) / union
        if score > best[1]:
            best = (float(off), float(score))
    if best[1] <= 0:
        raise ValueError("no offset in the search range makes the two overlap")
    return best


def rpm_from_breakpoints(t_points, f_points, events_per_rev: int,
                         t_grid: np.ndarray | None = None, name: str = "rpm",
                         source: str = "manual spectrogram reading") -> Channel:
    """Manual extraction, the OT-1 method.

    `t_points`/`f_points` are the (time, frequency) pairs read off the
    spectrogram by eye.  Linear interpolation between them, exactly as OT-1 did,
    and no extrapolation outside the read range.
    """
    t_points = np.asarray(t_points, float)
    f_points = np.asarray(f_points, float)
    if t_points.shape != f_points.shape:
        raise ValueError("t_points and f_points must be the same length")
    fn = interp1d(t_points, f_points, bounds_error=False, fill_value=np.nan)
    t = t_grid if t_grid is not None else np.linspace(t_points[0], t_points[-1], 400)
    return Channel(name=name, t=t, v=60.0 * fn(t) / events_per_rev, units="RPM",
                   source=source,
                   desc=f"{len(t_points)} hand-read points / {events_per_rev} per rev")


def band_for_rpm(rpm_lo: float, rpm_hi: float, events_per_rev: int) -> tuple[float, float]:
    """Frequency band to search, given the speed range the rotor could plausibly be in."""
    return (rpm_lo * events_per_rev / 60.0, rpm_hi * events_per_rev / 60.0)
