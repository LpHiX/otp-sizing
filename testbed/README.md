# otp-sizing testbed

A working end-to-end implementation of the experiment-driven turbomachinery
workflow, in a directory you can throw away. Nothing outside `testbed/` is
touched. When you are happy with the shape, `otp/` moves up to the repo root and
`testbed/` disappears.

```bash
uv run python testbed/fetch_data.py --audio      # get the data (150 MB)
uv run python testbed/run_ot2_111.py             # the analysis
uv run python testbed/test_smoke.py              # package self-checks
```

The running code targets **OT-2 run 20260701-111** and nothing else. That run
holds one steady operating point for 14.5 s and measures everything needed to
close it. An earlier version toured OT-1's 2-second spin-up as well; those
windows were not steady — flow swung by a factor of eight between adjacent
windows, heads came out negative, and the rpm came from a hand-read spectrogram
— so nothing built on them meant anything. The OT-1 loader and campaign file are
still here, because the channel-role differences between the two rigs are the
clearest illustration of why the architecture is shaped this way, but no
analysis runs on them.

---

## The recommendation, in one paragraph

Keep the four-record separation you already have — **requirements / choices /
geometry / calibration → performance** — and add one more type that sits between
the experiments and the models: **`ObservedPoint`**. Every dataset, whatever its
format, is loaded into a `Run` and reduced to `ObservedPoint`s. Every model
produces the same quantities in the same units at the same stations. Comparison
and calibration then work on those two things and never learn anything about
HDF5, channel names, or which rig produced the numbers. That single seam is what
makes "any dataset I get in the future" true rather than aspirational.

```
   any file format ──loader──▶ Run ──reduce──▶ ObservedPoint ─┐
                                                              ├─▶ compare ─▶ calibrate
   Requirements ─size─▶ Geometry ──forward──▶ Performance ────┘                  │
        ▲                                                                        │
        └──────────────────── Calibration ◀──────────────────────────────────────┘
```

The loop closes: measurements calibrate the coefficients, the calibrated model
sizes the next machine, the next machine produces measurements.

## Why the seam is `ObservedPoint` and not a DataFrame

A DataFrame would work for OT-2 and break on the second dataset. The three
things that actually differ between campaigns are not column names:

- **Which sensor plays which role.** PTX105 was the GG chamber on OT-1 and the
  orifice inlet on OT-2. PTX103 was the engine chamber on OT-1 and the GG
  chamber on OT-2. Code that keys off `PTX105` reads the wrong sensor and says
  nothing about it. A per-campaign `ChannelMap` makes that a one-line, reviewable
  declaration in `campaigns/ot1.py` and `campaigns/ot2.py`.
- **What was measured at all.** OT-1 metered the pumped fluid; OT-2 did not fit a
  meter on the water side. Every field on `ObservedPoint` is `Optional`, so a
  missing measurement stays missing and the comparison drops that residual
  instead of scoring a model against a default.
- **What has to be derived.** OT-2's water flow is not metered at all — it comes
  out of the load cell (below). That is a `derive` hook on the channel map, not a
  change to the physics.

## The water flow, and how the rig measures it without a flow meter

This is the piece that makes run 111 a complete operating point rather than a
partial one. The stand discharges the pumped water through an orifice to
atmosphere and **LC190 measures the reaction**, so the load cell is the flow
meter:

    F = ṁ V   and   ṁ = ρ A V   ⟹   A = F / (ρ V²),   V = C_v √(2 Δp / ρ)

The orifice area falls *out* of the measurement instead of being fed into it,
which is what makes this a measurement rather than an estimate. The turbine
exhaust pushes on the same load cell and is about a third of the total, so it is
subtracted using the measured gas flow.

| | run 111, 2–13 s |
|---|---|
| LC190 total | 328.8 N |
| − gas jet, 0.178 kg/s × 655 m/s | 116.6 N |
| = water jet | 212.1 N |
| orifice Δp (PTX105 − PT142) | 87.0 bar → V = 129.4 m/s |
| A_eff = F/(ρV²) | 12.69 mm² (d_eq 4.02 mm) |
| **ṁ = ρ A_eff V** | **1.639 kg/s** (1.642 L/s) |

Two things say this is right rather than merely arithmetic. That A_eff is a 5 mm
orifice at Cd = 0.646, the textbook sharp-edged value — and running it the other
way, assuming 5 mm and Cd = 0.62, gives 1.572 kg/s, within 4%. And the implied
pump useful power of 12.1 kW against a turbine net of 35 kW puts overall pump
efficiency at 35%, where a Barske at specific speed 8.4 belongs.

The one modelled input is the turbine exit velocity used to strip the gas jet. It
enters as a subtraction, so ±20% there moves the water flow by ∓12%. Everything
else in the chain is directly measured.

## Layout

```
testbed/
  fetch_data.py          the exact Drive files this needs, and nothing else
  run_ot2_111.py         the analysis: eight stages, prints its own reasoning
  test_smoke.py          self-checks for the paths the analysis does not touch
  notebooks/
    ot2_111.ipynb        the same ground, interactively, with the plots
  campaigns/
    ot2.py               OT-2 hardware, channel map, water-flow derivation
    ot1.py               OT-1 the same, kept for the channel-role contrast
  otp/
    core/                component decorator, units, Propellant/LiquidState/GasState
    engine/              ported from the notebook, unchanged
    pump/                ported; records, Lock/Barske curves, sizing, forward analysis
    turbine/             NEW: records, forward analysis, sizing
    system/              similarity scaling, hydraulic system, steady-state solver
    experiments/         Channel/Run/ObservedPoint, loaders, acoustics, reduction
    compare/             predict-vs-measured, and coefficient fitting
    plotting/            the notebook's Panel sweeps, plus measurement overlays
  data/                  downloaded, gitignored
```

## What changed from the notebook, and why

The engine and pump physics are a straight move — same equations, same numbers.
Four things did change:

1. **`Pump` no longer holds `req` and `choices`.** A forward calculation predicts
   a point from hardware; design requirements are not hardware. Holding them made
   it impossible to represent a *measured* pump, which has geometry and no
   requirements. `n_blades` moved to geometry and the Barske coefficient to
   calibration for the same reason.

2. **One signature for every liquid component:**
   `point(Q, rpm, inlet) -> PumpPerformance`, where `inlet` is a `LiquidState`.
   The Lock/Barske correlation, an affinity-scaled measured curve, and later a
   pump+inducer assembly are interchangeable. The steady-state solver takes any
   of them without knowing which.

3. **`size_pump` raises instead of falling through.** The notebook printed
   "Failed to converge on d_2" and then carried on with `d_2` unbound, so the
   real failure surfaced as an `UnboundLocalError` on an unrelated line.

4. **A `TurbineCalibration` separate from `TurbineGeometry`.** Coefficients you
   fitted and dimensions you machined are different kinds of fact and should not
   live in the same record.

## What is new

**Fixed-geometry turbine forward analysis** (`otp/turbine/analysis.py`). Geometry
+ gas state + backpressure + speed → mass flow, net shaft power, torque. This is
the thing your actual question needs and the notebook did not have. FYP's
`partload_rpm` works the other way round — it prescribes speed and a cubic pump
load and solves for the gas supply — so it cannot answer "what speed does this
turbopump settle at on a given gas supply".

**Each loss is a separate named term with its own coefficient**, so calibration
can move one without moving the others:

| term | scaling | default | where the default comes from |
|---|---|---|---|
| tip leakage | `k · δ/h · P_euler` | 1.0 | reproduces turboRocket's `phi_l = 0.199` at OT-2's δ/h = 0.179 |
| sector fill/empty | `k · chord/(π·D·ε) · ṁc₁²/2` | 0.05 | Stenning-type, O(0.05) |
| inactive-arc windage | `k · (1−ε)·ρu³·D·h` | 0.0105 | Stodola-family, O(0.01) |
| disc friction | `k · ρω³R⁵` | 0.02 | standard |

These are literature-range placeholders, and `TurbineCalibration.source` says so
in every record that carries them. Replacing them is what `otp/compare/calibrate.py`
is for.

**Audio → rpm** (`otp/experiments/acoustics.py`), both ways. The divisor is the
**rotor blade count**: OT-1 has 18 and its notebook used `f·60/18`; OT-2 has 35,
so the same audio band means a very different speed on the two rigs. Two modes:
`rpm_from_breakpoints` reproduces OT-1's hand-read ridge exactly, so the published
OT-1 numbers stay reproducible; `track_ridge` follows the band automatically and
reports per-frame prominence.

Two settings in `track_ridge` decide whether it works at all, and both were got
wrong first time round:

- **`whiten`** (default on). A rocket stand's noise is loud and strongly
  coloured. On OT-2 run 111 the broadband energy near 8–9 kHz sits about 10 dB
  *above* the 17 kHz blade tone, so a plain peak-pick locks onto the noise floor
  and pins itself near the bottom of whatever band you give it — confidently, on
  840 of 858 frames. Subtracting each frequency bin's own median level over the
  record makes the tone stand 15–25 dB clear, because the noise is steady and the
  tone is not.
- **Continuity anchoring.** `max_jump_hz` rejects frames that move further than
  the rotor could have. Chaining that forward from the first surviving frame lets
  pre-ignition noise define the track and rejects every real frame after it. It
  now anchors on the *strongest* frame and grows outwards in both directions,
  tolerating short dropouts.

`align_to_channel` finds the audio-to-DAQ offset by maximising the overlap
between the tone envelope and a DAQ channel that is on during the burn. It scores
intersection-over-union, not frame agreement — agreement is maximised by sliding
the burn entirely outside the recording, where both envelopes are off and the
score approaches 1 while the alignment is meaningless.

**Steady-state solver** (`otp/system/solve.py`). Two unknowns (rpm, Q), two
balances (shaft power, head vs system demand). It multi-starts, because this
system really does have more than one root — a partial-admission turbine carries a
scavenging loss that does not scale with blade speed, so net power crosses zero at
low speed and "no power balanced against no load" is a valid root and a useless
answer. It reports every root it found.

## What run 111 gives, and how firm each number is

Run `run_ot2_111.py` for the current output. As of 2026-09-10:

| quantity | value | how it is known |
|---|---|---|
| shaft speed | 29,806 ± 50 rpm (0.17%) over 5–13 s | measured, blade tone / 35 blades |
| water flow | 1.639 kg/s, 1.642 L/s | measured, load cell + orifice Δp |
| pump rise | 73.84 bar, 754 m head | measured, PTX104 − PTX101 |
| useful power | 12.13 kW | measured, Q·Δp |
| GG chamber | 27.33 bar(a), MR 1.62 | measured |
| gas flow | 0.1781 kg/s | measured, M730 + M850 |
| specific speed | 8.4 (rpm, m³/s, m) | derived from the above |
| impeller d₂ | 69–75 mm | **inferred** from Barske's head relation |
| turbine net | 35.1 kW, 11.3 Nm | **modelled**, uncalibrated |
| pump efficiency | ~35% | modelled, via the turbine |

The scatter on the speed is at the level of one spectrogram bin (9.2 rpm), so
most of what is left is measurement resolution rather than the machine moving.
Across the burn `Δp` follows `N²` to **0.2% RMS**, which is as clean a
confirmation as this data can give that the tone really is the shaft.

Model-side checks, unchanged by the rewrite:

- **Velocity triangle and nozzle: exact.** Against turboRocket's published OT-2
  design point, ṁ, u, c₁, c_is, w₁, w₂ and Δh_is all agree to better than 0.001%.
- **Loss model: 8% high** on turboRocket's own design point (25.8 vs 23.9 kW).
- **Turbine gas flow against run 111: −14%.** 0.153 kg/s predicted against 0.178
  measured. `Cd` is already 1.0, so no discharge coefficient closes this — see
  item 1 below.

## The five things that would move this furthest

1. **Gas properties at the measured mixture ratio.** Run 111's window ran at
   MR 1.62; `gg_out.yaml`'s `R` and `gamma` are the MR-0.5 design values, so the
   model is being handed the wrong gas. This is also why the thermocouples read
   ~1500 K against a 970 K design T0 — it ran oxidiser-richer than designed,
   though still deeply fuel-rich in absolute terms. Re-running CEA at MR 1.62 is
   a bounded job with a definite answer.
2. **Measure the orifice.** It turns the cross-check above into a genuine second
   route rather than a consistency argument.
3. **Pump geometry as a `PumpGeometry`**, so the Lock/Barske correlation can be
   run *against* this point instead of the point being used to infer `d₂`.
4. **Water temperature.** TC140 and TC141 were dead, so density is assumed. Weak
   dependence, but an assumption.
5. **A torque sensor on any future build.** Everything about efficiency here is
   inferred through an uncalibrated turbine model.

## Known gaps, stated rather than hidden

- **No inducer.** `inducer_prelim.ipynb` has no inducer code in it despite the
  name, so there was nothing to port and inventing the physics would have been
  worse than leaving the gap.
- **No torque measurement on either rig.** The only power number on the measured
  side is the pump's useful hydraulic power, which differs from turbine shaft
  power by the pump's internal losses, bearing and seal drag, and rotor inertia.
  `compare_turbine` therefore refuses to score power and compares mass flow only.
- **Spin-up slices are not steady points.** OT-1 run 002 is a 2-second spin-up;
  the shaft power balance does not hold inside those windows because the rotor is
  accelerating. They are labelled, and used only where that does not matter.
- **The gas supply is prescribed.** Once the pump feeds the engine that feeds the
  GG, this solve becomes the inner loop of a larger one. Prescribing inlet
  pressure, temperature, a choked throat *and* a demanded power over-specifies
  the problem — only three of those are independent.

## Adding a dataset

Write a loader that fills a `Run`, and a `Rig`. Nothing else changes.

```python
CHANNEL_MAP = ChannelMap(
    campaign="OT-3",
    mapping={"p_pump_in": "PTX101", "p_pump_out": "PTX104", ...},
    gauge_to_absolute=("p_pump_in", "p_pump_out", ...),
    derive={"Q_pump": orifice_flow("dp_orifice", Cd=0.62, A_throat=..., rho=998)},
)
run = load_run(path, CHANNEL_MAP, RIG)
point = reduce_point(run, window, liquid, gg_R=..., gg_gamma=..., gg_T0_K=...)
```

For a format that is not Airborne HDF5 — your FYP schema, a table from a paper —
write the equivalent of `load_run` for it. Its only job is to produce `Channel`
objects with canonical names. From `Run` onwards the path is identical.
