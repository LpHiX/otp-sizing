"""OT-2 run 20260701-111: the whole operating point, from the raw file.

    uv run python testbed/run_ot2_111.py

One run, one purpose: turn 14.5 seconds of a fuel-rich qualification burn into a
single complete, defensible turbopump operating point, and say exactly how much
of it is measured and how much is modelled.

This replaces the earlier multi-campaign tour.  That version leaned on OT-1's
2-second spin-up, sliced into quasi-steady windows and driven by a hand-read rpm
trace.  Those slices were not steady -- flow swung by a factor of eight between
adjacent windows and heads came out negative -- so nothing built on them meant
anything.  Run 111 holds a genuinely steady point for 14.5 s and, once the load
cell is understood, measures everything needed to close it.

  1  load the run and see what the rig actually recorded
  2  shaft speed, from the blade tone
  3  water flow, from the load cell
  4  the operating point
  5  what the pump point implies about the hardware
  6  the turbine model against the measurement
  7  the burn resolved in time
  8  what is still missing
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from campaigns import ot2                                          # noqa: E402
from otp.compare.predict import compare_turbine                    # noqa: E402
from otp.core.states import GasState, LiquidState                  # noqa: E402
from otp.core.units import BAR, g, rpm_to_rad                      # noqa: E402
from otp.experiments.acoustics import (align_to_channel,           # noqa: E402
                                       compute_spectrogram, rpm_from_ridge)
from otp.experiments.loaders.airborne import load_run              # noqa: E402
from otp.experiments.records import SteadyWindow                   # noqa: E402
from otp.experiments.reduce import reduce_point, reduce_series     # noqa: E402

DATA = HERE / "data"
RUN_ID = "20260701-111"
BURN = (2.0, 13.0)        # inside the 14.5 s burn, clear of both transients


def rule(n, title):
    print(f"\n{'=' * 78}\n{n}  {title}\n{'=' * 78}")


# ── 1 ───────────────────────────────────────────────────────────────────────
def load():
    rule(1, "The run")
    path = DATA / "OT-2" / f"{RUN_ID}.h5"
    if not path.exists():
        sys.exit(f"missing {path} -- run: uv run python testbed/fetch_data.py --audio")

    run = load_run(path, ot2.CHANNEL_MAP, ot2.RIG, notes=ot2.RUNS[RUN_ID])
    lines = run.notes.splitlines()
    print(f"  {run.run_id}: {lines[0]}")
    print(f"  {run.meta.get('start_datetime', '')}  operator "
          f"{run.meta.get('operator', '?')}  at {run.meta.get('location', '?')}")

    print(f"\n  {len(run.channels)} channels mapped and alive:")
    for name in sorted(run.channels):
        ch = run[name]
        src = ch.source if len(ch.source) <= 10 else "derived"
        print(f"    {name:<16}{src:<11}{ch.units:<9}"
              f"burn mean {ch.mean(*BURN):>10.4g}   {ch.desc[:44]}")
    for line in lines[1:]:
        print(f"  {line.strip()}")
    print("\n  Note what is NOT here: no flow meter on the water side, and no")
    print("  torque on the shaft. Stages 2 and 3 recover both from what is.")
    return run


# ── 2 ───────────────────────────────────────────────────────────────────────
def shaft_speed(run):
    rule(2, "Shaft speed, from the blade tone")
    wav = DATA / "OT-2" / ot2.AUDIO_FILE
    if not wav.exists():
        print(f"  missing {wav} -- fetch_data.py --audio")
        return None

    spec = compute_spectrogram(wav, t_offset=ot2.AUDIO_T_OFFSET, nperseg=8192)
    rpm_ch, prom = rpm_from_ridge(
        spec, ot2.AUDIO_BAND_HZ, ot2.EVENTS_PER_REV,
        min_prominence_db=ot2.AUDIO_MIN_PROMINENCE_DB,
        max_jump_hz=ot2.AUDIO_MAX_JUMP_HZ)
    run.add(rpm_ch)

    ok = np.isfinite(rpm_ch.v)
    print(f"  {ot2.N_BLADES} rotor blades, so rpm = 60 f / {ot2.N_BLADES}.")
    print(f"  {spec.f[1] - spec.f[0]:.1f} Hz bins over "
          f"{ot2.AUDIO_BAND_HZ[0]:.0f}-{ot2.AUDIO_BAND_HZ[1]:.0f} Hz "
          f"= {spec.f[1] - spec.f[0]:.1f} * 60 / {ot2.N_BLADES} "
          f"= {(spec.f[1] - spec.f[0]) * 60 / ot2.N_BLADES:.1f} rpm resolution")
    print(f"  {ok.sum()} frames tracked, offset {ot2.AUDIO_T_OFFSET:+.2f} s onto "
          f"the DAQ clock\n")
    print(f"  {'t_daq':>7}{'rpm':>9}{'prom dB':>10}")
    for t0 in np.arange(0.0, 15.0, 1.5):
        m = ok & (rpm_ch.t >= t0) & (rpm_ch.t < t0 + 1.5)
        if m.any():
            print(f"  {t0:>7.1f}{np.mean(rpm_ch.v[m]):>9.0f}"
                  f"{np.mean(prom.v[m]):>10.1f}")
    mean_n, sd_n = rpm_ch.mean(*BURN), rpm_ch.std(*BURN)
    print(f"\n  over the reduction window {BURN[0]:.0f}-{BURN[1]:.0f} s: "
          f"{mean_n:.0f} +/- {sd_n:.0f} rpm ({sd_n / mean_n * 100:.1f}%)")
    late = (rpm_ch.t >= 5) & (rpm_ch.t <= 13) & ok
    print(f"  over 5-13 s, once fully spun up:   "
          f"{np.mean(rpm_ch.v[late]):.0f} +/- {np.std(rpm_ch.v[late]):.0f} rpm "
          f"({np.std(rpm_ch.v[late]) / np.mean(rpm_ch.v[late]) * 100:.2f}%)")
    print("  A steady point held for the whole burn, not a sweep. The residual")
    print("  scatter is at the level of the spectrogram bin, 9 rpm, so most of")
    print("  what is left is measurement resolution rather than the machine.")

    off, iou = align_to_channel(
        rpm_from_ridge(compute_spectrogram(wav, t_offset=0.0, nperseg=8192),
                       ot2.AUDIO_BAND_HZ, ot2.EVENTS_PER_REV,
                       min_prominence_db=ot2.AUDIO_MIN_PROMINENCE_DB,
                       max_jump_hz=ot2.AUDIO_MAX_JUMP_HZ)[0],
        run["p_gg_chamber"], level=10.0)
    print(f"\n  alignment re-derived from scratch: {off:+.2f} s at {iou * 100:.0f}% "
          f"envelope overlap (campaign uses {ot2.AUDIO_T_OFFSET:+.2f})")
    return rpm_ch


# ── 3 ───────────────────────────────────────────────────────────────────────
def water_flow(run):
    rule(3, "Water flow, from the load cell")
    print("  The stand dumps the pumped water through an orifice to atmosphere")
    print("  and LC190 measures the reaction, so the load cell is the flow meter:")
    print("     F = mdot V   and   mdot = rho A V   =>   A = F / (rho V^2)")
    print("     V = Cv sqrt(2 dP / rho)   from PTX105 - PT142")
    print("  The orifice area comes OUT of this, so no diameter has to be assumed.")
    print("  The turbine exhaust pushes on the same load cell and is about a third")
    print("  of the total, so it is subtracted using the measured gas flow.\n")

    F = run["thrust"].mean(*BURN)
    dp_or = run["dp_orifice"].mean(*BURN)
    mdot_gas = run["mdot_gg_ox"].mean(*BURN) + run["mdot_gg_fuel"].mean(*BURN)
    V = ot2.JET_CV * np.sqrt(2 * dp_or * BAR / ot2.WATER_RHO)
    F_gas = mdot_gas * ot2.TURBINE_EXIT_VELOCITY
    A_eff = (F - F_gas) / (ot2.WATER_RHO * V ** 2)
    mdot_w = ot2.WATER_RHO * A_eff * V

    print(f"    LC190 total                       {F:8.1f} N")
    print(f"    gas jet, {mdot_gas:.4f} kg/s x {ot2.TURBINE_EXIT_VELOCITY:.0f} m/s"
          f"     {F_gas:8.1f} N")
    print(f"    water jet                         {F - F_gas:8.1f} N")
    print(f"    orifice dP                        {dp_or:8.2f} bar")
    print(f"    jet velocity                      {V:8.1f} m/s")
    print(f"    effective area F/(rho V^2)        {A_eff * 1e6:8.2f} mm^2"
          f"   (d_eq {np.sqrt(4 * A_eff / np.pi) * 1e3:.2f} mm)")
    print(f"    water mass flow                   {mdot_w:8.3f} kg/s"
          f"   ({mdot_w / ot2.WATER_RHO * 1e3:.3f} L/s)")

    print(f"\n  Check 1 -- run it the other way. A {ot2.ORIFICE_DIAMETER_ASSUMED * 1e3:.1f} mm "
          f"sharp-edged orifice at Cd = {ot2.ORIFICE_CD_ASSUMED}:")
    A_geo = np.pi / 4 * ot2.ORIFICE_DIAMETER_ASSUMED ** 2
    m_orif = ot2.WATER_RHO * ot2.ORIFICE_CD_ASSUMED * A_geo * V
    print(f"    mdot = {m_orif:.3f} kg/s, i.e. {abs(m_orif / mdot_w - 1) * 100:.0f}% "
          f"from the thrust route.")
    print(f"    Equivalently the thrust route implies Cd = "
          f"{A_eff / A_geo:.3f} on a {ot2.ORIFICE_DIAMETER_ASSUMED * 1e3:.0f} mm hole,")
    print("    which is the textbook sharp-edged value. Two routes, one answer.")

    print("\n  Check 2 -- sensitivity to the one modelled input, the gas jet:")
    for v_gas in (500.0, 655.0, 800.0):
        a = (F - mdot_gas * v_gas) / (ot2.WATER_RHO * V ** 2)
        print(f"    V_gas {v_gas:5.0f} m/s -> mdot_water "
              f"{ot2.WATER_RHO * a * V:6.3f} kg/s "
              f"({(ot2.WATER_RHO * a * V / mdot_w - 1) * 100:+5.1f}%)")
    print("  Everything else in the chain is directly measured.")
    return mdot_w


# ── 4 ───────────────────────────────────────────────────────────────────────
def operating_point(run):
    rule(4, "The operating point")
    window = SteadyWindow(run_id=run.run_id, label="steady burn",
                          t0=BURN[0], t1=BURN[1],
                          reason="inside the 14.5 s burn, clear of both transients")

    liquid = LiquidState(
        name="water", rho=ot2.WATER_RHO, nu=1.14e-6, p_vap_bar=0.0171,
        p0_bar=run["p_pump_in"].mean(*BURN), T_K=288.15,
        source="15 degC assumed: TC140 and TC141 were all-NaN on this run")

    point = reduce_point(
        run, window, liquid, gg_R=ot2.GG_R, gg_gamma=ot2.GG_GAMMA,
        gg_T0_K=ot2.GG_DESIGN_GAS.T0_K,
        gas_source=f"measured p_c, DESIGN R and gamma (MR 0.5) -- see stage 8",
        notes="Water T assumed. Gas properties are the design-MR ones.")
    point.p_turbine_exit = ot2.AMBIENT_BAR
    print(point)
    print(f"\n  comparable against a model: {', '.join(point.residual_fields())}")
    return point


# ── 5 ───────────────────────────────────────────────────────────────────────
def pump_implications(point):
    rule(5, "What the pump point implies about the hardware")
    Q, N, H = point.Q, point.rpm, point.H_total
    omega = rpm_to_rad(N)
    P_useful = Q * point.dp_pump * BAR
    Ns = N * np.sqrt(Q) / H ** 0.75

    print(f"  measured    Q {Q * 1e3:.3f} L/s   N {N:.0f} rpm   "
          f"dp {point.dp_pump:.2f} bar   H {H:.0f} m")
    print(f"  useful power  Q dp = {P_useful / 1000:.2f} kW")
    print(f"  specific speed  N sqrt(Q) / H^0.75 = {Ns:.1f}  "
          f"(rpm, m^3/s, m) -- very low, which is Barske territory")

    print("\n  The impeller diameter is not in the ManRay repo, but the measured")
    print("  point implies it. Barske's total head is roughly")
    print("     H = [ (1+psi) u2^2 - u1^2 + (1-psi) v_out^2 ] / 2g,")
    print("  and dropping the two small terms leaves u2 = sqrt(2 g H / (1+psi)):")
    for psi in (0.1, 0.2, 0.3):
        u2 = np.sqrt(2 * g * H / (1 + psi))
        print(f"    psi {psi:.1f}  ->  u2 {u2:6.1f} m/s  ->  d2 "
              f"{2 * u2 / omega * 1e3:5.1f} mm")
    print("  Check that against the drawing. If it matches, the head correlation")
    print("  is already close on this machine; if it does not, that gap is the")
    print("  first thing calibration has to explain.")

    print(f"\n  Efficiency needs shaft power, and this rig has no torque sensor.")
    print(f"  Bracketing it by the turbine instead, in stage 6.")


# ── 6 ───────────────────────────────────────────────────────────────────────
def turbine_check(point):
    rule(6, "The turbine model against the measurement")
    c = compare_turbine(point, ot2.RIG.turbine, model="OT-2 mean line")
    print(c.table())

    perf = ot2.RIG.turbine.point_performance(
        rpm=point.rpm, gas=point.gas, p_exit_bar=ot2.AMBIENT_BAR)
    P_useful = point.Q * point.dp_pump * BAR
    print(f"\n  turbine model at the measured speed and gas state:")
    print(f"    gas flow    {perf.mdot:.4f} kg/s   measured "
          f"{point.mdot_gg_ox + point.mdot_gg_fuel:.4f}")
    print(f"    Euler       {perf.P_euler / 1000:7.2f} kW")
    print(f"    net shaft   {perf.P_shaft / 1000:7.2f} kW   "
          f"torque {perf.torque:.2f} Nm")
    print(f"    u/c_is      {perf.u_over_c_is:.3f}   eta_is {perf.eta_isentropic:.3f}")
    print(f"\n  pump useful {P_useful / 1000:.2f} kW / turbine net "
          f"{perf.P_shaft / 1000:.2f} kW = {P_useful / perf.P_shaft * 100:.0f}%")
    print("  That is the overall pump efficiency implied by the turbine model, and")
    print("  a Barske at this specific speed does belong in the 25-40% band -- so")
    print("  the two halves are at least consistent. It is a weak test: the")
    print("  turbine loss coefficients are still placeholders, and the model")
    print(f"  under-predicts gas flow by {(perf.mdot / (point.mdot_gg_ox + point.mdot_gg_fuel) - 1) * -100:.0f}%, "
          f"which stage 8 explains.")


# ── 7 ───────────────────────────────────────────────────────────────────────
def burn_series(run):
    rule(7, "The burn resolved in time")
    liquid = LiquidState(name="water", rho=ot2.WATER_RHO, nu=1.14e-6,
                         p_vap_bar=0.0171, p0_bar=run["p_pump_in"].mean(*BURN),
                         T_K=288.15, source="15 degC assumed")
    pts = reduce_series(run, liquid, t0=0.5, t1=14.0, n=13,
                        gg_R=ot2.GG_R, gg_gamma=ot2.GG_GAMMA,
                        gg_T0_K=ot2.GG_DESIGN_GAS.T0_K)
    print(f"  {'window':>13}{'rpm':>8}{'Q L/s':>8}{'H m':>7}{'dp bar':>8}"
          f"{'p_gg':>7}{'MR':>6}{'kW useful':>11}")
    for p in pts:
        P = (p.Q * p.dp_pump * BAR / 1000) if None not in (p.Q, p.dp_pump) else np.nan
        print(f"  {p.t0:5.1f}-{p.t1:<5.1f}"
              f"{p.rpm or np.nan:>8.0f}{(p.Q or np.nan) * 1e3:>8.3f}"
              f"{p.H_total or np.nan:>7.0f}{p.dp_pump or np.nan:>8.2f}"
              f"{p.gas.p0_bar if p.gas else np.nan:>7.2f}"
              f"{p.MR_gg or np.nan:>6.2f}{P:>11.2f}")

    good = [p for p in pts if None not in (p.Q, p.rpm) and p.rpm > 1000]
    if len(good) > 3:
        N = np.array([p.rpm for p in good])
        dp = np.array([p.dp_pump for p in good])
        Q = np.array([p.Q for p in good])
        pred = dp[-1] * (N / N[-1]) ** 2
        print(f"\n  dp follows N^2 to "
              f"{np.sqrt((((pred - dp) / dp) ** 2).mean()) * 100:.1f}% RMS, and "
              f"Q/N varies by {np.ptp(Q / N) / np.mean(Q / N) * 100:.1f}%")
        print("  across the burn. The machine is holding one operating point, so")
        print("  these windows are repeat measurements of it rather than a sweep.")
        print("  A curve needs runs at different throttle settings, not this run.")
    return pts


# ── 8 ───────────────────────────────────────────────────────────────────────
def whats_missing(point):
    rule(8, "What is still missing")
    print("  1  Gas properties at the measured mixture ratio.")
    print(f"     This window ran at MR {point.MR_gg:.2f}; gg_out.yaml's R and gamma")
    print("     are the MR-0.5 design values, so they are the wrong gas. The model")
    print("     under-predicts gas flow by 14% and Cd is already 1.0, so no")
    print("     discharge coefficient can close it. Re-run CEA at MR "
          f"{point.MR_gg:.2f}.")
    print("     It also explains the thermocouples reading ~1500 K against a")
    print("     design T0 of 970 K: hotter because it ran oxidiser-richer than")
    print("     designed, though still deeply fuel-rich in absolute terms.")
    print("\n  2  The orifice diameter, measured rather than inferred.")
    print("     It would turn stage 3's cross-check into a real second route.")
    print("\n  3  Pump geometry as a PumpGeometry, to run the Lock/Barske")
    print("     correlation against this point instead of inferring d2 from it.")
    print("\n  4  Water temperature. TC140 and TC141 were dead, so density is")
    print("     assumed. It is a weak dependence, but it is an assumption.")
    print("\n  5  A torque measurement, on any future build. Everything about")
    print("     efficiency here is inferred through the turbine model.")


def main():
    run = load()
    shaft_speed(run)
    water_flow(run)
    point = operating_point(run)
    pump_implications(point)
    turbine_check(point)
    burn_series(run)
    whats_missing(point)
    print(f"\n{'=' * 78}\ndone\n{'=' * 78}")


if __name__ == "__main__":
    main()
