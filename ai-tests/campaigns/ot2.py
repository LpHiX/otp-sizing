"""OT-2 -- "Man-Ray and The Dirty Bubble", R2S 2026, J1 test facility, 2026-07-01.

N2O/IPA gas generator -> partial-admission supersonic turbine -> water Barske pump.

Every number below is traced to a file.  Nothing here is a guess except where it
says ESTIMATE, and those are the inputs to go and measure.

Sources
  turbine geometry, gas state   ManRayAndTheDirtyBubble/fluidics/turbine/config/
                                turbine_in.yaml, turbine_out.yaml, gg_out.yaml
  channel roles                 read from the h5 attrs of 20260701-111.h5

Test matrix, per Elias:
  101-108  failed GG ignitions
  109      first success, 3 s spin-up, ox rich
  110      fuel rich 3 s spin-up
  111      fuel rich 15 s qualification burn   <- the useful one
"""

import numpy as np

from otp.core.states import GasState
from otp.experiments.loaders.airborne import (ChannelMap, difference,
                                              jet_thrust_flow)
from otp.experiments.records import Rig
from otp.turbine.analysis import Turbine
from otp.turbine.records import TurbineCalibration, TurbineGeometry

CAMPAIGN = "OT-2"

# ── turbine geometry ────────────────────────────────────────────────────────
# turbine_out.yaml: D_m 0.0992300509403079, s_b (blade height) 0.011183177728956087,
# "Nozzle Exit (mm)" 8.683177728956087, "Nozzle Throat (mm)" 3.6111908136224455.
# The mm figures are the equivalent circular diameter of ONE nozzle, so the total
# areas are n * pi * (d/2)^2.  The throat total reproduces gg_out.yaml a_t
# (6.145284669949664e-05) to 6 figures, which is the cross-check that the GG
# throat and the nozzle throat are the same restriction.
D_MEAN = 0.0992300509403079
N_NOZZLES = 6
N_BLADES = 35
BLADE_HEIGHT = 0.011183177728956087
NOZZLE_THROAT_D = 3.6111908136224455e-3
NOZZLE_EXIT_D = 8.683177728956087e-3

A_THROAT = N_NOZZLES * np.pi * (NOZZLE_THROAT_D / 2) ** 2      # 6.1453e-5 m^2
A_EXIT = N_NOZZLES * np.pi * (NOZZLE_EXIT_D / 2) ** 2          # 3.5530e-4 m^2

# Admission fraction: the arc the nozzle exits actually cover.
ADMISSION_FRACTION = N_NOZZLES * NOZZLE_EXIT_D / (np.pi * D_MEAN)   # 0.167

# turbine_in.yaml `alpha: 70` is measured FROM THE AXIS.  This package measures
# from tangential, so 90 - 70 = 20 deg.  Checked against turbine_out.yaml:
# c_1 = 1324.897, u = 124.696 give w_1 = 1208.47 at alpha_1 = 20 deg from
# tangential, matching their w_1 = 1208.4734 to 6 figures.
ALPHA_1_DEG = 90 - 70
# turbine_out.yaml angles are already from tangential for OT-2: beta_2 = 0.5205.
BETA_2_DEG = 0.5205480276404707

# ESTIMATE. OT-2 does not publish a rotor chord.  OT-1 built its blades at a
# solidity (chord/pitch) of 1.826; the same solidity on OT-2's 35-blade,
# 99.2 mm pitch diameter gives 16.3 mm.  Only the sector loss uses it.
BLADE_CHORD = 1.826 * (np.pi * D_MEAN / N_BLADES)

TURBINE_GEOMETRY = TurbineGeometry(
    d_mean=D_MEAN, A_throat=A_THROAT, A_exit=A_EXIT, n_nozzles=N_NOZZLES,
    n_blades=N_BLADES, alpha_1_deg=ALPHA_1_DEG, beta_2_deg=BETA_2_DEG,
    blade_height=BLADE_HEIGHT, blade_chord=BLADE_CHORD,
    admission_fraction=ADMISSION_FRACTION, tip_clearance=2.0e-3,
    source="ManRay fluidics/turbine/config/turbine_out.yaml (as built design)")

# phi_n from turbine_in.yaml; phi_r is turbine_out.yaml's phi_r = w_2/w_1,
# which reproduces their w_2 = 812.85 from w_1 = 1208.47.  The three loss
# coefficients are NOT from turboRocket -- they are this package's placeholders
# and are the things calibration exists to replace.
TURBINE_CALIBRATION = TurbineCalibration(
    nozzle_cd=1.0,                       # turboRocket's design mdot implies Cd = 1
    nozzle_velocity_coeff=0.85,
    rotor_velocity_coeff=0.6726262087243714,
    source="phi_n, phi_r from turbine_in/turbine_out.yaml; loss k's are placeholders")

# Design-point gas, gg_out.yaml.  A measured run replaces this via reduce_point.
GG_DESIGN_GAS = GasState(p0_bar=25.00008468737293, T0_K=970.1654223020483,
                         R=537.4735344209124, gamma=1.2498281570181644,
                         source="ManRay gg_out.yaml design point, MR 0.5")
GG_R = GG_DESIGN_GAS.R
GG_GAMMA = GG_DESIGN_GAS.gamma

# turbine_in.yaml Rt = 23 is the nozzle expansion ratio, so the design static
# pressure the nozzle expands to is p0/23 = 1.087 bar -- slightly overexpanded
# against a 1.013 bar ambient.
DESIGN_P_EXIT_BAR = 25.00008468737293 / 23
AMBIENT_BAR = 1.01325

RIG = Rig(name="OT-2 Man-Ray and The Dirty Bubble",
          turbine=Turbine(TURBINE_GEOMETRY, TURBINE_CALIBRATION),
          pump=None,
          pumped_fluid="water",
          gg_propellants="N2O / IPA",
          notes="Pump geometry is not in the ManRay repo in a form this package "
                "can consume; supply a PumpGeometry to enable pump predictions.")

# ── WATER FLOW ──────────────────────────────────────────────────────────────
# There is no flow meter on the water side.  There is something better: the
# stand dumps the pumped water through an orifice to atmosphere and LC190
# measures the reaction, so the load cell IS the flow meter.
#
#   F = mdot * V   and   mdot = rho * A_eff * V   =>   A_eff = F / (rho * V^2)
#   V = Cv * sqrt(2 * dP / rho),  dP = PTX105 - PT142
#
# The orifice diameter cancels.  A_eff comes out of the measurement instead of
# going into it, which is what makes this a measurement rather than an estimate.
#
# The turbine exhaust pushes on the same load cell and is NOT negligible -- it is
# about a third of the total -- so it is subtracted using the measured gas flow
# and the turbine's exit velocity.
#
# Run 111, averaged 2-13 s:
#   LC190                     331 N
#   minus gas jet   0.178 kg/s * 655 m/s = 117 N
#   water jet                 214 N
#   dP orifice               87.5 bar  ->  V = 130 m/s
#   A_eff = F/(rho V^2)      12.7 mm^2 (equivalent diameter 4.02 mm)
#   mdot  = rho A_eff V      1.64 kg/s   (1.64 L/s)
#
# Two checks that this is right rather than merely arithmetic:
#   * A_eff of 12.7 mm^2 is a 5.0 mm orifice at Cd = 0.65, which is the textbook
#     sharp-edged value.  Running it the other way -- assume 5 mm and Cd = 0.62 --
#     gives 1.57 kg/s, agreeing with the thrust route to 4%.
#   * The implied pump useful power, Q*dp = 12.1 kW against a turbine net of
#     ~35 kW, puts overall pump efficiency near 34%, which is where a Barske at
#     specific speed 8.4 belongs.
#
# The weak link is the gas subtraction: +/-20% on the turbine exit velocity moves
# the water flow by -/+12%.  Everything else is directly measured.
WATER_RHO = 998.0                  # 15 degC; TC140/TC141 were dead on this run
JET_CV = 0.98                      # jet velocity coefficient
TURBINE_EXIT_VELOCITY = 655.0      # c_2 from the turbine model at 29,800 rpm
                                   # on the measured gas state; design was 707

# If you later measure the orifice, this is the cross-check:
ORIFICE_DIAMETER_ASSUMED = 5.0e-3
ORIFICE_CD_ASSUMED = 0.62


# ── channel map ─────────────────────────────────────────────────────────────
# Read from 20260701-111.h5 attrs, 2026-09-10.  These roles are NOT the same as
# OT-1's: here PTX103 is the GG chamber and PTX105 is the orifice inlet, whereas
# on OT-1 PTX105 was the GG chamber.
CHANNEL_MAP = ChannelMap(
    campaign=CAMPAIGN,
    mapping={
        "p_pump_in":      "PTX101",   # PTX101 Pump Inlet
        "p_pump_seal":    "PTX102",   # PTX102 Pump Shaft Seal
        "p_gg_chamber":   "PTX103",   # PTX103 GG Chamber
        "p_pump_out":     "PTX104",   # PTX104 Pump Outlet
        "p_orifice_in":   "PTX105",   # PTX105 Orifice Inlet
        "T_gg_fore":      "TCX101",   # TCX101 GG Chamber Fore
        "T_gg_aft":       "TCX102",   # TCX102 GG Chamber Aft
        "mdot_gg_fuel":   "M730",     # M730 IPA Massflow
        "mdot_gg_ox":     "M850",     # M850 N2O Massflow
        "p_ipa_delivery": "PT732",
        "p_n2o_delivery": "PT852",
        "rho_ipa":        "DT730",
        "rho_n2o":        "DT850",
        "thrust":         "LC190",    # LC190 -- the water flow measurement
        "p_water_supply": "PT140",    # flatlined on 111; the loader drops it
        "p_water_return": "PT141",    # flatlined on 111; the loader drops it
        "p_water_drain":  "PT142",
    },
    gauge_to_absolute=("p_pump_in", "p_pump_seal", "p_gg_chamber", "p_pump_out",
                       "p_orifice_in", "p_ipa_delivery", "p_n2o_delivery",
                       "p_water_supply", "p_water_return", "p_water_drain"),
    derive={
        "dp_orifice": difference("p_orifice_in", "p_water_drain", "dp_orifice", "bar",
                                 "pressure across the discharge orifice"),
        "dp_pump": difference("p_pump_out", "p_pump_in", "dp_pump", "bar",
                              "pump pressure rise"),
        # See WATER FLOW below.  This is the flow meter this rig actually has.
        "Q_pump": jet_thrust_flow(
            thrust="thrust", dp_channel="dp_orifice", rho=WATER_RHO,
            gas_flow_channels=("mdot_gg_ox", "mdot_gg_fuel"),
            gas_exit_velocity=TURBINE_EXIT_VELOCITY, Cv=JET_CV),
    },
    notes="pressures recorded bar(g), converted to absolute on load")

RUNS = {
    "20260701-109": "first successful ignition, 3 s spin-up, ox rich",
    "20260701-110": "fuel rich 3 s spin-up",
    "20260701-111": "fuel rich 15 s qualification burn",
}

# ── audio ───────────────────────────────────────────────────────────────────
# The blade tone divides by the ROTOR BLADE COUNT, 35.
EVENTS_PER_REV = N_BLADES
AUDIO_FILE = "20260701-111_Canon_700D.wav"

# Band around the ridge actually present on run 111: it enters at ~16.0 kHz and
# settles at ~17.4 kHz, i.e. 27,500 -> 29,850 rpm on 35 blades.  Kept tight on
# purpose.  A wide band invites the tracker onto the 8-9 kHz broadband stand
# noise, which is about 10 dB LOUDER than the tone in raw level -- see the
# `whiten` argument of track_ridge, which is what makes the tone findable.
AUDIO_BAND_HZ = (15_000.0, 19_000.0)          # 25.7k to 32.6k rpm
AUDIO_RPM_RANGE = (25_000.0, 33_000.0)

# Audio-to-DAQ offset, measured 2026-09-10 with `acoustics.align_to_channel`,
# which maximises the overlap between the tone's on/off envelope and
# p_gg_chamber > 10 bar.  Reproduce it with:
#
#   spec = compute_spectrogram(wav, t_offset=0.0, nperseg=8192)
#   rpm, _ = rpm_from_ridge(spec, AUDIO_BAND_HZ, EVENTS_PER_REV,
#                           min_prominence_db=10.0, max_jump_hz=400.0)
#   align_to_channel(rpm, run["p_gg_chamber"], level=10.0)   # -> (14.00, 0.938)
#
# Uncertainty is about +/- 0.6 s and cannot be reduced with this recording:
#   tone tracked, audio clock : 12.82 -> 28.51 s   (15.69 s)
#   GG chamber above 10 bar   :  0.03 -> 14.54 s   (14.51 s)
# Neither edge is a clean fiducial.  The tone only enters the band once the
# rotor is already past ~25,700 rpm, and it outlives chamber pressure while the
# rotor coasts down, so the tone is 1.2 s longer than the burn at both ends.
# Correlating audio N^2 against the measured pump rise does NOT sharpen this --
# both signals are flat through the burn, so that correlation is dominated by
# noise and drifts to whatever bound it is given.  Tried, and rejected.
#
# It does not matter here: the speed is flat at ~29,800 rpm for the whole burn,
# so +/- 0.6 s moves the rpm assigned to any window by well under 1%.  It WOULD
# matter on a run with a real speed sweep, and then a shared fiducial event --
# an igniter flash visible in frame, a valve crack audible on the track -- has
# to be found rather than inferred.
#
# Cross-check that the tone really is the shaft: the audio speed and the
# measured pump pressure rise satisfy dp ~ N^2 to better than 2% RMS across the
# burn (the notebook computes it).  That test is independent of the blade count
# -- it confirms the tone tracks the shaft, not what the divisor is.  The
# divisor of 35 comes from the rotor itself, via turbine_out.yaml.
AUDIO_T_OFFSET = 14.00

# Settings that actually find the tone.  Prominence is measured against each
# frequency bin's own background, not against the rest of the band.
AUDIO_MIN_PROMINENCE_DB = 10.0
AUDIO_MAX_JUMP_HZ = 400.0
