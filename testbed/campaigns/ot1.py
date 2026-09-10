"""OT-1 -- "Mermaid Man and Barnacle Boy", R2S 2025, J1 test facility, 2025-07-09.

N2O/IPA gas generator -> partial-admission supersonic turbine -> IPA Barske pump
feeding an engine.  Unlike OT-2, OT-1 measured the pumped-fluid flow, so it is
the campaign that can produce a full pump curve.

Sources
  turbine geometry, gas state   MermaidManBarnacleBoy/docs/fluidics/turbine_design.ipynb
                                (dic_gg, dic_gg_geom, dic_rotor, and the printed
                                geometry / velocity / angle tables)
  RPM method                    MermaidManBarnacleBoy/docs/test/R2S_2025.ipynb
  channel roles                 read from the h5 attrs of 20250709-002-release.h5

Angle-convention warning: OT-1's notebook prints beta from the AXIAL direction
(beta_1 67.41, beta_2 89.45) while OT-2's turbine_out.yaml prints from
TANGENTIAL (beta_1 22.02, beta_2 0.52).  Both are converted to this package's
from-tangential convention here.  Verified: at alpha_1 = 20 deg from tangential,
c_1 = 1110.65 and u = 130.66 give w_1 = 988.9, matching their w_1 = 988.873.
"""

import numpy as np

from otp.core.states import GasState
from otp.experiments.loaders.airborne import ChannelMap, difference
from otp.experiments.records import Rig
from otp.turbine.analysis import Turbine
from otp.turbine.records import TurbineCalibration, TurbineGeometry

CAMPAIGN = "OT-1"

# ── turbine geometry ────────────────────────────────────────────────────────
D_MEAN = 0.103979
N_NOZZLES = 8
N_BLADES = 18
BLADE_HEIGHT = 0.007493              # s_b; D_tip 0.111472 - D_hub 0.096487
NOZZLE_EXIT_D = 0.004993             # s_c
A_THROAT = 4.45357383e-05            # dic_gg_geom["Acc"], matches A_0 to 0.1%
A_EXIT = 0.000157                    # A_1
BLADE_CHORD = 0.033144014338487      # printed "Blade Chord Length: 33.14 mm"

ADMISSION_FRACTION = N_NOZZLES * NOZZLE_EXIT_D / (np.pi * D_MEAN)    # 0.122

ALPHA_1_DEG = 90 - 70                # dic_rotor["alpha"] = 70 from the axis
BETA_2_DEG = 90 - 89.449438          # printed beta_2 is from the axis

TURBINE_GEOMETRY = TurbineGeometry(
    d_mean=D_MEAN, A_throat=A_THROAT, A_exit=A_EXIT, n_nozzles=N_NOZZLES,
    n_blades=N_BLADES, alpha_1_deg=ALPHA_1_DEG, beta_2_deg=BETA_2_DEG,
    blade_height=BLADE_HEIGHT, blade_chord=BLADE_CHORD,
    admission_fraction=ADMISSION_FRACTION, tip_clearance=2.0e-3,
    source="MermaidManBarnacleBoy docs/fluidics/turbine_design.ipynb (as built design)")

TURBINE_CALIBRATION = TurbineCalibration(
    nozzle_cd=1.0,
    nozzle_velocity_coeff=0.85,                  # dic_rotor["phi_n"]
    rotor_velocity_coeff=0.603743,               # printed phi_r = w_2/w_1
    source="phi_n, phi_r from turbine_design.ipynb; loss k's are placeholders")

GG_DESIGN_GAS = GasState(p0_bar=25.0, T0_K=868.1694130102591,
                         R=504.5591863293963, gamma=1.290332008352664,
                         source="turbine_design.ipynb dic_gg, MR 1.0")
GG_R = GG_DESIGN_GAS.R
GG_GAMMA = GG_DESIGN_GAS.gamma

DESIGN_P_EXIT_BAR = 25.0 / 13        # dic_rotor["Rt"] = 13
AMBIENT_BAR = 1.01325

RIG = Rig(name="OT-1 Mermaid Man and Barnacle Boy",
          turbine=Turbine(TURBINE_GEOMETRY, TURBINE_CALIBRATION),
          pump=None,
          pumped_fluid="IPA",
          gg_propellants="N2O / IPA",
          notes="Pump was a Barske sized with turboRocket's Barske class; its "
                "diameters are not published in a form this package can consume. "
                "The measured curve route (similarity) works without them.")

# ── channel map ─────────────────────────────────────────────────────────────
# Read from 20250709-002-release.h5 attrs, 2026-09-10.
# Pump head on OT-1 is PTX104 (Pump Outlet) minus PT732 (IPA Delivery), which is
# what R2S_2025.ipynb plotted as "Head Rise".
CHANNEL_MAP = ChannelMap(
    campaign=CAMPAIGN,
    mapping={
        "p_pump_in":       "PT732",    # PT732 IPA Delivery -- pump suction
        "p_pump_out":      "PTX104",   # PTX104 Pump Outlet
        "p_gg_chamber":    "PTX105",   # PTX105 GG Chamber   (NOTE: PTX103 on OT-2)
        "p_engine_in":     "PTX101",   # PTX101 Engine IPA Inlet -- flatlined on 002
        "p_engine_inj":    "PTX102",   # PTX102 Engine IPA Injector
        "p_engine_c":      "PTX103",   # PTX103 Engine Chamber
        "T_gg_chamber":    "TCX104",   # soak thermocouple, NOT a gas temperature
        "mdot_pump":       "M730",     # M730 IPA Massflow -- the pumped fluid
        "mdot_gg_ox":      "M850",     # M850 N2O Massflow
        "p_ipa_throttle":  "PT733",
        "p_n2o_delivery":  "PT852",
        "rho_ipa":         "DT730",
        "rho_n2o":         "DT850",
        "thrust":          "LC190",
    },
    gauge_to_absolute=("p_pump_in", "p_pump_out", "p_gg_chamber", "p_engine_in",
                       "p_engine_inj", "p_engine_c", "p_ipa_throttle",
                       "p_n2o_delivery"),
    derive={
        "dp_pump": difference("p_pump_out", "p_pump_in", "dp_pump", "bar",
                              "pump pressure rise, the R2S_2025 head rise"),
    },
    notes="TCX104 reads 33-66 degC on a nominally 868 K gas: it is a wall/soak "
          "reading, not the combustion temperature. Pass gg_T0_K explicitly "
          "rather than reducing it from this channel.")

RUNS = {
    "20250709-001-release": "1 s GG spin, functional demonstration",
    "20250709-002-release": "2 s GG spin -- the run R2S_2025.ipynb analysed",
    "20250709-003-release": "third firing",
    "20250709-004-release": "fourth firing",
}

# ── audio ───────────────────────────────────────────────────────────────────
# R2S_2025.ipynb used `N = f * 60 / 18` -- the 18-blade rotor.
EVENTS_PER_REV = N_BLADES
AUDIO_FILE = "20250709-002_2160p60.wav"
# Frame 161 at 60 fps is DAQ t = 0, per R2S_2025.ipynb cell 57.
AUDIO_T_OFFSET = 161 / 60
AUDIO_BAND_HZ = (300.0, 5_000.0)              # ~1k to ~16.7k rpm on 18 blades

# The hand-read ridge from R2S_2025.ipynb cell 57, kept verbatim so the published
# OT-1 numbers stay reproducible.  Times are already on the DAQ base.
MANUAL_RIDGE_T = [0, 1, 1.25, 1.45, 1.9, 2.5]
MANUAL_RIDGE_F = [500, 500, 600, 1100, 4000, 3300]
