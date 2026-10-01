import os

import numpy as np

from otp.pump.records import PumpGeometry, PumpRequirements, PumpChoices
from otp.pump.analysis import Pump
from otp.experiments.loaders.airborne import orifice_flow, difference, ChannelMap, csv_channel
from otp.experiments.records import Run

from pathlib import Path

DATA_PATH = Path(__file__).parent / '20260630-204.h5'

CHANNEL_MAP = ChannelMap(
    mapping={
        # sequencer / AMV
        "n2o_amv_demand":        "DAU1002_demand",         # N2O AMV Demand (deg)
        "n2o_amv_target":        "DAU1002_mdot_req",       # N2O AMV Target (kg/s)
        "ipa_amv_demand":        "DAU1003_demand",         # IPA AMV Demand (turns)
        "ipa_amv_target":        "DAU1003_mdot_req",       # IPA AMV Target (kg/s)
        "dau1018_armed":         "DAU1018_armed",          # DAU1018 Armed
        "dau1018_safearm_timer": "DAU1018_safearm_timer",  # DAU1018 Safe/Arm Timer
        "dau1018_seq_start":     "DAU1018_seq_start",      # DAU1018 Sequence Start
        "dau1019_armed":         "DAU1019_armed",          # DAU1019 Armed
        "dau1019_seq_start":     "DAU1019_seq_start",      # DAU1019 Sequence Start
        "dau1020_armed":         "DAU1020_armed",          # DAU1020 Armed
        "dau1020_safearm_timer": "DAU1020_safearm_timer",  # DAU1020 Safe/Arm Timer
        "dau1020_seq_start":     "DAU1020_seq_start",      # DAU1020 Sequence Start
        # tank level / density / massflow / thrust
        "dp_ipa_tank_level":     "DP730",    # DP730 IPA Tank Level (mbar(d))
        "dp_n2o_tank_level":     "DP850",    # DP850 N2O Tank Level (mbar(d))
        "rho_ipa":               "DT730",    # DT730 IPA Density
        "rho_n2o":               "DT850",    # DT850 N2O Density
        "thrust":                "LC600",    # LC600 Thrust
        "mdot_ipa":              "M730",     # M730 IPA Massflow
        "mdot_n2o":              "M850",     # M850 N2O Massflow
        # facility pressures
        "p_igniter_h2":          "PT290",    # PT290 Igniter H2
        "p_water_reg_ref":       "PT401",    # PT401 Water Regulator Ref
        "p_water_tank":          "PT402",    # PT402 Water Tank
        "p_n2_supply":           "PT520",    # PT520 N2 Supply
        "p_n2o_n2_ref":          "PT521",    # PT521 N2O N2 Ref
        "p_ipa_n2_ref":          "PT522",    # PT522 IPA N2 Ref
        "p_n2_purge":            "PT523",    # PT523 N2 Purge
        "p_impulse_air":         "PT524",    # PT524 Impulse Air
        "p_ipa_tank":            "PT730",    # PT730 IPA Tank
        "p_ipa_line":            "PT731",    # PT731 IPA Line
        "p_ipa_delivery":        "PT732",    # PT732 IPA Delivery
        "p_ipa_throttle":        "PT733",    # PT733 IPA Throttle
        "p_n2o_tank":            "PT850",    # PT850 N2O Tank
        "p_n2o_line":            "PT851",    # PT851 N2O Line
        "p_n2o_delivery":        "PT852",    # PT852 N2O Delivery
        "p_n2o_throttle":        "PT853",    # PT853 N2O Throttle
        "p_igniter_o2":          "PT890",    # PT890 Igniter O2
        # engine pressures
        "p_chamber":             "PTX101",   # PTX101 PT-C
        "p_nozzle_m":            "PTX102",   # PTX102 PT-NOZ-M
        "p_injector_m":          "PTX103",   # PTX103 PT-INJ-M
        # temperatures
        "T_ipa_line":            "TC731",    # TC731 IPA Line
        "T_ipa_delivery":        "TC732",    # TC732 IPA Delivery
        "T_n2o_tank":            "TC850",    # TC850 N2O Tank (all NaN this run)
        "T_n2o_line":            "TC851",    # TC851 N2O Line
        "T_n2o_delivery":        "TC852",    # TC852 N2O Delivery
        "T_igniter":             "TC890",    # TC890 Igniter
        "T_nozzle_m":            "TCX101",   # TCX101 TC-NOZ-M
        "T_injector_m":          "TCX102",   # TCX102 TC-INJ-M
        # valves
        "v_igniter_h2_run":      "V291",     # V291 Igniter H2 Run
        "v_hp_water_ref_up":     "V401",     # V401 HP Water Reference Up
        "v_hp_water_ref_down":   "V402",     # V402 HP Water Reference Down
        "v_hp_water_air_iso":    "V403",     # V403 HP Water Air Isolator
        "v_hp_water_tank_vent":  "V404",     # V404 HP Water Tank Vent
        "v_hp_water_run":        "V406",     # V406 HP Water Run Valve
        "v_n2o_n2_ref_up":       "V520",     # V520 N2O N2 Ref Up
        "v_n2o_n2_ref_down":     "V521",     # V521 N2O N2 Ref Down
        "v_n2o_n2_pressurant":   "V522",     # V522 N2O N2 Pressurant
        "v_ipa_n2_ref_up":       "V525",     # V525 IPA N2 Ref Up
        "v_ipa_n2_ref_down":     "V526",     # V526 IPA N2 Ref Down
        "v_ipa_n2_pressurant":   "V527",     # V527 IPA N2 Pressurant
        "v_n2_purge":            "V531",     # V531 N2 Purge
        "v_chronos":             "V690",     # V690 Chronos
        "v_komodo":              "V691",     # V691 Komodo
        "v_ipa_run":             "V730",     # V730 IPA Run
        "v_ipa_throttle":        "V731",     # V731 IPA Throttle (turns)
        "v_ipa_vent":            "V734",     # V734 IPA Vent
        "v_n2o_run":             "V851",     # V851 N2O Run
        "v_n2o_throttle":        "V852",     # V852 N2O Throttle (deg)
        "v_n2o_vent":            "V857",     # V857 N2O Vent
        "v_igniter_o2_run":      "V891",     # V891 Igniter O2 Run
        "v_glow_plug":           "V893",     # V893 Glow Plug
        # throttle feedback
        "x_ipa_throttle_fb":     "XT731",    # XT731 IPA Throttle FB (turns)
        "x_n2o_throttle_fb":     "XT852",    # XT852 N2O Throttle FB (deg)
    },
    gauge_to_absolute=(
        "p_igniter_h2",
        "p_water_reg_ref",
        "p_water_tank",
        "p_n2_supply",
        "p_n2o_n2_ref",
        "p_ipa_n2_ref",
        "p_n2_purge",
        "p_impulse_air",
        "p_ipa_tank",
        "p_ipa_line",
        "p_ipa_delivery",
        "p_ipa_throttle",
        "p_n2o_tank",
        "p_n2o_line",
        "p_n2o_delivery",
        "p_n2o_throttle",
        "p_igniter_o2",
        "p_chamber",
        "p_nozzle_m",
        "p_injector_m"),
    notes="Integza hotfire 4, nozzle ext (J2). pressures recorded bar(g), converted to absolute on load")
