import os

import numpy as np

from otp.pump.records import PumpGeometry, PumpRequirements, PumpChoices
from otp.pump.analysis import Pump
from otp.experiments.loaders.airborne import orifice_flow, difference, ChannelMap, csv_channel
from otp.experiments.records import Run

from pathlib import Path
DATA_PATH_212 = Path(__file__).parent / '20261001-212.h5'
DATA_PATH_206 = Path(__file__).parent / '20261001-206.h5'
DATA_PATH_209 = Path(__file__).parent / '20261001-209.h5'

PUMP_GEOM = PumpGeometry(d_inlet=0.0298 / 1.2,
                    d_1=0.0298, 
                    d_2=0.0689,
                    b_1=0.0074,
                    b_2=0.0032,
                    d_throat=0.0062, 
                    d_outlet=0.0115,
                    d_3=0.0, 
                    b_3=0.0, 
                    s_ax=0.0)            # unused
PUMP = Pump(PumpRequirements(None, 1.2e-3, 760.0), PumpChoices(24e3, 6, 1, coeff_p = 0.2), PUMP_GEOM)

SS_RPM = 18750

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
        "m_ipa_tank":            "DP730",    # DP730 IPA Tank Level (kg)
        "m_n2o_tank":            "DP850",    # DP850 N2O Tank Level (kg)
        "rho_ipa":               "DT730",    # DT730 IPA Density
        "rho_n2o":               "DT850",    # DT850 N2O Density
        "thrust":                "LC600",    # LC600 Thrust
        "mdot_ipa":              "M730",     # M730 IPA Massflow
        "mdot_n2o":              "M850",     # M850 N2O Massflow
        "mdot_gg_n2o":           "M851",     # M851 GG N2O Massflow (g/s)
        # pump
        "p_pump_in":             "PT-PMP-IN",    # PT-PMP-IN Pump Inlet
        "p_pump_out":            "PT-PMP-OUT",   # PT-PMP-OUT Pump Outlet
        "p_pump_brg_cavity":     "PT-PMP-BRC",   # PT-PMP-BRC Bearing Cavity
        "p_pump_brg_return":     "PT-PMP-BRR",   # PT-PMP-BRR Bearing Return (logger name says BRC)
        "p_dump_orifice":        "PT-DMP-ORF",   # PT-DMP-ORF Dump Orifice
        "T_pump_in":             "TC-PMP-IN",    # TC-PMP-IN Pump Inlet
        "T_pump_brg_return":     "TC-PMP-BRR",   # TC-PMP-BRR Bearing Return
        # gas generator
        "p_gg_chamber":          "PT-GG-CC",     # PT-GG-CC Chamber
        "p_gg_fuel_del":         "PT-GG-FD",     # PT-GG-FD Fuel Del
        "p_gg_ox_del":           "PT-GG-OD",     # PT-GG-OD Ox Del
        "p_gg_ipa_tank":         "PT-GG-TNK",    # PT-GG-TNK GG IPA Tank
        "T_gg_aft":              "TC-GG-AFT",    # TC-GG-AFT Chamber Aft
        "T_gg_fore":             "TC-GG-FORE",   # TC-GG-FORE Chamber Fore
        "T_gg_fuel_del":         "TC-GG-FD",     # TC-GG-FD Fuel Del
        "T_gg_ox_del":           "TC-GG-OD",     # TC-GG-OD Ox Del
        "T_gg_igniter":          "TC-GG-IGN",    # TC-GG-IGN Igniter
        # thrust chamber
        "p_tca_chamber":         "PT-TCA-CC",    # PT-TCA-CC Chamber
        "p_tca_fuel_nozzle":     "PT-TCA-FM1",   # PT-TCA-FM1 Fuel Nozzle
        "p_tca_fuel_inj":        "PT-TCA-FM2",   # PT-TCA-FM2 Fuel Inj
        "T_tca_fuel_nozzle":     "TC-TCA-FM1",   # TC-TCA-FM1 Fuel Nozzle
        "T_tca_fuel_inj":        "TC-TCA-FM2",   # TC-TCA-FM2 Fuel Inj
        "T_tca_ox_del":          "TC-TCA-OD",    # TC-TCA-OD Ox Del
        "T_tca_igniter":         "TC-TCA-IGN",   # TC-TCA-IGN Igniter
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
        # facility temperatures
        "T_ipa_line":            "TC731",    # TC731 IPA Line
        "T_ipa_delivery":        "TC732",    # TC732 IPA Delivery
        "T_n2o_tank":            "TC850",    # TC850 N2O Tank
        "T_n2o_line":            "TC851",    # TC851 N2O Line
        "T_n2o_delivery":        "TC852",    # TC852 N2O Delivery
        "T_igniter":             "TC890",    # TC890 Igniter
        # rig valves
        "v_gg_ipa":              "V-GG-F",       # V-GG-F GG IPA
        "v_gg_ipa_run":          "V-GG-FR",      # V-GG-FR GG IPA Run
        "v_gg_ign_fuel":         "V-GG-IGN-F",   # V-GG-IGN-F GG Ign Fuel
        "v_gg_ign_ox":           "V-GG-IGN-O",   # V-GG-IGN-O GG Ign Ox
        "v_gg_n2o":              "V-GG-O",       # V-GG-O GG N2O
        "v_gg_n2o_run":          "V-GG-OR",      # V-GG-OR GG N2O Run
        "v_pump_in":             "V-PMP-IN",     # V-PMP-IN Pump In
        "v_pump_out":            "V-PMP-OUT",    # V-PMP-OUT Pump Out
        "v_tca_ign_fuel":        "V-TCA-IGN-F",  # V-TCA-IGN-F TCA Ign Fuel
        "v_tca_ign_ox":          "V-TCA-IGN-O",  # V-TCA-IGN-O TCA Ign Ox
        # facility valves
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
        "v_ipa_n2_purge":        "V531",     # V531 IPA N2 Purge
        "v_n2o_n2_purge":        "V536",     # V536 N2O N2 Purge
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
        "p_pump_in",
        "p_pump_out",
        "p_pump_brg_cavity",
        "p_pump_brg_return",
        "p_dump_orifice",
        "p_gg_chamber",
        "p_gg_fuel_del",
        "p_gg_ox_del",
        "p_gg_ipa_tank",
        "p_tca_chamber",
        "p_tca_fuel_nozzle",
        "p_tca_fuel_inj",
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
        "p_igniter_o2"),
    notes="OTP pump spin test (J2). pressures recorded bar(g), converted to absolute on load")
