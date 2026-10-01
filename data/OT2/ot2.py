import os

import numpy as np

from otp.pump.records import PumpGeometry, PumpRequirements, PumpChoices
from otp.pump.analysis import Pump
from otp.experiments.loaders.airborne import orifice_flow, difference, ChannelMap, csv_channel
from otp.experiments.records import Run


WATER_RHO = 1000.0
CD = 0.8
D_OR = 0.0049
A_THROAT = np.pi / 4 * D_OR ** 2

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

from pathlib import Path
DATA_PATH = Path(__file__).parent / '20260701-111.h5'

RPM_PATH = Path(__file__).parent / 'rpm.csv'
RPM_SHIFT = -14

CHANNEL_MAP = ChannelMap(
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
    },
    gauge_to_absolute=(
        "p_pump_in",
        "p_pump_seal",
        "p_gg_chamber",
        "p_pump_out",
        "p_orifice_in",
        "p_ipa_delivery",
        "p_n2o_delivery"),
    derive={
        "dp_pump": difference(a="p_pump_out", b="p_pump_in", out_name="dp_pump", units="bar", desc="pump pressure rise"),
        "Q_pump": orifice_flow(dp_channel="p_orifice_in", Cd=CD, A_throat=A_THROAT, rho=WATER_RHO, out_name="Q_pump"),
        "rpm":    csv_channel(RPM_PATH, "rpm", units="rpm", shift=RPM_SHIFT),
    },
    notes="pressures recorded bar(g), converted to absolute on load")
