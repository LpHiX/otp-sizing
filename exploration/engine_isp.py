from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
import matplotlib.pyplot as plt

OFs = np.linspace(0.5, 7, 100)  # Oxidizer-to-fuel ratios

propellent_combinations = [
    ("LOX", "CH4"),
    ("LOX", "Isopropanol"),
    ("LOX", "Kerosene"),
]

fig, axs = plt.subplots(2, 1, figsize=(8, 10))
isp_plot = axs[0]
isp_plot.set_title("Specific Impulse vs O/F Ratio")
isp_plot.set_xlabel("O/F Ratio")
isp_plot.set_ylabel("Specific Impulse (s)")
isp_plot.grid()
tc_plot = axs[1]
tc_plot.set_title("Chamber Temperature vs O/F Ratio")
tc_plot.set_xlabel("O/F Ratio")
tc_plot.set_ylabel("Chamber Temperature (K)")
tc_plot.grid()


for ox, fuel in propellent_combinations:
    cea = CEA_Obj(
        oxName = ox,
        fuelName = fuel,
        isp_units='sec',
        cstar_units = 'm/s',
        pressure_units='Bar',
        temperature_units='K',
        sonic_velocity_units='m/s',
        enthalpy_units='J/g',
        density_units='kg/m^3',
        specific_heat_units='J/kg-K',
        viscosity_units='centipoise', # stored value in pa-s
        thermal_cond_units='W/cm-degC', # stored value in W/m-K
        # fac_CR=self.cr,
        make_debug_prints=False)

    isp_values = np.zeros_like(OFs)
    chamber_temps = np.zeros_like(OFs)
    for i, OF in enumerate(OFs):
        isp_values[i] = cea.get_Isp(Pc=15, MR=OF, eps=5)
        chamber_temps[i] = cea.get_Tcomb(Pc=15, MR=OF)

    
    isp_plot.plot(OFs, isp_values, label=f"{ox}/{fuel}")
    isp_plot.legend()

    tc_plot.plot(OFs, chamber_temps, label=f"{ox}/{fuel}")
    tc_plot.legend()

plt.tight_layout()
plt.savefig("isp_vs_of_ratio.png")