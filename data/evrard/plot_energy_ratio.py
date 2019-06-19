"""
Plots energy(t) for several components for the various simulations.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

from scipy.interpolate import interp1d

plt.style.use("mnras_durham")

simulations = {
    "minimal": "Density-Energy",
    "pressure-energy": "Pressure-Energy",
    "anarchy-du": "ANARCHY-DU",
    "anarchy-pu": "ANARCHY-PU",
}

highlight_time = 0.8

ground_truth = "anarchy-du"

fig, ax = plt.subplots(4, 1, figsize=(3.321, 3.321 * 4 / 3), sharex=True)
ax = ax.flatten()

handles_energy = ["Total", "Kinetic $E_K$", "Thermal $E_U$", "Potential $E_P$"]
handles_type = list(simulations.values())
scale_function = [lambda x: (x - 1) * 100.0] * 3 + [lambda x: (1 - x) * 100]

for axis, energy, scale in zip(ax, [2, 3, 4, 5], scale_function):
    # Need to set these up for the legends
    lines_type = []

    # Plot our highlighted time for reference to the grid solution
    axis.axvline(highlight_time, linestyle="dashed", linewidth=1, color="grey")

    ground_truth_energy = np.genfromtxt(f"{ground_truth}/energy.txt").T

    ground_truth_time = ground_truth_energy[0]
    ground_truth_property = ground_truth_energy[energy]

    # We need to interpolate that so we can get things at the
    # exact same time to divide out; timesteps are not guaranteed
    # to occur at the exact same point for all schemes.

    ground_truth_interpolated = interp1d(ground_truth_time, ground_truth_property)

    for color, (handle, name) in enumerate(simulations.items()):
        energy_data = np.genfromtxt(f"{handle}/energy.txt").T

        time = energy_data[0]
        this_energy = energy_data[energy]
        this_energy_ratio = scale(this_energy / ground_truth_interpolated(time))

        total = axis.plot(time, this_energy_ratio, label=name, color=f"C{color}")[0]

        lines_type.append(total)

    if axis == ax[1]:
        axis.legend(lines_type, handles_type, markerfirst=False, fontsize=6)


ax[-1].set_xlabel("Simulation time $t$")

ax[-1].set_xlim(0, 5)

for name, axis in zip(handles_energy, ax):
    axis.set_ylabel(name)

fig.tight_layout(pad=0.1)

fig.savefig("EvrardEnergyRatio.pdf")
