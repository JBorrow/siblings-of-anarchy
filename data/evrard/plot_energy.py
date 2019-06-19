"""
Plots energy(t) for several components for the various simulations.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

plt.style.use("mnras_durham")

simulations = {
    "minimal": "Density-Energy",
    "pressure-energy": "Pressure-Energy",
    "anarchy-du": "ANARCHY-DU",
    "anarchy-pu": "ANARCHY-PU",
}

highlight_time = 0.8

lines_type = []
handles_type = list(simulations.values())
lines_energy = []
handles_energy = ["Total", "Kinetic $E_K$", "Thermal $E_U$", "Potential $E_P$"]

fig, ax = plt.subplots()

ax.axvline(highlight_time, linestyle="dashed", linewidth=1, color="grey")

for color, (handle, name) in enumerate(simulations.items()):
    energy = np.genfromtxt(f"{handle}/energy.txt").T

    time = energy[0]
    total_energy = energy[2]

    kinetic_energy = energy[3]
    internal_energy = energy[4]
    potential_energy = energy[5]

    total = ax.plot(time, total_energy, label=name, color=f"C{color}")[0]

    kinetic = ax.plot(time, kinetic_energy, color=f"C{color}", linestyle="dashed")[0]
    thermal = ax.plot(time, internal_energy, color=f"C{color}", linestyle="dotted")[0]
    potential = ax.plot(time, potential_energy, color=f"C{color}", linestyle="-.")[0]

    # Build other legend
    if color == 3:
        lines_energy = [total, kinetic, thermal, potential]

    lines_type.append(total)


ax.set_xlabel("Simulation time $t$")
ax.set_ylabel("Energy in component $E_{i}$")
legend_type = plt.legend(
    # Empty rectangle is there to create a space in the legend
    lines_type,
    handles_type,
    markerfirst=False,
    fontsize=6,
    loc="upper right",
)

legend_other = plt.legend(
    lines_energy, handles_energy, markerfirst=False, fontsize=6, loc="lower right"
)

ax.add_artist(legend_type)
ax.add_artist(legend_other)

ax.set_xlim(0, 4)
ax.set_ylim(None, None)

fig.tight_layout()

fig.savefig("EvrardEnergy.pdf")
