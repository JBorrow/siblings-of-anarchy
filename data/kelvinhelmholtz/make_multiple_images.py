"""
Makes an image of snapshot 175.
"""

from swiftsimio import load
from swiftsimio.visualisation.projection import project_gas_pixel_grid

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.style.use("mnras_durham")

schemes = {
    "minimal": "Density-Energy",
    "pressure-energy": "Pressure-Energy",
    "anarchy-du": "ANARCHY-DU",
    "anarchy-pu": "ANARCHY-PU",
}

fig, ax = plt.subplots(2, 2, figsize=(6.97, 6.97))
fig.subplots_adjust(0, 0, 1, 1, 0, 0)
ax = ax.flatten()

for a, (folder, name) in zip(ax, schemes.items()):
    data = load(f"{folder}/kelvinHelmholtz_0175.hdf5")
    grid = project_gas_pixel_grid(data, 1024)

    a.imshow(grid.T, cmap="RdBu_r", origin="lower", vmin=1, vmax=2)

    a.set_xticks([])
    a.set_yticks([])

    a.text(
        0.5,
        0.965,
        name,
        ha="center",
        va="top",
        transform=a.transAxes,
        color="white",
#        bbox=dict(boxstyle="round", edgecolor="none", facecolor=(1.0, 1.0, 1.0, 0.6)),
    )

fig.savefig("kelvin_helmholtz.pdf")
