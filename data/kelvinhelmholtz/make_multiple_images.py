"""
Makes an image of snapshot 175.
"""

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.style.use("mnras_durham")

schemes = {
    "minimal": "Density-Energy",
    "pressure-energy": "Pressure-Energy",
    "anarchy-du-original": "ANARCHY-DU",
    "anarchy-pu": "ANARCHY-PU",
}

fig, ax = plt.subplots(2, 2, figsize=(6.97, 6.97))
fig.subplots_adjust(0, 0, 1, 1, 0, 0)
ax = ax.flatten()

for a, (folder, name) in zip(ax, schemes.items()):
    data = load(f"{folder}/kelvinHelmholtz_0175.hdf5")
    grid = project_gas_pixel_grid(data, 1024)

    print(grid.max(), grid.min())

    a.imshow(grid, cmap="Spectral", origin="lower", vmin=2.29, vmax=20.92)

    a.set_xticks([])
    a.set_yticks([])

    a.text(
        0.5,
        0.95,
        name,
        ha="center",
        va="top",
        transform=a.transAxes,
        bbox=dict(boxstyle="round", edgecolor="none", facecolor=(1.0, 1.0, 1.0, 0.6)),
    )

fig.savefig("kelvin_helmholtz.pdf")
