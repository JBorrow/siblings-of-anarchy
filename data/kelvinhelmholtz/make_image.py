"""
Makes an image of snapshot 175.
"""

from swiftsimio import load
from swiftsimio.visualisation.projection import project_gas_pixel_grid

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

data = load("kelvinHelmholtz_0175.hdf5")
resolution = 2048

grid = project_gas_pixel_grid(data, resolution=resolution, parallel=True).T

fig, ax = plt.subplots(figsize=(1, 1), dpi=resolution)
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")
ax.imshow(grid, norm=Normalize(1, 2), cmap="RdBu", origin="lower")
fig.savefig("kelvin_helmholtz.png")

