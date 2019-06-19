""" Makes an image of snapshot 175.
"""

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid

import matplotlib.pyplot as plt

data = load("kelvinHelmholtz_0320.hdf5")

empty_grid = project_gas_pixel_grid(data, 7 * 300, None)
grid = project_gas_pixel_grid(data, 7 * 300, "internal_energy") / empty_grid

fig, a = plt.subplots(figsize=(7,7))
fig.subplots_adjust(0, 0, 1, 1, 0, 0)

a.imshow(grid, cmap="Spectral", origin="lower")

a.set_xticks([])
a.set_yticks([])

fig.savefig("kelvin_helmholtz_highres_energy.png", dpi=300)
