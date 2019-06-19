""" Makes an image of snapshot 175.
"""

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid

import matplotlib.pyplot as plt

data = load("kelvinHelmholtz_0164.hdf5")

grid = project_gas_pixel_grid(data, 7 * 300)

fig, a = plt.subplots(figsize=(7,7))
fig.subplots_adjust(0, 0, 1, 1, 0, 0)

a.imshow(grid, cmap="Spectral", origin="lower", vmin=2.29, vmax=20.92)

a.set_xticks([])
a.set_yticks([])

fig.savefig("kelvin_helmholtz_highres.pdf")
