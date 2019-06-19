"""
Makes an image of snapshot 175.
"""

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid

from matplotlib.pyplot import imsave
from matplotlib.colors import Normalize

data = load("kelvinHelmholtz_0175.hdf5")

grid = project_gas_pixel_grid(data, 1024)

imsave("kelvin_helmholtz.png", Normalize(1, 2)(grid), cmap="Spectral")
