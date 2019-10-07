"""
Makes an image of snapshot 175.
"""

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid, scatter

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

data = load("kelvinHelmholtz_0175.hdf5")

resolution = 1024
n_streamline = 512

grid = project_gas_pixel_grid(data, resolution)

sub_mask_velocity = 1024 * 2
x, y, _ = data.gas.coordinates[:].value.T
u, v, _ = data.gas.velocities[:].value.T
h = data.gas.smoothing_lengths.value

u_grid = scatter(x, y, u, h, resolution * 2).T
v_grid = scatter(x, y, v, h, resolution * 2).T
x = np.linspace(0, 1, resolution * 2)
y = np.linspace(0, 1, resolution * 2)

speed = np.sqrt(u_grid * u_grid + v_grid * v_grid)

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")

# ax.quiver(
#        x[::sub_mask_velocity],
#        y[::sub_mask_velocity],
#        u[::sub_mask_velocity],
#        v[::sub_mask_velocity],
#        color="white",
#        alpha=0.6,
#        width=0.001
# )

lw = 0.25 * speed / speed.max()
ax.streamplot(
    x,
    y,
    u_grid,
    v_grid,
    density=10,
    arrowstyle="-",
    linewidth=lw,
    color="white",

)

ax.imshow(grid.T, cmap="RdBu", extent=[0, 1, 0, 1], origin="lower")

fig.savefig("quiver.png")
