"""
Makes an image of snapshot 175.
"""

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid

from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from p_tqdm import p_map

import matplotlib.pyplot as plt

data = [load("kelvinHelmholtz_{:04d}.hdf5".format(x)) for x in range(452)]

def baked(snap):
    grid = project_gas_pixel_grid(data[snap], 1024)

    return grid

grid = p_map(baked, list(range(452)))

fig, ax = plt.subplots(figsize=(1,1), dpi=1024)
fig.subplots_adjust(0, 0, 1, 1)
ax.axis("off")

image = ax.imshow(grid[0], norm=Normalize(2, 20), cmap="Spectral", origin="lower")

def anim(x):
    image.set_array(grid[x])

    return image,

animation = FuncAnimation(
    fig,
    anim,
    list(range(452)),
    interval=40,
)

animation.save("kh_movie.mp4")

