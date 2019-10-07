###############################################################################
# This file is part of SWIFT.
# Copyright (c) 2019 Josh Borrow (joshua.borrow@durham.ac.uk)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################
"""
Plots the solution of the square test in a smoothed way using SWIFTsimIO's 
smoothed plotting.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.colors import Normalize

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid

snap = 40

simulations = {
    "anarchy-du-lr": "$64^2$",
    "anarchy-du": "$128^2$",
    "anarchy-du-hr": "$256^2$",
    "anarchy-du-hrhr": "$512^2$",
}

sims = [load(f"{s}/square_{snap:04d}.hdf5") for s in simulations.keys()]
resolution = 512

# First create a grid that gets the particle density so we can divide it out later
density = [project_gas_pixel_grid(sim, 512, "densities") / project_gas_pixel_grid(sim, 512, None) for sim in sims]

# Set up plotting stuff
try:
    plt.style.use("mnras_durham")
except:
    rcParams = {
        "font.serif": ["STIX", "Times New Roman", "Times"],
        "font.family": ["serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
    }
    plt.rcParams.update(rcParams)


# Now we can do the plotting.
fig, ax = plt.subplots(1, 4, figsize=(6.974, 6.974 * 1.2 / 4))
ax = ax.flatten()

for axis, image, resolution in zip(ax, density, simulations.values()):
    axis.imshow(
        1.0 - Normalize(vmin=1.5, vmax=3.5)(image),
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap="RdBu",
        vmin=0, vmax=1.0
    )

    # Exact solution, a square!
    axis.plot(
        [0.25, 0.75, 0.75, 0.25, 0.25],
        [0.25, 0.25, 0.75, 0.75, 0.25],
        linestyle="dashed",
        color="white",
        alpha=0.5,
    )

    axis.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
    )

    axis.set_title(resolution)


fig.tight_layout()
fig.savefig("SquareTestConvergence.pdf", dpi=300)
