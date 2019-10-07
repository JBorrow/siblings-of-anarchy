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

import cmocean.cm as cm

snap = 40

simulations = {
    "minimal": "Density-Energy",
    "anarchy-du": "ANARCHY-DU",
    "pressure-energy": "Pressure-Energy",
    "anarchy-pu": "ANARCHY-PU",
}

sims = [load(f"{s}/square_{snap:04d}.hdf5") for s in simulations.keys()]
resolution = 512

# First create a grid that gets the particle density so we can divide it out later
unweighted_grids = [project_gas_pixel_grid(sim, 512, None) for sim in sims]

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
fig, ax = plt.subplots(4, 3, figsize=(6.974, 6.974 * 4 / 3 ))

# These are stored in priority order
plot = dict(
    internal_energy="Internal Energy ($u$)",
    density=r"Density ($\rho$)",
    pressure="Pressure ($P$)",
)

minmax = dict(internal_energy=[1.5, 3.5], density=[3.5, 1.5], pressure=[2.5 + 0.2, 2.5 - 0.2])

for sim, unweighted_grid, axis_row, name in zip(
    sims, unweighted_grids, ax, simulations.values()
):
    current_axis = 0

    axis_row[current_axis].set_ylabel(name, fontsize=10)

    for key, label in plot.items():
        axis = axis_row[current_axis]
        vmin, vmax = minmax[key]

        try:
            grid = (
                project_gas_pixel_grid(sim, resolution=resolution, project=key)
                / unweighted_grid
            )
        except:
            grid = (
                project_gas_pixel_grid(sim, resolution=resolution, project=key.replace("y", "ies").replace("ure", "ures"))
                / unweighted_grid
            )

        if vmin > vmax:
            normed_grid = 1.0 - Normalize(vmin=vmax, vmax=vmin)(grid)
        else:
            normed_grid = Normalize(vmin=vmin, vmax=vmax)(grid)

        axis.imshow(
            normed_grid,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap="RdBu",
            vmin=0.0,
            vmax=1.0,
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

        current_axis += 1

for axis, name in zip(ax[0], plot.values()):
    axis.set_title(name, fontsize=10)


fig.subplots_adjust(0.03, 0.01, 0.99, 0.97, hspace=0, wspace=0)
fig.savefig("SquareTest.pdf", dpi=300)
