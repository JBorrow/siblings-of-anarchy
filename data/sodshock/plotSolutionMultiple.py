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

from swiftsimio import load
from swiftsimio.visualisation import project_gas_pixel_grid

from analyticSolution import analytic

snap = 1

simulations = {
    "minimal": "Density-Energy",
    "anarchy-du-old": "ANARCHY-DU",
    "pressure-energy": "Pressure-Energy",
    "anarchy-pu": "ANARCHY-PU",
}

sims = [load(f"{s}/sodshock_{snap:04d}.hdf5") for s in simulations.keys()]
resolution = 512

data = [
    dict(
        x=sim.gas.coordinates.value[:, 0],
        v=sim.gas.velocities.value[:, 0],
        u=sim.gas.internal_energy.value,
        S=sim.gas.entropy.value,
        P=sim.gas.pressure.value,
        rho=sim.gas.density.value,
        y=sim.gas.coordinates.value[:, 1],
        z=sim.gas.coordinates.value[:, 2],
    )
    for sim in sims
]

ref = analytic(sims[0].metadata.time.value)

# We only want to plot this for the region that we actually have data for, hence the masking.
masks = [
    np.logical_and(ref["x"] < np.max(d["x"]), ref["x"] > np.min(d["x"])) for d in data
]
refs = [{k: v[mask] for k, v in ref.items()} for mask in masks]


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
fig, ax = plt.subplots(4, 4, figsize=(6.974, 6.974), sharey="col")

# These are stored in priority order
plot = dict(
    u="Internal Energy ($u$)",
    rho=r"Density ($\rho$)",
    P="Pressure ($P$)",
    v="Velocity ($v_x$)",
)

ylim = dict(v=(-0.05, 1.0), diff=(0.0, None), visc=(0.0, None))

for sim, axis_row, name, d, mask, ref in zip(
    sims, ax, simulations.values(), data, masks, refs
):
    current_axis = 0

    axis_row[current_axis].set_ylabel(name, fontsize=10)

    for key, label in plot.items():
        if current_axis > 3:
            break
        else:
            axis = axis_row[current_axis]

        axis.plot(
            d["x"],
            d[key],
            ".",
            color="C1",
            markersize=0.5,
            alpha=0.5,
            rasterized=True,
            markeredgecolor="none",
            zorder=-1,
        )

        mask_noraster = np.logical_and.reduce(
            [d["y"] < 0.52, d["y"] > 0.48, d["z"] < 0.52, d["z"] > 0.48]
        )

        axis.plot(
            d["x"][mask_noraster],
            d[key][mask_noraster],
            ".",
            color="C3",
            rasterized=False,
            markeredgecolor="none",
            markersize=3,
            zorder=0,
        )

        # Exact solution
        try:
            axis.plot(ref["x"], ref[key], c="C0", ls="dashed", zorder=1, lw=1)
        except KeyError:
            # No solution :(
            pass

        axis.set_xlim(0.6, 1.5)

        try:
            axis.set_ylim(*ylim[key])
        except KeyError:
            # No worries pal
            pass

        current_axis += 1

for axis, name in zip(ax[0], plot.values()):
    axis.set_title(name, fontsize=10)

for axis_row in ax[:-1]:
    for axis in axis_row:
        axis.set_xticklabels([])

fig.tight_layout()


# Plot the zoomed region; must do this after tight layout in a separate loop!
for sim, axis_row, name, d, mask, ref in zip(
    sims, ax, simulations.values(), data, masks, refs
):
    current_axis = 0

    axis_row[current_axis].set_ylabel(name, fontsize=10)

    for key, label in plot.items():
        if current_axis > 3:
            break
        else:
            axis = axis_row[current_axis]

        if key == "P":
            axins = axis.inset_axes([0.5, 0.5, 0.47, 0.47])

            axins.plot(
                d["x"],
                d[key],
                ".",
                color="C1",
                markersize=0.5,
                alpha=0.5,
                rasterized=True,
                markeredgecolor="none",
                zorder=-1,
            )

            mask_noraster = np.logical_and.reduce(
                [d["y"] < 0.52, d["y"] > 0.48, d["z"] < 0.52, d["z"] > 0.48]
            )

            axins.plot(
                d["x"][mask_noraster],
                d[key][mask_noraster],
                ".",
                color="C3",
                rasterized=False,
                markeredgecolor="none",
                markersize=3,
                zorder=0,
            )

            # Exact solution
            try:
                axins.plot(ref["x"], ref[key], c="C0", ls="dashed", zorder=1, lw=1)
            except KeyError:
                # No solution :(
                pass

            axins.set_xlim(1.1, 1.25)
            axins.set_ylim(0.2, 0.4)

            for func in ["xticks", "yticks", "yticklabels", "xticklabels"]:
                getattr(axins, f"set_{func}")([])

            axis.indicate_inset_zoom(axins)

        current_axis += 1


fig.savefig("SodShock.pdf", dpi=300)
