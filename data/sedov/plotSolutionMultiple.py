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

snap = 5

simulations = {
    "minimal": "Density-Energy",
    "anarchy-du": "ANARCHY-DU",
    "anarchy-du-switch": "ANARCHY-DU (Switch)",
    "pressure-energy": "Pressure-Energy",
    "anarchy-du-nodiff": "ANARCHY-DU (No diffusion)",
    "anarchy-pu-nodiff": "ANARCHY-PU (No diffusion)",
}

sims = [load(f"{s}/sedov_{snap:04d}.hdf5") for s in simulations.keys()]
resolution = 512

data = [
    dict(
        x=np.sqrt(
            np.sum(
                (sim.gas.coordinates - sim.metadata.boxsize * 0.5).value ** 2, axis=1
            )
        ),
        v=sim.gas.velocities.value[:, 0],
        u=sim.gas.internal_energy.value,
        S=sim.gas.entropy.value,
        P=sim.gas.pressure.value,
        rho=sim.gas.density.value,
    )
    for sim in sims
]

ref = analytic(sims[0].metadata.time.value)
r_shock = ref.pop("r_shock")

# We only want to plot this for the region that we actually have data for, hence the masking.
masks = [
    np.logical_and(ref["x"] < np.max(d["x"]), ref["x"] > np.min(d["x"])) for d in data
]
refs = [{k: v[mask] for k, v in ref.items()} for mask in masks]

# Bin the data
x_bin_edge = np.linspace(0.15, 1.3 * r_shock, 25)
x_bin = 0.5 * (x_bin_edge[1:] + x_bin_edge[:-1])

binned = [
    {
        k: stats.binned_statistic(d["x"], v, statistic="mean", bins=x_bin_edge)[0]
        for k, v in d.items()
    }
    for d in data
]
square_binned = [
    {
        k: stats.binned_statistic(d["x"], v ** 2, statistic="mean", bins=x_bin_edge)[0]
        for k, v in d.items()
    }
    for d in data
]
sigma = [
    {k: np.sqrt(v2 - v ** 2) for k, v2, v in zip(b.keys(), sq.values(), b.values())}
    for b, sq in zip(binned, square_binned)
]


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
fig, ax = plt.subplots(2, 3, figsize=(6.974, 6.974 * 2 / 3), sharey=True, sharex=True)
ax = ax.flatten()

# These are stored in priority order
plot = dict(P="Pressure ($P$)")

ylim = dict(
    v=(-0.05, 1.0), diff=(0.0, None), visc=(0.0, None), rho=(-0.05, 4.5), P=(-0.5, 6.5)
)

for sim, axis, name, d, mask, ref, binned_data, sigma_binned in zip(
    sims, ax, simulations.values(), data, masks, refs, binned, sigma
):

    axis.set_title(name, fontsize=10)

    for key, label in plot.items():
        # Raw data
        axis.plot(
            d["x"],
            d[key],
            ".",
            color="C1",
            ms=0.5,
            alpha=0.7,
            markeredgecolor="none",
            rasterized=True,
            zorder=0,
        )
        # Binned data
        axis.errorbar(
            x_bin,
            binned_data[key],
            yerr=sigma_binned[key],
            fmt=".",
            ms=3.0,
            color="C3",
            lw=0.5,
            zorder=2,
        )
        # Exact solution
        try:
            axis.plot(ref["x"], ref[key], c="C0", ls="dashed", zorder=1, lw=1)
        except KeyError:
            # No solution :(
            pass

        axis.set_xlim(0.15, 1.3 * r_shock)

        try:
            axis.set_ylim(*ylim[key])
        except KeyError:
            # No worries pal
            pass

ax[0].set_ylabel(list(plot.values())[0])
ax[3].set_ylabel(list(plot.values())[0])
ax[4].set_xlabel("Radius ($r$)")


fig.subplots_adjust(0.03, 0.01, 0.99, 0.97, hspace=0, wspace=0)
fig.tight_layout()
fig.savefig("SedovBlast.pdf", dpi=300)
