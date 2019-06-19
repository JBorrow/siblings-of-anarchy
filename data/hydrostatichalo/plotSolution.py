"""
Plots two schemes against each other for the hydrostatic halo. Shows
the energy, pressure, and density profiles, along with residuals.
"""

from swiftsimio import load
from matplotlib.gridspec import GridSpec
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

from unyt import Mpc, km, s, G

from scipy import stats

plt.style.use("mnras_durham")

gamma = 5.0 / 3.0
hubble_constant = 67.777 * (km / s) / Mpc

schemes = {"anarchy-du": "ANARCHY-DU", "anarchy-pu": "ANARCHY-PU"}

plot = {
    "internal_energy": "Internal Energy ($u$)",
    "density": "Density ($\\rho$)",
    "pressure": "Pressure ($P$)",
}

ylim = {
    "internal_energy": [28000, 35000],
    "density": [1e-4, 1e-1],
    "pressure": [(gamma - 1) * 28000 * 1e-4, (gamma - 1) * 35000 * 1e-1],
}

ylog = {"internal_energy": False, "density": True, "pressure": True}

snapshot_name = "Hydrostatic_0300.hdf5"

radial_range = (50, 1500)
radial_bins = np.linspace(*radial_range, 25)
radial_bin_centers = [0.5 * (x + y) for x, y in zip(radial_bins[:-1], radial_bins[1:])]


@lru_cache(32)
def get_r(data):
    dx = data.gas.coordinates - data.metadata.boxsize * 0.5
    r2 = np.sum(dx * dx, axis=1)
    r = np.sqrt(r2)

    return r


def plot_internal_energy(data, axis, axis_residual):
    r = get_r(data)
    u = data.gas.internal_energy

    binned = (
        stats.binned_statistic(r, u, statistic="mean", bins=radial_bins)[0] * u.units
    )
    binned_square = (
        stats.binned_statistic(r, u * u, statistic="mean", bins=radial_bins)[0]
        * u.units
        * u.units
    )
    sigma = np.sqrt(binned_square - binned * binned)

    # Plot all particles to form a nice background
    axis.scatter(r, u, s=0.2, alpha=0.5, edgecolor="none", color="C1", rasterized=True)

    # Plot the binned solution
    axis.errorbar(
        radial_bin_centers,
        binned,
        yerr=sigma,
        fmt=".",
        ms=3.0,
        color="C3",
        lw=0.5,
        zorder=2,
    )

    # Now we can plot the 'exact' solution
    circular_velocity = (
        float(data.metadata.parameters["IsothermalPotential:vrot"])
        * data.units.length
        / data.units.time
    )

    ideal_solution = (circular_velocity ** 2 / (2.0 * (gamma - 1.0))).to(u.units)

    axis.axhline(ideal_solution, linestyle="dashed", color="C0", linewidth=1)

    # Now we can compare against the ideal solution and plot the residual

    residual_binned = (binned - ideal_solution) / ideal_solution
    residual_sigma = sigma / ideal_solution

    # Plot the binned solutionA
    axis_residual.errorbar(
        radial_bin_centers,
        residual_binned,
        yerr=residual_sigma,
        fmt=".",
        ms=3.0,
        color="C3",
        lw=0.5,
        zorder=2,
    )

    axis_residual.axhline(0.0, linestyle="dashed", color="C0", linewidth=1)

    return


def plot_density(data, axis, axis_residual):
    r = get_r(data)
    rho = data.gas.density

    binned = (
        stats.binned_statistic(r, rho, statistic="mean", bins=radial_bins)[0]
        * rho.units
    )
    binned_square = (
        stats.binned_statistic(r, rho * rho, statistic="mean", bins=radial_bins)[0]
        * rho.units
        * rho.units
    )
    sigma = np.sqrt(binned_square - binned * binned)

    # Plot all particles to form a nice background
    axis.scatter(
        r, rho, s=0.2, alpha=0.5, edgecolor="none", color="C1", rasterized=True
    )

    # Plot the binned solution
    axis.errorbar(
        radial_bin_centers,
        binned,
        yerr=sigma,
        fmt=".",
        ms=3.0,
        color="C3",
        lw=0.5,
        zorder=2,
    )

    # Now we can plot the 'exact' solution. Here we just show that it goes
    # like 1 / r^2

    ideal_solution_r = np.linspace(*radial_range, 1000)
    ideal_solution_rho = (
        binned[7] * (radial_bin_centers[7] ** 2) / (ideal_solution_r ** 2)
    )

    axis.plot(
        ideal_solution_r,
        ideal_solution_rho,
        linestyle="dashed",
        color="C0",
        linewidth=1,
    )

    # Now we can compare against the ideal solution and plot the residual

    ideal_solution = (
        [(radial_bin_centers[7] ** 2) / (x ** 2) for x in radial_bin_centers]
    ) * binned[7]

    residual_binned = (binned - ideal_solution) / ideal_solution
    residual_sigma = sigma / ideal_solution

    # Plot the binned solutionA
    axis_residual.errorbar(
        radial_bin_centers,
        residual_binned,
        yerr=residual_sigma,
        fmt=".",
        ms=3.0,
        color="C3",
        lw=0.5,
        zorder=2,
    )

    axis_residual.axhline(0.0, linestyle="dashed", color="C0", linewidth=1)

    return


def plot_pressure(data, axis, axis_residual):
    r = get_r(data)
    pressure = data.gas.pressure

    binned = (
        stats.binned_statistic(r, pressure, statistic="mean", bins=radial_bins)[0]
        * pressure.units
    )
    binned_square = (
        stats.binned_statistic(
            r, pressure * pressure, statistic="mean", bins=radial_bins
        )[0]
        * pressure.units
        * pressure.units
    )
    sigma = np.sqrt(binned_square - binned * binned)

    # Plot all particles to form a nice background
    axis.scatter(
        r, pressure, s=0.2, alpha=0.5, edgecolor="none", color="C1", rasterized=True
    )

    # Plot the binned solution
    axis.errorbar(
        radial_bin_centers,
        binned,
        yerr=sigma,
        fmt=".",
        ms=3.0,
        color="C3",
        lw=0.5,
        zorder=2,
    )

    # Now we can plot the 'exact' solution. Here we just show that it goes
    # like 1 / r^2
    # Get the one for _u_
    circular_velocity = (
        float(data.metadata.parameters["IsothermalPotential:vrot"])
        * data.units.length
        / data.units.time
    )

    ideal_solution_u = (circular_velocity ** 2 / (2.0 * (gamma - 1.0))).to(
        data.gas.internal_energy.units
    )

    # Now let's do rho; for this we need the binned rho from above...

    rho = data.gas.density

    binned_rho = (
        stats.binned_statistic(r, rho, statistic="mean", bins=radial_bins)[0]
        * rho.units
    )

    ideal_solution_r = np.linspace(*radial_range, 1000)
    ideal_solution_rho = (
        binned_rho[7] * (radial_bin_centers[7] ** 2) / (ideal_solution_r ** 2)
    )

    ideal_solution_P = ideal_solution_rho * ideal_solution_u * (gamma - 1)

    axis.plot(
        ideal_solution_r, ideal_solution_P, linestyle="dashed", color="C0", linewidth=1
    )

    # Now we can compare against the ideal solution and plot the residual

    ideal_solution = (
        ([(radial_bin_centers[7] ** 2) / (x ** 2) for x in radial_bin_centers])
        * binned_rho[7]
        * ideal_solution_u
        * (gamma - 1)
    )

    residual_binned = (binned - ideal_solution) / ideal_solution
    residual_sigma = sigma / ideal_solution

    # Plot the binned solutionA
    axis_residual.errorbar(
        radial_bin_centers,
        residual_binned,
        yerr=residual_sigma,
        fmt=".",
        ms=3.0,
        color="C3",
        lw=0.5,
        zorder=2,
    )

    axis_residual.axhline(0.0, linestyle="dashed", color="C0", linewidth=1)

    return


if __name__ == "__main__":
    """ Actually create the plots! """

    fig = plt.figure(figsize=(6.974, 6.974 * (8 / 9)))

    gs = GridSpec(8, 3, figure=fig)

    # Who even likes this, gridspec is horrible (it's a shame it's so useful)
    ax_first = [plt.subplot(gs[:3, x]) for x in range(3)]
    ax_residual_first = [plt.subplot(gs[3, x]) for x in range(3)]
    ax_second = [plt.subplot(gs[4:-1, x]) for x in range(3)]
    ax_residual_second = [plt.subplot(gs[-1, x]) for x in range(3)]

    ax = [ax_first, ax_second]
    ax_residual = [ax_residual_first, ax_residual_second]

    for axes, residual_axes, (scheme, name) in zip(ax, ax_residual, schemes.items()):
        try:
            data = load(f"{scheme}/{snapshot_name}")
        except (FileNotFoundError, OSError):
            continue

        for (
            axis,
            residual_axis,
            (part_property, part_property_name),
            vertical_plot_limit,
            log_vertical_axis,
        ) in zip(axes, residual_axes, plot.items(), ylim.values(), ylog.values()):
            try:
                analysis_function = locals()[f"plot_{part_property}"]
            except:
                continue

            analysis_function(data, axis, residual_axis)

            axis.set_ylim(*vertical_plot_limit)

            if log_vertical_axis:
                axis.semilogy()

    # Don't know how to do this apart from by hand in gridspec
    for axis in ax + ax_residual[:-1]:
        for axis in axis:
            axis.set_xticklabels([])

    for axis in ax + ax_residual:
        for axis in axis:
            axis.set_xlim(*radial_range)

    # Now we need to set all of the labels
    for axis, name in zip(ax, schemes.values()):
        axis[0].set_ylabel(name)

    for axis, name in zip(ax[0], plot.values()):
        axis.set_title(name)

    ax_residual[-1][1].set_xlabel("Radius $r$ [kpc]")

    fig.tight_layout()
    fig.savefig("hydrostatic_halo.pdf")
