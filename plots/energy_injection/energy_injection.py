"""
This function looks at energy injection into a medium in a very simple way.

We set up a 32^3 particle cubic grid on a BCC lattice. We select the central
particle. Then, we (using pressure entropy) try to inject some amount of energy
into it.
"""

import numpy as np
from math import sqrt

from numba import jit
from scipy.optimize import newton

from typing import Union


kernel_gamma = 1.936_492
kernel_constant = 21 / (2 * 3.14159)
gas_gamma = 5.0 / 3.0
one_over_gamma = 1.0 / gas_gamma
eta = 1.2


def generate_cube(num_on_side, side_length=1.0):
    """
    Generates a cube of particles
    """

    values = np.linspace(0.0, side_length, num_on_side + 1)[:-1]

    positions = np.empty((num_on_side ** 3, 3), dtype=float)

    for x in range(num_on_side):
        for y in range(num_on_side):
            for z in range(num_on_side):
                index = x * num_on_side + y * num_on_side ** 2 + z

                positions[index, 0] = values[x]
                positions[index, 1] = values[y]
                positions[index, 2] = values[z]

    return positions


def generate_two_cube(num_on_side, side_length=1.0):
    """
    Generates two cubes of particles overlaid to make a BCC lattice.
    """

    cube = generate_cube(num_on_side // 2, side_length)

    mips = side_length / num_on_side

    positions = np.concatenate([cube, cube + mips * 0.5])

    return positions


@jit(nopython=True, fastmath=True)
def kernel(r: Union[float, np.float32], H: Union[float, np.float32]):
    """
    Kernel implementation for swiftsimio. This is the Wendland-C2
    kernel as shown in Denhen & Aly (2012).
    Give it a radius and a kernel width (i.e. not a smoothing length, but the
    radius of compact support) and it returns the contribution to the
    density.
    """
    inverse_H = 1.0 / H
    ratio = r * inverse_H

    kernel = 0.0

    if ratio < 1.0:
        ratio_2 = ratio * ratio
        ratio_3 = ratio_2 * ratio

        if ratio < 0.5:
            kernel += 3.0 * ratio_3 - 3.0 * ratio_2 + 0.5

        else:
            kernel += -1.0 * ratio_3 + 3.0 * ratio_2 - 3.0 * ratio + 1.0

        kernel *= kernel_constant * inverse_H * inverse_H * inverse_H

    return kernel


@jit(nopython=True, fastmath=True)
def calculate_pressure(
    position: np.float64,
    smoothing_length: np.float64,
    positions: np.float64,
    entropies: np.float64,
) -> np.float64:
    """
    Calculates the smoothed pressure at {position} for a 
    particle with {smoothing_length}. {positions} are the positions
    of all particles in the box and {entropies} are the entropies
    of those particles in the same order.
    """
    entropy_to_one_over_gamma = entropies ** (one_over_gamma)
    kernel_width = smoothing_length * kernel_gamma

    P = 0.0

    # This hack is required to numba me.
    for i in range(len(entropy_to_one_over_gamma)):
        pos = positions[i]
        A = entropy_to_one_over_gamma[i]

        dx = pos - position
        square = dx * dx
        r = sqrt(square[0] + square[1] + square[2])

        if r <= kernel_width:
            P += A * kernel(r, kernel_width)

    return P ** gas_gamma


def calculate_number_density(position, smoothing_length, positions):
    kernel_width = smoothing_length * kernel_gamma

    N = 0.0

    for pos in positions:
        dx = pos - position
        r = sqrt(np.sum(dx * dx))

        if r <= kernel_width:
            N += kernel(r, kernel_width)

    return N


def calculate_u(P, A):
    return A ** (one_over_gamma) * P ** (1.0 - one_over_gamma) / ((gas_gamma - 1.0))


def calculate_A(P, u):
    return (P ** (1.0 - gas_gamma)) * (gas_gamma - 1.0) ** gas_gamma * u ** gas_gamma


def find_h(position, positions):
    """
    Find the smoothing length.
    """

    # Start off with a first guess
    initial_guess = 3.0 / positions.size ** (1 / 3)

    def to_root(h):
        h_from_number_density = eta * (
            1.0 / (calculate_number_density(position, h, positions))
        ) ** (1 / 3)

        return h - h_from_number_density

    return newton(to_root, initial_guess)


def subcycle(positions, entropies, index, target_energy_addition, smoothing_length):
    """
    This does the heavy lifting. This sub-cycles, keeping track of the particle entropy and
    energy, to find a set of particle entropies that mean that u and P are consistent.
    """

    kernel_width = smoothing_length * kernel_gamma

    # First, let's find all particles we're going to interact with.
    # We need to calculate their initial internal energies as defined
    # by their smoothed pressure. Then, when we inject ENTROPY into one
    # particle, we can see how much this has changed the energy of the entire
    # system.
    position = positions[index]

    dx = positions - position
    r = np.sqrt(np.sum(dx * dx, axis=1))
    interact = r <= kernel_width

    interact_positions = positions[interact]

    # A couple of quick things to calculate the pressures of the relevant particles
    # (i.e. those that we interact with) and their energies
    def get_pressures():
        return np.array(
            [
                calculate_pressure(x, smoothing_length, positions, entropies)
                for x in interact_positions
            ]
        )

    def get_total_energy():
        pressures = get_pressures()
        interact_entropies = entropies[interact]

        return sum([calculate_u(p, A) for p, A in zip(pressures, interact_entropies)])

    initial_energy = get_total_energy()
    target_energy = initial_energy + target_energy_addition

    # Set up tracking variables, so we can plot convergence
    entropy_history = [entropies[index]]
    energy_history = [initial_energy]
    energy_injected_this_step = [0.0]
    energy_diff = [0.0]

    cycles = 0

    # Parameters taken from EAGLE
    tol = 1e-6
    max_cycles = 10

    while (
        1.0 - tol > energy_diff[-1] / target_energy_addition
        or 1.0 + tol < energy_diff[-1] / target_energy_addition
    ) and cycles < max_cycles:
        # First figure out what we want to inject in _this_ step, this is
        # always the difference between the total energy of the system and our
        # target energy as this forms a basic iteration scheme
        new_injection = target_energy - energy_history[-1]

        central_particle_pressure = calculate_pressure(
            position, smoothing_length, positions, entropies
        )
        central_particle_energy = calculate_u(
            central_particle_pressure, entropies[index]
        )

        central_particle_energy_new = central_particle_energy + new_injection

        # Based on our new energy, calculate the entropy and set it as the
        # particle's real entropy
        central_particle_entropy_new = calculate_A(
            central_particle_pressure, central_particle_energy_new
        )

        entropies[index] = central_particle_entropy_new

        # See what effect we _really_ had on the system.
        new_total_energy = get_total_energy()

        entropy_history.append(central_particle_entropy_new)
        energy_history.append(new_total_energy)
        energy_diff.append(new_total_energy - initial_energy)
        energy_injected_this_step.append(new_injection)

        cycles += 1

    return entropy_history, energy_history, energy_diff, energy_injected_this_step


if __name__ == "__main__":
    # First, generate positions:

    positions = generate_two_cube(32)

    # Now find the one closest to the centre:
    arg_our_favourite = np.argmin(np.sum((positions - 0.5) ** 2, axis=1))
    position = positions[arg_our_favourite]

    print(f"Central particle at {position}.")

    # Now have a gander at its smoothing length
    smoothing_length = find_h(position, positions)

    print(f"It has h={smoothing_length:e}, MIPS={1 / 32:e}")

    # Calculate number of particles here
    volume = (4.0 * np.pi / 3.0) * (smoothing_length * kernel_gamma) ** 3
    number_density = calculate_number_density(position, smoothing_length, positions)

    print(f"This gives it N={number_density * volume}")

    # Now generate entropies
    entropies = np.ones(positions.shape[0], dtype=float)

    pressure = calculate_pressure(position, smoothing_length, positions, entropies)

    print(f"The particle has initial pressure P={pressure:e}")
    print(f"We would expect this to be P={number_density**gas_gamma:e}")
    print(f"The particle has energy u={calculate_u(pressure, 1.0):e}")

    # Now the fun begins.
    print(f"Let's try to multiply its energy by 100!")

    target_injection = calculate_u(pressure, 1.0) * 99

    entropy_history, energy_history, energy_diff, energy_injected = subcycle(
        positions, entropies, arg_our_favourite, target_injection, smoothing_length
    )

    print(f"Attempting to inject {target_injection:e} energy")

    # print("Entropy as a function of time")
    # print(entropy_history)
    # print("Energy as a function of time")
    # print(energy_history)
    # print("Diff from target as a function of time")
    # print(energy_diff)
    # print("Ratio to expected")
    # print([x / target_injection for x in energy_diff])
    # print("Injected this step")
    # print(energy_injected)

    print("Creating plots")

    import matplotlib.pyplot as plt

    plt.style.use("mnras_durham")

    fig, ax = plt.subplots(2, 2, figsize=(6.974, 6.974), sharex=True)
    ax = ax.flatten()

    to_plot = {
        # Scale the values for plotting otherwise it messes up y-axis
        "Entropy of particle injected into": np.array(entropy_history) * 0.01,
        "Total energy of system": np.array(energy_history) * 0.0001,
        "Energy actually injected": np.array(energy_diff) * 0.0001,
        "Ratio to required injection": [x / target_injection for x in energy_diff],
    }

    hlines_at = [
        None,
        (energy_history[0] + target_injection) * 0.0001,
        target_injection * 0.0001,
        1.0,
    ]

    for axis, hline, (name, stat) in zip(ax, hlines_at, to_plot.items()):
        axis.plot(range(len(stat)), stat)
        axis.set_ylabel(name)

        if hline is not None:
            axis.axhline(hline, linestyle="dashed", linewidth=1.0, color="C3")

    for axis in ax[-2:]:
        axis.set_xlabel("Iterations")

    fig.tight_layout()

    fig.savefig("energy_injection.pdf")

    print("Saved plot to energy_injection.pdf")

