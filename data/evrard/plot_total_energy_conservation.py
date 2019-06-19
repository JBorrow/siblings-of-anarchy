"""
Plots energy(t) for several components for the various simulations.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

from scipy.interpolate import interp1d


def smooth(x, window_len=11, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y[(window_len // 2 - 1) : -(window_len // 2 + 1)]


plt.style.use("mnras_durham")

simulations = {
    "minimal": "Density-Energy",
    "pressure-energy": "Pressure-Energy",
    "anarchy-du": "ANARCHY-DU",
    "anarchy-pu": "ANARCHY-PU",
}

highlight_time = 0.8

ground_truth = "anarchy-du"

fig, ax = plt.subplots()

# Plot our highlighted time for reference to the grid solution
ax.axvline(highlight_time, linestyle="dashed", linewidth=1, color="grey")


for color, (handle, name) in enumerate(simulations.items()):
    energy_data = np.genfromtxt(f"{handle}/energy.txt").T

    time = energy_data[0]
    initial_energy = energy_data[2][0]
    this_energy = (energy_data[2] - initial_energy) / initial_energy

    total = ax.plot(time, smooth(this_energy), label=name, color=f"C{color}")[0]

ax.legend(markerfirst=False, fontsize=6)


ax.set_xlabel("Simulation time $t$")
ax.set_ylabel("Total energy conservation (%)")

ax.set_xlim(0, 5)

fig.tight_layout()

fig.savefig("EvrardEnergyConservation.pdf")
