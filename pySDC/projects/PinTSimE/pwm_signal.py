import numpy as np
import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl
from pySDC.helpers.plot_helper import figsize_by_journal


def v_pwm(t, V_i, fsw, duty):
    Tsw = 1 / fsw
    if 0 <= ((t / Tsw) % 1) <= duty:
        return V_i
    else:
        return 0


def plot_pwm(filename="pwm_signal", journal="BUW_thesis"):
    V_i = 100
    duty_cycles = [0.25, 0.75]
    switching_frequencies = [1e0, 1e1]

    t_eval = np.linspace(0, 1.0, num=200)

    my_setup_mpl(fontsize=10)

    figsize = figsize_by_journal(journal, scale=0.8, ratio=0.5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    linestyle = ["solid", "dashed"]
    color = ["blue", "dodgerblue", "forestgreen", "limegreen"]

    i = 0
    for d, duty in enumerate(duty_cycles):
        for f, fsw in enumerate(switching_frequencies):
            v_eval = [v_pwm(t, V_i, fsw, duty) for t in t_eval]

            axs[d].plot(t_eval, v_eval, linestyle=linestyle[f], color=color[i], label=rf"$d=${duty}, $f_s=${fsw}")

            i += 1

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel("voltage")
        ax.grid(linewidth=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=2)

    filename = "data" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)
