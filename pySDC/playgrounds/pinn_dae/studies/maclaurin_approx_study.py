import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.playgrounds.pinn_dae.utils import my_setup_mpl
from pySDC.playgrounds.pinn_dae.approximation.maclaurin_series import MacLaurinApproximation


def y_exact(t, lambda_d=-2.0):
    t = np.asarray(t)
    return np.exp(2 * lambda_d * t)


def z_exact(t, lambda_d=-2.0, lambda_a=1.0):
    t = np.asarray(t)
    return lambda_d / lambda_a * np.exp(2 * lambda_d * t)


def main():
    problem_name = "LINEAR-TEST"

    t0 = 0.0
    y0 = y_exact(t=t0)
    z0 = z_exact(t=t0)

    degree_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    approx_y = MacLaurinApproximation(u0=y0, is_diff=True)
    approx_z = MacLaurinApproximation(u0=z0, is_diff=False)

    t_end = 1.0
    t_eval = np.linspace(t0, t_end, num=100)
    y_ex_eval = y_exact(t_eval)
    z_ex_eval = z_exact(t_eval)

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for degree in degree_list:
        y_approx_eval = approx_y.eval(t_eval, degree)
        axs[0].plot(t_eval, y_approx_eval, label=rf"$p = ${degree}")

        z_approx_eval = approx_z.eval(t_eval, degree)
        axs[1].plot(t_eval, z_approx_eval, label=rf"$p = ${degree}")

    axs[0].plot(t_eval, y_ex_eval, color="black", linestyle="dotted", label="exact")
    axs[1].plot(t_eval, z_ex_eval, color="black", linestyle="dotted", label="exact")

    for ax in axs:
        ax.set_xlabel(r"time $t$")
        ax.set_xlim(t0, t_end)

    axs[0].set_ylabel(r"solution y(t)")
    axs[1].set_ylabel(r"solution z(t)")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=5)

    plot_name = f"maclaurin_approx_{t_end=}"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + ".png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=500, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
