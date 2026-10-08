from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.playgrounds.pinn_dae.utils import compute_solution, my_setup_mpl


def choose_time_step_sizes(problem_name):
    """Returns time step sizes suitable for each problem."""
    if problem_name == "ANDREWS-SQUEEZER":
        n_steps_list = [30, 60, 100, 300, 600, 1000, 3000, 6000]
        Tend = 0.03
    elif problem_name in ["LINEAR-TEST", "NONSTIFF-LINEAR-EX1"]:
        n_steps_list = [2, 5, 10, 20, 50, 100, 200, 500]
        Tend = 1.0
    elif problem_name == "REACTION-DIFFUSION":
        n_steps_list = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        Tend = 0.25
    else:
        raise NotImplementedError

    dt_list = [Tend / n_steps for n_steps in n_steps_list]
    return dt_list, Tend


def compute_constant_reference_order(dt_list, err_iter, k):
    """Computes constants to shift reference order lines in plot."""

    dt_ref = dt_list[0]
    err_ref = err_iter[0]

    C = err_ref / dt_ref ** (k + 1)
    return C


def get_error_values(solution_stats, variable, mode):
    sort_key = "sweep" if mode == "sweep" else "iter"
    pre_suffix = f"pre_{mode}"
    post_suffix = f"post_{mode}"

    err_pre = [
        me[1]
        for me in get_sorted(
            solution_stats,
            type=f"e_global_{variable}_{pre_suffix}",
            sortby=sort_key,
        )
    ][0]

    err_post = [
        me[1]
        for me in get_sorted(
            solution_stats,
            type=f"e_global_{variable}_{post_suffix}",
            sortby=sort_key,
        )
    ]
    return [err_pre] + err_post


def sync_xlim(axs, min_x_set=1e-15):
    """Synchronize x-axis limits across all subplots by finding the global min/max."""
    min_x, max_x = None, None

    # Find global min/max y-limits across all axes
    for ax in axs:
        x_limits = ax.get_xlim()

        # Ignore non-positive values for log scale
        if ax.get_xscale() == "log":
            x_limits = [x for x in x_limits if x > 0]  
            if not x_limits:
                continue

        if min_x is None or x_limits[0] < min_x:
            min_x = x_limits[0]
        if max_x is None or x_limits[1] > max_x:
            max_x = x_limits[1]

    # Apply the same limits to all subplots
    for ax in axs:
        if ax.get_xscale() == "log":
            if min_x is not None and min_x <= 0:
                min_x = min_x_set
            ax.set_xlim(min_x, max_x)
        else:
            ax.set_xlim(min_x, max_x)

    return axs


def sync_ylim(axs, min_y_set=1e-15):
    """Synchronize y-axis limits across all subplots by finding the global min/max."""
    min_y, max_y = None, None

    # Find global min/max y-limits across all axes
    for ax in axs:
        y_limits = ax.get_ylim()

        # Ignore non-positive values for log scale
        if ax.get_yscale() == "log":
            y_limits = [y for y in y_limits if y > 0]
            if not y_limits:
                continue

        if min_y is None or y_limits[0] < min_y:
            min_y = y_limits[0]
        if max_y is None or y_limits[1] > max_y:
            max_y = y_limits[1]

    # Apply the same limits to all subplots
    for ax in axs:
        if ax.get_yscale() == "log":
            if min_y is not None and min_y <= 0:
                min_y = min_y_set
            ax.set_ylim(min_y, max_y)
        else:
            ax.set_ylim(min_y, max_y)

    return axs


def plot_order_linear(num_nodes=3, sweeper_type="SDC-C-ML", journal="BUW_thesis"):
    """Plots the order in each iteration."""

    from pySDC.projects.DAE.misc.hooksDAE import (
        LogGlobalErrorDiffVar, LogGlobalErrorAlgVar
    )

    problem_name = "LINEAR-TEST"
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.5)

    colors = ["yellow", "gold", "orange", "red", "pink", "mediumpurple"]
    markers = ["o", "^", "h", "s", "d", "H", "*", "v", "D"]
    linestyles = ["solid", "dotted"]

    initial_guess = "ML_train"
    degree = 2

    QI_list = ["EE", "IE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]
    maxiter = 2 * num_nodes - 1 - degree  # 2 * 3 - 1 = 5 - degree = 3
    e_tol = -1

    kwargs = {"e_tol": e_tol, "initial_guess": initial_guess, "degree": degree}

    t0 = 0.0
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[3:7]

    hook_class = [LogGlobalErrorDiffVar, LogGlobalErrorAlgVar]

    my_setup_mpl(fontsize=7.5)
    plt.rcParams['patch.linewidth'] = 0.3

    offsets = [0.7, 0.45, 0.6, 0.55, 0.55, 0.5, 0.45]

    for q, QI in enumerate(QI_list):
        print(f"Running for {QI}..")
        errors_y, errors_z = [], []

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        for dt in dt_list:
            solution_stats = compute_solution(
                problem_name=problem_name,
                t0=t0,
                dt=dt,
                Tend=t0 + dt,
                num_nodes=num_nodes,
                QI=QI,
                sweeper_type=sweeper_type,
                use_mpi=False,
                hook_class=hook_class,
                measure=False,
                maxiter=maxiter,
                **kwargs,
            )

            errors_y.append(get_error_values(solution_stats, "differential", "iteration"))
            errors_z.append(get_error_values(solution_stats, "algebraic", "iteration"))

        for k in range(maxiter + 1):
            err_y_iter = [res[k] for res in errors_y]
            err_z_iter = [res[k] for res in errors_z]

            axs[0].loglog(
                dt_list,
                err_y_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
                label=f"k = {k}",
            )

            axs[1].loglog(
                dt_list,
                err_z_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
            )

            Cy = compute_constant_reference_order(dt_list, err_y_iter, k)
            Cz = compute_constant_reference_order(dt_list, err_z_iter, k)

            # Reference order
            p = degree + k + 1
            ref_y = [Cy * dt ** p for dt in dt_list_short]
            axs[0].loglog(
                dt_list_short,
                ref_y,
                color="black",
                linewidth=1.0,
                linestyle="dashed",
            )

            axs[0].text(
                dt_list_short[-1] * 0.8,
                ref_y[-1] * offsets[k],
                rf"${p}$",
                fontsize=7,
                va="center",
                ha="left",
                color="black",
            )

            ref_z = [Cz * dt ** p for dt in dt_list_short]
            axs[1].loglog(
                dt_list_short,
                ref_z,
                color="black",
                linewidth=1.0,
                linestyle="dashed",
            )

            axs[1].text(
                dt_list_short[-1] * 0.8,
                ref_z[-1] * offsets[k],
                rf"${p}$",
                fontsize=7,
                va="center",
                ha="left",
                color="black",
            )

        for ax in axs:
            ax.tick_params(axis="both", which="minor", bottom=False, left=False)

            ax.set_xlabel(r"time step size $\Delta t$")

            ax.grid(linewidth=0.5)

        axs[0].set_ylabel(r"LTE $||y(t_1) - y^k_{M,t_1}||_{\infty}$")
        axs[1].set_ylabel(r"LTE $||z(t_1) - z^k_{M,t_1}||_{\infty}$")

        axs = sync_ylim(axs, min_y_set=1e-15)

        handles, labels = axs[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=3)

        plot_name = f"order_iteration_linear_ML_predict_{degree=}_{num_nodes=}_{sweeper_type}_{QI}"
        # plot_name = f"order_iteration_linear_{num_nodes=}_{sweeper_type}_{QI}"
        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + ".png"
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=500, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    num_nodes = 3
    plot_order_linear(num_nodes=num_nodes)
    # plot_order_andrews(num_nodes=num_nodes)
    # plot_order_reaction_diffusion(num_nodes=num_nodes)
