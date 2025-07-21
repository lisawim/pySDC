from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.utils import compute_solution
from pySDC.projects.DAE import my_setup_mpl
from pySDC.projects.DAE.misc.hooksDAE import (
    LogGlobalErrorPreIterDifferentialVariable,
    LogGlobalErrorPreIterationAlgebraicVariable,
    LogGlobalErrorPostIterDiff,
    LogGlobalErrorPostIterAlg,
)


def choose_time_step_sizes(problem_name):
    """Returns time step sizes suitable for each problem."""
    if problem_name == "ANDREWS-SQUEEZER":
        n_steps_list = [30, 60, 100, 300, 600, 1000, 3000, 6000]
        Tend = 0.03
    elif problem_name == "LINEAR-TEST":
        n_steps_list = [2, 5, 10, 20, 50, 100, 200, 500]
        Tend = 1.0
    else:
        raise NotImplementedError

    dt_list = [Tend / n_steps for n_steps in n_steps_list]
    return dt_list, Tend

def compute_constants_reference_order(dt_list, err_y_iter, err_z_iter, k):
    """Computes constants to shift reference order lines in plot."""

    dt_ref  = dt_list[0]
    err_y_ref, err_z_ref = err_y_iter[0], err_z_iter[0]

    Cy, Cz = err_y_ref / dt_ref ** (k + 1), err_z_ref / dt_ref ** (k + 1)
    return Cy, Cz

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

def run_and_plot_order(problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"):
    figsize = figsize_by_journal(journal, scale=0.71, ratio=0.6)

    colors = ["yellow", "gold", "orange", "red", "pink", "mediumpurple"]
    markers = ["o", "^", "h", "s", "d", "H", "*", "v", "D"]
    linestyles = ["solid", "dotted"]

    sweeper_type = "constrainedDAE"
    QI_list = ["IE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]
    num_nodes = 3
    maxiter = 2 * num_nodes - 1
    e_tol = -1

    kwargs = {"e_tol": e_tol, "maxiter": maxiter}

    t0 = 0.0
    dt_list, Tend = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[3 : 7]

    hook_class = [
        LogGlobalErrorPreIterDifferentialVariable,
        LogGlobalErrorPreIterationAlgebraicVariable,
        LogGlobalErrorPostIterDiff,
        LogGlobalErrorPostIterAlg,
    ]

    my_setup_mpl(fontsize=8)

    for q, QI in enumerate(QI_list):
        errors_y, errors_z = [], []

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        for dt in dt_list:
            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                t0 + dt,
                num_nodes,
                QI,
                sweeper_type,
                False,
                hook_class=hook_class,
                measure=False,
                **kwargs,
            )

            err_diff_values_spread = [me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_pre_iteration", sortby="iter")][0]
            err_alg_values_spread = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_pre_iteration", sortby="iter")][0]

            err_diff_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_post_iteration", sortby="iter")]
            err_alg_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]

            err_diff_values.insert(0, err_diff_values_spread)
            err_alg_values.insert(0, err_alg_values_spread)

            errors_y.append(err_diff_values)
            errors_z.append(err_alg_values)

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

            Cy, Cz = compute_constants_reference_order(dt_list, err_y_iter, err_z_iter, k)

            # Reference order
            axs[0].loglog(
                dt_list_short,
                [Cy * dt ** (k + 1) for dt in dt_list_short],
                color="black",
                linewidth=1.0,
                linestyle="dashed",
            )

            axs[1].loglog(
                dt_list_short,
                [Cz * dt ** (k + 1) for dt in dt_list_short],
                color="black",
                linewidth=1.0,
                linestyle="dashed",
            )

        for ax in axs:
            ax.tick_params(axis="both", which="minor", bottom=False, left=False)

            ax.set_xlabel(r"$\Delta t$")

        axs[0].set_ylabel("local truncation error in y")
        axs[1].set_ylabel("local truncation error in z")

        axs = sync_ylim(axs, min_y_set=1e-15)

        handles, labels = axs[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

        if QI == "MIN-SR-NS":
            if problem_name == "ANDREWS-SQUEEZER":
                plot_name = "Fig6.png"
            elif problem_name == "LINEAR-TEST":
                plot_name = "Fig3.eps"
        else:
            plot_name = f"order_iteration_{num_nodes=}_{sweeper_type}_{QI}.png"

        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    # run_and_plot_order("LINEAR-TEST")
    run_and_plot_order("ANDREWS-SQUEEZER")
