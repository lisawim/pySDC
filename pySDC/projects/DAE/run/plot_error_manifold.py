from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.utils import compute_solution
from pySDC.projects.DAE.run.plot_order_iteration import sync_ylim
from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config

from pySDC.projects.DAE.misc.hooksDAE import (
    LogGlobalErrorDiffVar,
    LogGlobalErrorAlgVar,
)

from pySDC.projects.DAE.misc.hooksDAE import LogAbsValuePostIterAlgebraicConstraints


def plot_manifold_val_and_error_vs_iteration(
    dt, num_nodes, problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"
):
    figsize = figsize_by_journal(journal, scale=1.0, ratio=0.8)

    sweeper_types = ["constrainedDAE", "semiImplicitDAE", "fullyImplicitDAE"]
    QI_list = ["IE", "LU", "MIN-SR-S"]

    t0 = 0.0
    maxiter = 10

    hook_class = [LogGlobalErrorDiffVar, LogGlobalErrorAlgVar, LogAbsValuePostIterAlgebraicConstraints]

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(3, 3, figsize=figsize)
    ax_flatten = axs.flatten()

    for q, QI in enumerate(QI_list):
        axs[0, q].set_title(f"{QI}"), axs[1, q].set_title(f"{QI}"), axs[2, q].set_title(f"{QI}")

        for sweeper_type in sweeper_types:
            print(f"Running for {sweeper_type} with {QI}..")
            key = f"{sweeper_type}_LU"

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
                e_tol=-1,
                maxiter=maxiter,
            )

            x = [me[0] for me in get_sorted(solution_stats, type="abs_g_post_iteration", sortby="iter")]
            g_abs_values = [me[1] for me in get_sorted(solution_stats, type="abs_g_post_iteration", sortby="iter")]
            err_diff_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_post_iteration", sortby="iter")
            ]
            err_alg_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")
            ]

            axs[0, q].semilogy(x, err_diff_values, color=colors[key], label=sweeper_labels[sweeper_type])
            axs[1, q].semilogy(x, err_alg_values, color=colors[key], label=sweeper_labels[sweeper_type])
            axs[2, q].semilogy(x, g_abs_values, color=colors[key], label=sweeper_labels[sweeper_type])

        axs[0, q].set_ylabel(r"global error in $y$")
        axs[1, q].set_ylabel(r"global error in $z$")
        axs[2, q].set_ylabel(r"$|g(y,z)|$")

    for i in range(len(ax_flatten)):
        ax_flatten[i].set_xlabel(r"iteration")
        ax_flatten[i].set_xticks([k for k in range(1, maxiter + 1)])

    ax_flatten = sync_ylim(ax_flatten, min_y_set=1e-15)

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3)

    plot_name = f"abs_g_err_y_z_vs_iteration_{num_nodes=}_{dt=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_manifold_value_vs_iteration(
    dt, num_nodes, problem_name="REACTION-DIFFUSION", journal="Springer_Scientific_Computing"
):
    figsize = figsize_by_journal(journal, scale=0.45, ratio=0.6)

    sweeper_types = ["constrainedDAE", "semiImplicitDAE"]
    QI_list = ["IE", "LU", "MIN-SR-S"]

    t0 = 0.0
    maxiter = 20

    hook_class = [LogAbsValuePostIterAlgebraicConstraints]

    my_setup_mpl(fontsize=5)
    plt.rcParams['axes.linewidth'] = 0.45
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for q, QI in enumerate(QI_list):
        for sweeper_type in sweeper_types:
            print(f"Running for {sweeper_type} with {QI}..")
            key = f"{sweeper_type}_{QI}"

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
                e_tol=-1,
                maxiter=maxiter,
            )

            x = [me[0] for me in get_sorted(solution_stats, type="abs_g_post_iteration", sortby="iter")]
            g_abs_values = [me[1] for me in get_sorted(solution_stats, type="abs_g_post_iteration", sortby="iter")]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            ax.semilogy(
                x,
                g_abs_values,
                color=colors[key],
                marker=markers[key],
                linewidth=1.0,
                markersize=2.7,
                markeredgewidth=0.5,
                label=label,
            )

    ax.set_ylabel(r"$||g(y^k_{M,t_1}, z^k_{M,t_1})||_\infty$")

    ax.set_xlabel(r"iteration $k$")
    ax.set_xticks([k for k in range(1, maxiter + 1, 2)])
    ax.set_xticklabels([k for k in range(1, maxiter + 1, 2)])

    ax.set_xlim((0.8, maxiter + 0.2))

    ax.tick_params(axis="both", which="major", length=2.5, width=0.45)
    ax.tick_params(axis="both", which="minor", bottom=True, left=False, length=1.5, width=0.45)

    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.5)
    ax.grid(axis="both", which="minor", linewidth=0.2, alpha=0.15)

    handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.07), ncol=3)

    plot_name = "Fig13.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes

    problem_name = "REACTION-DIFFUSION"
    dt_list, _ = choose_time_step_sizes(problem_name=problem_name)
    dt = dt_list[0]
    print(dt)
    num_nodes = 6
    plot_manifold_value_vs_iteration(dt=dt, num_nodes=num_nodes, problem_name=problem_name)
