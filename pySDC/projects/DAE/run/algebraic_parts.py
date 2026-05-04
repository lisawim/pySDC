import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE import compute_solution, my_setup_mpl, my_plot_style_config
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run.study_embedding_linear import get_hooks, get_ylabel


def plot_absolute_value_g_vs_iterations_dae_solvers(dt, num_nodes, problem_name, QI="MIN-SR-NS", journal="BUW_thesis", ax=None, return_ax=False):
    created_fig = ax is None

    sweeper_types = ["constrainedDAE", "fullyImplicitDAE", "semiImplicitDAE"]

    colors, markers, sweeper_labels = my_plot_style_config()
    linestyles = ["solid", "dashdot", "dashed"]
    if created_fig:
        figsize = figsize_by_journal(journal=journal, scale=0.7, ratio=0.83)
        my_setup_mpl(fontsize=5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    maxiter = 20
    for s, sweeper_type in enumerate(sweeper_types):
        key = f"{sweeper_type}_LU"

        solution_stats = compute_solution(
            problem_name,
            t0=0.0,
            dt=dt,
            Tend=dt,
            num_nodes=num_nodes,
            QI=QI,
            sweeper_type=sweeper_type,
            hook_class=get_hooks(stat="abs_g", along="iterations", eps=0.0),
            measure=False,
            maxiter=maxiter,
            e_tol=-1,
        )

        x = [me[0] for me in get_sorted(solution_stats, type=f"g_abs_post_iteration", sortby="iter")]
        g_abs_values = [me[1] for me in get_sorted(solution_stats, type=f"g_abs_post_iteration", sortby="iter")]

        label = sweeper_labels[sweeper_type]
        ax.plot(x, g_abs_values, color=colors[key], linestyle=linestyles[s], label=label)

    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"$||g(y^k, z^k)||_\infty$")

    ax.set_xlim((1, maxiter))

    ax.set_yscale("log", base=10)
    ax.set_ylim((1e-16, 1e3))

    ax.grid(linewidth=0.5)

    if created_fig:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=2)

        filename = "data" + "/" + f"{problem_name}" + "/" + f"absolute_value_g_vs_iterations_dae_solvers_{num_nodes=}_{dt=}.png"
        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

        if not return_ax:
            plt.close(fig)
            return None
        
    return ax if return_ax else None


def plot_error_z_vs_iterations_dae_solvers(dt, num_nodes, problem_name, QI="MIN-SR-NS", journal="BUW_thesis", ax=None, return_ax=False):
    created_fig = ax is None

    sweeper_types = ["constrainedDAE", "fullyImplicitDAE", "semiImplicitDAE"]

    colors, markers, sweeper_labels = my_plot_style_config()
    linestyles = ["solid", "dashdot", "dashed"]
    if created_fig:
        figsize = figsize_by_journal(journal=journal, scale=0.7, ratio=0.83)
        my_setup_mpl(fontsize=5)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    for s, sweeper_type in enumerate(sweeper_types):
        key = f"{sweeper_type}_LU"

        solution_stats = compute_solution(
            problem_name,
            t0=0.0,
            dt=dt,
            Tend=dt,
            num_nodes=num_nodes,
            QI=QI,
            sweeper_type=sweeper_type,
            hook_class=get_hooks(stat="diff_alg_error", along="iterations", eps=0.0),
            measure=False,
            e_tol=-1,
        )

        x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]
        alg_error_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]

        label = sweeper_labels[sweeper_type]
        ax.plot(x, alg_error_values, color=colors[key], linestyle=linestyles[s], label=label)

    ax.set_xlabel(r"iteration $k$")
    ax.set_ylabel(r"$||z(t_0 + \Delta t) - z^k_M||$")

    ax.set_yscale("log", base=10)
    ax.set_ylim((1e-16, 1e3))

    ax.grid(linewidth=0.5)

    if created_fig:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=2)

        filename = "data" + "/" + f"{problem_name}" + "/" + f"error_z_vs_iterations_dae_solvers_{num_nodes=}_{dt=}.png"
        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

        if not return_ax:
            plt.close(fig)
            return None
        
    return ax if return_ax else None


def absolute_values_g_thesis(dt, num_nodes, problem_name, journal="BUW_thesis", return_ax=False):
    figsize = figsize_by_journal(journal=journal, scale=0.7, ratio=0.83)

    my_setup_mpl(fontsize=5)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    QI_list = ["IE", "LU", "MIN-SR-NS", "MIN-SR-S"]
    for q, QI in enumerate(QI_list):
        ax_flatten[q].set_title(rf"$Q_\Delta=${QI}")

        plot_absolute_value_g_vs_iterations_dae_solvers(
            dt=dt,
            num_nodes=num_nodes,
            problem_name=problem_name,
            QI=QI,
            journal=journal,
            ax=ax_flatten[q],
            return_ax=return_ax,
        )

    handles, labels = ax_flatten[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"absolute_values_g_{num_nodes=}_{dt=}.png"
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_differential_algebraic_error_vs_iterations_qi(dt, num_nodes, problem_name, sweeper_type, journal="BUW_thesis", axs=None, return_axs=False):
    created_fig = axs is None

    QI_list = ["IE", "LU", "MIN-SR-NS", "MIN-SR-S"]

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    colors, _, sweeper_labels = my_plot_style_config()
    if created_fig:
        figsize = figsize_by_journal(journal=journal, scale=0.9, ratio=0.4)
        my_setup_mpl(fontsize=6)
        fig, axs = plt.subplots(1, 2, figsize=figsize)
    else:
        axs = np.atleast_1d(axs)
        fig = axs[0].figure

    for QI in QI_list:
        key = f"{sweeper_type}_{QI}"
        print(get_hooks("diff_alg_error", along="iterations", eps=0.0))
        solution_stats = compute_solution(
            problem_name,
            t0,
            dt,
            t0+dt,
            num_nodes,
            QI,
            sweeper_type,
            hook_class=get_hooks("diff_alg_error", along="iterations", eps=0.0),
            measure=False,
            e_tol=-1,
            maxiter=20,
        )

        x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_differential_post_iteration", sortby="iter")]
        diff_err_values_per_iter = [me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_post_iteration", sortby="iter")]
        alg_err_values_per_iter = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]

        label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
        axs[0].plot(x, diff_err_values_per_iter, label=label, color=colors[key])
        axs[1].plot(x, alg_err_values_per_iter, color=colors[key])

    ylabels = get_ylabel("diff_alg_error", along="iterations")
    axs[0].set_ylabel(ylabels[0])
    axs[1].set_ylabel(ylabels[1])

    for ax in axs:
        ax.set_xlabel(r"iteration $k$")

        ax.set_yscale("log", base=10)
        ax.set_ylim((1e-13, 1e1))

        ax.grid(linewidth=0.5)

    if created_fig:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=5)

        plot_name = f"diff_alg_error_vs_iterations_{sweeper_type}_{num_nodes=}_{dt=}.png"
        filename = "data" + "/" + f"{problem_name}" + "/" + "emb_thesis" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

        if not return_axs:
            plt.close(fig)
            return None
        
    return axs if return_axs else None


def dae_errors_thesis(dt, num_nodes, problem_name, journal="BUW_thesis", return_ax=False):
    figsize = figsize_by_journal(journal=journal, scale=0.7, ratio=0.83)

    my_setup_mpl(fontsize=5)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    QI_list = ["IE", "LU", "MIN-SR-NS", "MIN-SR-S"]
    for q, QI in enumerate(QI_list):
        ax_flatten[q].set_title(rf"$Q_\Delta=${QI}")

        plot_error_z_vs_iterations_dae_solvers(
            dt=dt,
            num_nodes=num_nodes,
            problem_name=problem_name,
            QI=QI,
            journal=journal,
            ax=ax_flatten[q],
            return_ax=return_ax,
        )

    for ax in ax_flatten:
        ax.set_xlim((1, 20))
        ax.set_ylim((1e-13, 1e6))

    handles, labels = ax_flatten[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"dae_errors_{num_nodes=}_{dt=}.png"
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # For Andrews
    problem_name = "ANDREWS-SQUEEZER"
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt = dt_list[3]

    QI = "LU"
    num_nodes = 8

    plot_absolute_value_g_vs_iterations_dae_solvers(
        dt=dt, num_nodes=num_nodes, problem_name=problem_name, QI=QI, journal="BUW_thesis"
    )

    plot_error_z_vs_iterations_dae_solvers(
        dt=dt, num_nodes=num_nodes, problem_name=problem_name, QI=QI, journal="BUW_thesis"
    )

    absolute_values_g_thesis(dt=dt, num_nodes=num_nodes, problem_name=problem_name, journal="BUW_thesis")

    dae_errors_thesis(dt=dt, num_nodes=num_nodes, problem_name=problem_name, journal="BUW_thesis")