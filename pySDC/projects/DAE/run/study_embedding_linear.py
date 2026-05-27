import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE import compute_solution, my_setup_mpl, my_plot_style_config
from pySDC.helpers.stats_helper import get_sorted


def get_color_embedding(sweeper_type, e):
    if sweeper_type == "SPP":
        colors_spp = [
            'mistyrose',
            'lightsalmon',
            'lightcoral',
            'indianred',
            'firebrick',
            'brown',
            'maroon',
            'lightgray',
            'darkgray',
            'gray',
            'dimgray',
        ]
        return colors_spp[e]
    elif sweeper_type == "embeddedDAE":
        return "royalblue"
    elif sweeper_type == "constrainedDAE":
        return "gold"
    else:
        raise ValueError(f"Unknown sweeper type: {sweeper_type}")


def get_hooks(stat, along="time", eps=0.0):
    if stat == "abs_g":
        if eps == 0.0:
            from pySDC.projects.DAE.misc.hooksDAE import LogAbsValueAlgConstraints

            return [LogAbsValueAlgConstraints]
        elif eps > 0.0:
            from pySDC.implementations.problem_classes.linearTestSPP import LogGlobalErrorLinearTestSPP

            return [LogGlobalErrorLinearTestSPP]
    elif stat == "diff_alg_error":
        if eps == 0.0:
            from pySDC.projects.DAE.misc.hooksDAE import LogGlobalErrorDiffVar, LogGlobalErrorAlgVar

            return [LogGlobalErrorDiffVar, LogGlobalErrorAlgVar]
        elif eps > 0.0:
            from pySDC.implementations.problem_classes.linearTestSPP import LogGlobalErrorLinearTestSPP

            return [LogGlobalErrorLinearTestSPP]
    elif stat == "error":
        from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep, LogGlobalErrorPostIter
        if along == "iterations":
            return [LogGlobalErrorPostIter]

        return [LogGlobalErrorPostStep]

    elif stat == "iterations":
        return []
    elif stat == "increment":
        from pySDC.implementations.hooks.log_embedded_error_estimate import (
            LogEmbeddedErrorEstimate, LogEmbeddedErrorEstimatePostIter
        )
        if along == "iterations":
            return [LogEmbeddedErrorEstimatePostIter]

        return [LogEmbeddedErrorEstimate]
    else:
        raise ValueError(f"Unknown stat: {stat}")

def get_stats_type(stat, along="time"):
    if stat in ["abs_g", "g_abs"]:
        if along == "iterations":
            return "g_abs_post_iteration"
        return "g_abs_post_step"
    elif stat == "error":
        if along == "iterations":
            return "e_global_post_iteration"

        return "e_global_post_step"
    elif stat == "iterations":
        return "niter"
    elif stat == "increment":
        if along == "iterations":
            return "error_embedded_estimate_post_iteration"

        return "error_embedded_estimate"
    else:
        raise ValueError(f"Unknown stat: {stat}")
    

def get_sortby_arg(along="time"):
    return "iter" if along == "iterations" else "time"


def get_suffix(along="iterations"):
    if along == "iterations":
        return "_post_iteration"
    return "_post_step"


def get_xlabel(along="time"):
    if along == "iterations":
        return r"iteration $k$"
    return r"time $t$"
    

def get_ylabel(stat, along="time"):
    if stat in ["abs_g", "g_abs"]:
        if along == "iterations":
            return r"$|g(y^k, z^k)|$"
        return r"$|g(y, z)|$"
    elif stat == "diff_alg_error":
        if along == "iterations":
            return (r"$||y(t_1) - y^k_{M,t_1}||$", r"$||z(t_1) - z^k_{M,t_1}||$")
        return (r"$|y(t) - y^{\tilde{k}}_{M,t}|$", r"$|z(t) - z^{\tilde{k}}_{M,t}|$")
    elif stat == "error":
        if along == "iterations":
            return r"$L_\infty$ error $||u(t_1) - u^k_{M,t_1}||$"

        return r"$L_\infty$ error"
    if stat == "iterations":
        return "number of iterations"
    elif stat == "increment":
        if along == "iterations":
            return r"increment $\|u^{k+1} - u^{k}\|$"
        return r"increment $\|u^{\tilde{k}+1} - u^{\tilde{k}}\|$"
    else:
        raise ValueError(f"Unknown stat: {stat}")


def plot_stats_vs_nodes(num_nodes, QI="MIN-SR-NS", stat="iterations", journal="BUW_thesis"):
    problem_name = "LINEAR-TEST"
    
    figsize = figsize_by_journal(journal=journal, scale=0.71, ratio=0.6)

    sweeper_types = ["SPP", "embeddedDAE", "constrainedDAE"]
    QI = "IE"
    num_nodes_list = range(2, 9)

    t0 = 0.0
    dt_list, Tend = choose_time_step_sizes("LINEAR-TEST")
    dt = dt_list[3]

    my_setup_mpl(fontsize=6)
    _, markers, _ = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    type = get_stats_type(stat)

    for sweeper_type in sweeper_types:
        eps_list = [10 ** (-m) for m in range(1, 12)] if sweeper_type == "SPP" else [0.0]

        for e, eps in enumerate(eps_list):
            values_per_num_nodes = []

            for num_nodes in num_nodes_list:
                key = f"{sweeper_type}_IE"  # Enforces the same markers for different QI

                solution_stats = compute_solution(
                    problem_name,
                    t0,
                    dt,
                    Tend,
                    num_nodes,
                    QI,
                    sweeper_type,
                    measure=False,
                    hook_class=get_hooks(stat),
                    eps=eps,
                )

                values_per_time = [me[1] for me in get_sorted(solution_stats, type=type, sortby="time")]
                if stat == "iterations":
                    val = np.mean(values_per_time)
                else:
                    val = max(values_per_time)
                values_per_num_nodes.append(val)

            color = get_color_embedding(sweeper_type, e)
            marker = None if sweeper_type == "SPP" else markers[key]
            label = rf"$\varepsilon=${eps}" if sweeper_type == "SPP" else f"{sweeper_type}"
            ax.plot(
                num_nodes_list,
                values_per_num_nodes,
                label=label,
                color=color,
                marker=marker,
            )

    ax.set_xlabel(r"number of nodes $M$")
    ylabel = get_ylabel(stat)
    ax.set_ylabel(ylabel)

    if stat != "iterations":
        ax.set_yscale("log", base=10)
        ax.set_ylim((1e-16, 1e-1))

    ax.grid(linewidth=0.5)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4)

    plot_name = f"{stat}_vs_nodes_{QI}_{dt=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + "emb_thesis" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_stats_vs_time(dt, num_nodes, QI="MIN-SR-NS", stat="iterations", journal="BUW_thesis"):
    problem_name = "LINEAR-TEST"
    
    figsize = figsize_by_journal(journal=journal, scale=0.71, ratio=0.6)

    sweeper_types = ["SPP", "embeddedDAE", "constrainedDAE"]

    t0 = 0.0
    _, Tend = choose_time_step_sizes("LINEAR-TEST")

    my_setup_mpl(fontsize=6)
    _, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    type = get_stats_type(stat)

    for sweeper_type in sweeper_types:
        eps_list = [10 ** (-m) for m in range(1, 12)] if sweeper_type == "SPP" else [0.0]

        for e, eps in enumerate(eps_list):
            key = f"{sweeper_type}_IE"  # Enforces the same markers for different QI

            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                Tend,
                num_nodes,
                QI,
                sweeper_type,
                hook_class=get_hooks(stat),
                measure=False,
                eps=eps,
            )

            t = [me[0] for me in get_sorted(solution_stats, type=type, sortby="time")]
            values_per_time = [me[1] for me in get_sorted(solution_stats, type=type, sortby="time")]

            color = get_color_embedding(sweeper_type, e)
            marker = None if sweeper_type == "SPP" else markers[key]
            label = rf"$\varepsilon=${eps}" if sweeper_type == "SPP" else sweeper_labels[sweeper_type]
            ax.plot(t, values_per_time, label=label, color=color, marker=marker)

    ax.set_xlabel(r"time $t$")
    ylabel = get_ylabel(stat)
    ax.set_ylabel(ylabel)

    if stat != "iterations":
        ax.set_yscale("log", base=10)
        ax.set_ylim((1e-16, 1e1))

    ax.grid(linewidth=0.5)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4)

    plot_name = f"{stat}_vs_time_{QI}_{num_nodes=}_{dt=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + "emb_thesis" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_stats_vs_iterations(
    dt,
    num_nodes,
    QI="MIN-SR-NS",
    stat="increment",
    journal="BUW_thesis",
    ax=None,
    return_ax=False,
):
    created_fig = ax is None

    problem_name = "LINEAR-TEST"

    sweeper_types = ["SPP", "embeddedDAE", "constrainedDAE"]

    t0 = 0.0

    _, _, sweeper_labels = my_plot_style_config()
    if created_fig:
        figsize = figsize_by_journal(journal=journal, scale=0.71, ratio=0.6)
        my_setup_mpl(fontsize=6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    type = get_stats_type(stat, along="iterations")

    for sweeper_type in sweeper_types:
        eps_list = [10 ** (-m) for m in range(1, 12)] if sweeper_type == "SPP" else [0.0]

        for e, eps in enumerate(eps_list):
            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                t0+dt,
                num_nodes,
                QI,
                sweeper_type,
                hook_class=get_hooks(stat, along="iterations", eps=eps),
                measure=False,
                eps=eps,
                e_tol=-1,
                maxiter=25,
            )

            x = [me[0] for me in get_sorted(solution_stats, type=type, sortby="iter")]
            values_per_iter = [me[1] for me in get_sorted(solution_stats, type=type, sortby="iter")]

            color = get_color_embedding(sweeper_type, e)
            label = rf"$\varepsilon=${eps}" if sweeper_type == "SPP" else sweeper_labels[sweeper_type]
            ax.plot(x, values_per_iter, label=label, color=color)

    ax.set_xlabel(r"iteration $k$")
    ylabel = get_ylabel(stat, along="iterations")
    ax.set_ylabel(ylabel)

    ax.set_yscale("log", base=10)

    ax.set_ylim((1e-16, 1e1))

    ax.grid(linewidth=0.5)

    if created_fig:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=4)

        plot_name = f"{stat}_vs_iteration_{QI}_{num_nodes=}_{dt=}.png"
        filename = "data" + "/" + f"{problem_name}" + "/" + "emb_thesis" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

        if not return_ax:
            plt.close(fig)
            return None
        
    return ax if return_ax else None


def plot_differential_algebraic_errors_vs_metric(
    dt,
    along,
    num_nodes,
    QI="MIN-SR-NS",
    journal="BUW_thesis",
    axs=None,
    return_axs=False,
):
    created_fig = axs is None

    problem_name = "LINEAR-TEST"

    sweeper_types = ["SPP", "embeddedDAE", "constrainedDAE"]

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    _, _, sweeper_labels = my_plot_style_config()
    if created_fig:
        figsize = figsize_by_journal(journal=journal, scale=0.9, ratio=0.4)
        my_setup_mpl(fontsize=6)
        fig, axs = plt.subplots(1, 2, figsize=figsize)
    else:
        axs = np.atleast_1d(axs)
        fig = axs[0].figure

    prefix = "e_global"
    suffix = get_suffix(along=along)
    sortby = get_sortby_arg(along=along)

    for sweeper_type in sweeper_types:
        eps_list = [10 ** (-m) for m in range(1, 12)] if sweeper_type == "SPP" else [0.0]

        for e, eps in enumerate(eps_list):
            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                t0+dt if along == "iterations" else Tend,
                num_nodes,
                QI,
                sweeper_type,
                hook_class=get_hooks("diff_alg_error", along=along, eps=eps),
                measure=False,
                eps=eps,
                e_tol=-1,
                maxiter=25,
            )

            x = [me[0] for me in get_sorted(solution_stats, type=f"{prefix}_differential{suffix}", sortby=sortby)]
            diff_err_values_per_iter = [me[1] for me in get_sorted(solution_stats, type=f"{prefix}_differential{suffix}", sortby=sortby)]
            alg_err_values_per_iter = [me[1] for me in get_sorted(solution_stats, type=f"{prefix}_algebraic{suffix}", sortby=sortby)]

            color = get_color_embedding(sweeper_type, e)
            label = rf"$\varepsilon=${eps}" if sweeper_type == "SPP" else sweeper_labels[sweeper_type]
            axs[0].plot(x, diff_err_values_per_iter, label=label, color=color)
            axs[1].plot(x, alg_err_values_per_iter, color=color)

    ylabels = get_ylabel("diff_alg_error", along=along)
    axs[0].set_ylabel(ylabels[0])
    axs[1].set_ylabel(ylabels[1])

    for ax in axs:
        ax.set_xlabel(get_xlabel(along=along))

        ax.set_yscale("log", base=10)
        ax.set_ylim((1e-13, 1e1))

        ax.grid(linewidth=0.5)

    if created_fig:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=5)

        plot_name = f"diff_alg_error_vs_{along}_{QI}_{num_nodes=}_{dt=}.png"
        filename = "data" + "/" + f"{problem_name}" + "/" + "emb_thesis" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

        if not return_axs:
            plt.close(fig)
            return None
        
    return axs if return_axs else None


def plot_absolute_value_g_vs_metric(
    dt,
    num_nodes,
    along="iterations",
    journal="BUW_thesis",
    ax=None,
    return_ax=False,
):
    created_fig = ax is None

    problem_name = "LINEAR-TEST"

    sweeper_types = ["embeddedDAE"]
    QI_list = ["EE", "IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard"]

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    colors, markers, sweeper_labels = my_plot_style_config()
    linestyles = ["solid", "dashed"]
    if created_fig:
        figsize = figsize_by_journal(journal=journal, scale=0.45, ratio=0.7)
        my_setup_mpl(fontsize=4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    stat = "abs_g"
    suffix = get_suffix(along=along)
    sortby = get_sortby_arg(along=along)

    kwargs = {"e_tol": -1} if along == "iterations" else {}

    for q, QI in enumerate(QI_list):
        for sweeper_type in sweeper_types:
            key = f"{sweeper_type}_{QI}"

            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                t0+dt if along == "iterations" else Tend,
                num_nodes,
                QI,
                sweeper_type,
                hook_class=get_hooks(stat=stat, along=along, eps=0.0),
                measure=False,
                maxiter=25,
                **kwargs,
            )

            x = [me[0] for me in get_sorted(solution_stats, type=f"g_abs{suffix}", sortby=sortby)]
            g_abs_values = [me[1] for me in get_sorted(solution_stats, type=f"g_abs{suffix}", sortby=sortby)]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            ax.plot(x, g_abs_values, color=colors[key], linestyle=linestyles[q % 2], markeredgewidth=0.4, label=label)

    ax.set_xlabel(get_xlabel(along=along))

    ax.set_yscale("log", base=10)
    ax.set_ylim((1e-16, 1e3))

    ax.grid(linewidth=0.5)

    if created_fig:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=3)

        plot_name = f"{stat}_vs_{along}_{num_nodes=}_{dt=}.png"
        filename = "data" + "/" + f"{problem_name}" + "/" + "emb_thesis" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

        if not return_ax:
            plt.close(fig)
            return None
        
    return ax if return_ax else None


def convergence_plot_thesis(dt, num_nodes, along="iterations", QI="MIN-SR-NS", journal="BUW_thesis", return_axs=False):
    from pySDC.projects.DAE.run.plots_scaling import save_fig

    problem_name = "LINEAR-TEST"

    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.83)

    my_setup_mpl(fontsize=5)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    plot_differential_algebraic_errors_vs_metric(
        dt=dt,
        along=along,
        num_nodes=num_nodes,
        QI=QI,
        journal="BUW_thesis",
        axs=ax_flatten[:2],
        return_axs=return_axs,
    )

    plot_stats_vs_iterations(
        dt=dt,
        num_nodes=num_nodes,
        QI=QI,
        stat="increment",
        journal=journal,
        ax=ax_flatten[2],
        return_ax=return_axs,
    )

    for ax in ax_flatten:
        ax.set_xlim((1, 25))

    handles, labels = ax_flatten[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=5)

    ax_flatten[3].remove()

    plot_name = f"convergence_{num_nodes=}_{dt=}"
    save_fig(plt, plot_name, problem_name)


def increment_plot_different_sweepers_thesis(dt, num_nodes, journal="BUW_thesis"):
    from pySDC.projects.DAE.run.plots_scaling import save_fig

    problem_name = "LINEAR-TEST"

    QI_list = ["EE", "IE", "LU", "MIN-SR-S", "Picard"]

    figsize = figsize_by_journal(journal, scale=0.67, ratio=1.0)

    my_setup_mpl(fontsize=5.5)
    fig, axs = plt.subplots(3, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    for q, QI in enumerate(QI_list):
        ax_flatten[q].set_title(rf"$Q_\Delta=${QI}")
        plot_stats_vs_iterations(
            dt=dt,
            num_nodes=num_nodes,
            QI=QI,
            stat="increment",
            journal=journal,
            ax=ax_flatten[q],
            return_ax=False,
        )

    for ax in ax_flatten:
        ax.set_xlim((2, 25))
        ax.set_ylim((1e-16, 1e5))

    handles, labels = ax_flatten[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=5)

    ax_flatten[-1].remove()

    plot_name = f"increment_different_sweepers_{num_nodes=}_{dt=}"
    save_fig(plt, plot_name, problem_name)


if __name__ == "__main__":
    dt_list, _ = choose_time_step_sizes("LINEAR-TEST")
    dt = dt_list[3]

    QI = "MIN-SR-S"
    num_nodes = 4

    # Iterations
    # plot_stats_vs_nodes(stat="iterations", journal="BUW_thesis")
    # plot_stats_vs_time(stat="iterations", journal="BUW_thesis")

    # Increment
    # plot_stats_vs_nodes(stat="increment", journal="BUW_thesis")
    plot_stats_vs_time(dt=dt, num_nodes=num_nodes, QI=QI, stat="increment", journal="BUW_thesis")
    plot_stats_vs_iterations(dt=dt, num_nodes=num_nodes, QI=QI, stat="increment", journal="BUW_thesis")

    # Error
    # plot_stats_vs_nodes(stat="error", journal="BUW_thesis")
    plot_stats_vs_time(dt=dt, num_nodes=num_nodes, QI=QI, stat="error", journal="BUW_thesis")
    # plot_stats_vs_iterations(QI=QI, num_nodes=num_nodes, stat="error", journal="BUW_thesis")
    plot_differential_algebraic_errors_vs_metric(dt=dt, along="iterations", num_nodes=num_nodes, QI=QI, journal="BUW_thesis")
    plot_differential_algebraic_errors_vs_metric(dt=dt, along="time", num_nodes=num_nodes, QI=QI, journal="BUW_thesis")

    # Absolute value of g
    plot_absolute_value_g_vs_metric(dt, num_nodes=num_nodes, along="iterations", journal="BUW_thesis")
    plot_absolute_value_g_vs_metric(dt, num_nodes=num_nodes, along="time", journal="BUW_thesis")

    convergence_plot_thesis(dt=dt, num_nodes=num_nodes, along="iterations", journal="BUW_thesis")

    increment_plot_different_sweepers_thesis(dt=dt, num_nodes=num_nodes, journal="BUW_thesis")
