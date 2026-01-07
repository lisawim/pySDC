import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.utils import compute_solution, newton_tol
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config

from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep, LogGlobalErrorPostIter
from pySDC.projects.DAE.misc.hooksDAE import (
    LogGlobalErrorPostIterDiff,
    LogGlobalErrorPostIterAlg,
    LogGlobalErrorPostStepDifferentialVariable,
    LogGlobalErrorPostStepAlgebraicVariable,
)
from pySDC.projects.DAE.problems.reactionDiffusionPDAE import LogGlobalErrorPostIterAlgebraicEquation
from pySDC.implementations.hooks.log_embedded_error_estimate import (
    LogEmbeddedErrorEstimate, LogEmbeddedErrorEstimatePostIter
)


def plot_error_vs_iteration(problem_name="LINEAR-TEST", sweeper_type="constrainedDAE", journal="Springer_Scientific_Computing"):
    figsize = figsize_by_journal(journal, scale=0.5, ratio=0.9)

    QI_list = ["IE", "EE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_fix = 1e-1
    num_nodes_list = range(2, 17)
    num_nodes_fix = 6

    t0 = 0.0

    hook_class = [LogGlobalErrorPostIter]

    my_setup_mpl(fontsize=8)
    colors, markers, _ = my_plot_style_config()
    for d, dt in enumerate(dt_list):
        fig, axs = plt.subplots(1, 1, figsize=figsize)

        for q, QI in enumerate(QI_list):
            if (sweeper_type, QI) not in [("fullyImplicitDAE", "EE"), ("fullyImplicitDAE", "Picard")]:
                key = f"constrainedDAE_{QI}"

                solution_stats = compute_solution(
                    problem_name,
                    t0,
                    dt,
                    t0 + dt,
                    num_nodes_fix,
                    QI,
                    sweeper_type,
                    False,
                    hook_class=hook_class,
                    measure=False,
                    e_tol=-1,
                    maxiter=15,
                )

                x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_post_iteration", sortby="iter")]
                err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_iteration", sortby="iter")]

                axs.semilogy(x, err_values, color=colors[key], label=f"{QI}")

        axs.set_xlabel(r"iteration $k$")
        axs.set_ylabel(r"$L_\infty$ error $||u(t_0 + \Delta t) - u^k_M||$")

        axs.set_yscale("log", base=10)

        axs.set_ylim((1e-16, 1e0))

        axs.grid(linewidth=0.5)
        # axs.set_axisbelow(True)

        handles, labels = axs.get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)

        plot_name = f"{d+1}_error_iteration_{sweeper_type}_{dt=}.png"
        filename = "data" + "/" + f"{problem_name}" + "/" + "error_iteration" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

    for num_nodes in num_nodes_list:
        fig, axs = plt.subplots(1, 1, figsize=figsize)

        for q, QI in enumerate(QI_list):
            if (sweeper_type, QI) not in [("fullyImplicitDAE", "EE"), ("fullyImplicitDAE", "Picard")]:
                key = f"constrainedDAE_{QI}"

                solution_stats = compute_solution(
                    problem_name,
                    t0,
                    dt_fix,
                    t0 + dt_fix,
                    num_nodes,
                    QI,
                    sweeper_type,
                    False,
                    hook_class=hook_class,
                    measure=False,
                    e_tol=-1,
                    maxiter=15,
                )

                x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_post_iteration", sortby="iter")]
                err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_iteration", sortby="iter")]

                axs.semilogy(x, err_values, color=colors[key], label=f"{QI}")

        axs.set_xlabel(r"iteration $k$")
        axs.set_ylabel(r"$L_\infty$ error $||u(t_0 + \Delta t) - u^k_M||$")

        axs.set_yscale("log", base=10)

        axs.set_ylim((1e-16, 1e0))

        axs.grid(linewidth=0.5)
        # axs.set_axisbelow(True)

        handles, labels = axs.get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)

        plot_name = f"error_iteration_{sweeper_type}_{num_nodes=}.png"
        filename = "data" + "/" + f"{problem_name}" + "/" + "error_iteration" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

def run_and_plot_error_vs_iteration(dt, num_nodes, problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"):
    figsize = figsize_by_journal(journal, scale=0.71, ratio=0.6)

    sweeper_types = ["constrainedDAE"]#["constrainedDAE", "semiImplicitDAE", "fullyImplicitDAE"]
    QI_list = ["EE", "MIN-SR-NS", "Picard"]

    t0 = 0.0

    hook_class = [LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg]

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(3, 2, figsize=(18, 12))
    ax_flatten = axs.flatten()
    for q, QI in enumerate(QI_list):
        ax_flatten[2 * q].set_title(QI)
        ax_flatten[2 * q + 1].set_title(QI)

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
                maxiter=40,
            )

            x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]
            err_diff_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_post_iteration", sortby="iter")]
            err_alg_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]

            ax_flatten[2 * q].semilogy(x, err_diff_values, color=colors[key], label=sweeper_labels[sweeper_type])
            ax_flatten[2 * q + 1].semilogy(x, err_alg_values, color=colors[key])

        ax_flatten[2 * q].set_xlabel(r"iteration")
        ax_flatten[2 * q + 1].set_xlabel(r"iteration")

        ax_flatten[2 * q].set_ylabel(r"global error in $y$")
        ax_flatten[2 * q + 1].set_ylabel(r"global error in $z$")

        ax_flatten[2 * q].set_ylim((1e-15, 1e0))
        ax_flatten[2 * q + 1].set_ylim((1e-15, 1e0))

        # ax_flatten[2 * q].set_yscale(scale="log", base=10)
        # ax_flatten[2 * q + 1].set_yscale(scale="log", base=10)

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)

    plot_name = f"plot_error_vs_iteration_{num_nodes=}_{dt=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def run_and_plot_error_vs_time(dt, num_nodes, problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"):
    figsize = figsize_by_journal(journal, scale=0.6, ratio=0.9)

    sweeper_types = ["constrainedDAE"]#["constrainedDAE", "semiImplicitDAE", "fullyImplicitDAE"]
    QI_list = ["IE", "LU", "MIN-SR-S"]

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    hook_class = [LogGlobalErrorPostStep]

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for q, QI in enumerate(QI_list):
        for sweeper_type in sweeper_types:
            print(f"Running for {sweeper_type} with {QI}..")
            key = f"{sweeper_type}_{QI}"

            runtime, solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                Tend,
                num_nodes,
                QI,
                sweeper_type,
                False,
                hook_class=hook_class,
                measure=True,  # False,
            )
            print(f"Runtime: {runtime} s")
            x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]
            err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]
            print([me[1] for me in get_sorted(solution_stats, type=f"niter", sortby="time")])
            label = sweeper_labels[sweeper_type] + "-" + f"{QI}" + "-" + f"{runtime:.2f} s"
            # ax.semilogy(x, err_values, color=colors[key], label=label)
            ax.semilogy(x, err_values, label=label)

        ax.set_xlabel(r"time")

        ax.set_ylabel(r"global error")

        ax.set_ylim((1e-15, 1e3))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)

    plot_name = f"error_vs_time_{num_nodes=}_{dt=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def plot_qend_error_vs_runtime(num_nodes, problem_name="ANDREWS-SQUEEZER", journal="Springer_Scientific_Computing"):
    from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import qend_ref_testset
    from pySDC.implementations.hooks.log_solution import LogSolution

    figsize = figsize_by_journal(journal, scale=0.6, ratio=0.9)

    sweeper_types = ["semiImplicitDAE"]#["constrainedDAE", "semiImplicitDAE", "fullyImplicitDAE"]
    QI_list = ["EE", "IE", "LU", "MIN-SR-NS"]

    t0 = 0.0
    dt_list, Tend = choose_time_step_sizes(problem_name)

    hook_class = [LogSolution]

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    for sweeper_type in sweeper_types:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for q, QI in enumerate(QI_list):
            err_all, runtimes_all = [], []
            for dt in dt_list:
                print(f"Running {dt} for {sweeper_type} with {QI}..")
                key = f"{sweeper_type}_{QI}"

                runtime, solution_stats = compute_solution(
                    problem_name,
                    t0,
                    dt,
                    Tend,
                    num_nodes,
                    QI,
                    sweeper_type,
                    False,
                    hook_class=hook_class,
                    measure=True,  # False,
                )
                print(f"Runtime: {runtime:.3f} s")
                runtimes_all.append(runtime)

                u_val = get_sorted(solution_stats, type="u", sortby="time")
                t = np.array([me[0] for me in u_val])
                u = np.array([me[1].flatten() for me in u_val])
                q = u[:, : 7]

                i = np.searchsorted(t, Tend)
                if i < len(t) and np.isclose(t[i], Tend, atol=1e-14):
                    ind = i
                elif i > 0 and np.isclose(t[i-1], Tend, atol=1e-14):
                    ind = i - 1
                else:
                    print("No suitable entry found.")

                t_ref = t[ind]
                qend_ref = qend_ref_testset(t_ref)

                qend = q[ind, :]
                qend_max_final_err = max(abs(qend - qend_ref))
                err_all.append(qend_max_final_err)
                print([me[1] for me in get_sorted(solution_stats, type=f"niter", sortby="time")])

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            # ax.semilogy(x, err_values, color=colors[key], label=label)
            ax.loglog(runtimes_all, err_all, color=colors[key], marker=markers[key], label=label)

        ax.set_xlabel(r"runtime")

        ax.set_ylabel(r"qend error")

        ax.set_ylim((1e-10, 1e-1))

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)

        plot_name = f"qend_error_vs_time_{num_nodes=}.png"
        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

def plot_error_vs_runtime(num_nodes, problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"):
    figsize = figsize_by_journal(journal, scale=0.6, ratio=0.9)

    sweeper_types = ["constrainedDAE"]#["constrainedDAE", "semiImplicitDAE", "fullyImplicitDAE"]
    QI_list = ["IE", "LU", "MIN-SR-NS"]

    t0 = 0.0
    dt_list, Tend = choose_time_step_sizes(problem_name)

    hook_class = [LogGlobalErrorPostStep]

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    for sweeper_type in sweeper_types:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        for q, QI in enumerate(QI_list):
            err_all, runtimes_all = [], []
            for dt in dt_list:
                print(f"Running {dt} for {sweeper_type} with {QI}..")
                key = f"{sweeper_type}_{QI}"

                runtime, solution_stats = compute_solution(
                    problem_name,
                    t0,
                    dt,
                    Tend,
                    num_nodes,
                    QI,
                    sweeper_type,
                    False,
                    hook_class=hook_class,
                    measure=True,  # False,
                )
                print(f"Runtime: {runtime:.3f} s")
                runtimes_all.append(runtime)

                err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]
                err_all.append(max(err_values))

                print([me[1] for me in get_sorted(solution_stats, type=f"niter", sortby="time")])

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            ax.loglog(runtimes_all, err_all, color=colors[key], marker=markers[key], label=label)

        ax.set_xlabel(r"runtime")

        ax.set_ylabel(r"global error")

        ax.set_ylim((1e-16, 1e0))

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2)

        plot_name = f"error_vs_runtime_{num_nodes=}.png"
        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

def plot_embedded_error_vs_iteration(dt, num_nodes, problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"):
    figsize = figsize_by_journal(journal, scale=0.71, ratio=0.6)

    sweeper_types = ["constrainedDAE", "semiImplicitDAE", "fullyImplicitDAE"]
    QI_list = ["IE", "LU", "MIN-SR-S"]

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    hook_class = [LogEmbeddedErrorEstimatePostIter]

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    ax_flatten = axs.flatten()

    for q, QI in enumerate(QI_list):
        ax_flatten[q].set_title(QI)

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
                maxiter=20,
                e_tol=-1,
            )

            x = [me[0] for me in get_sorted(solution_stats, type=f"error_embedded_estimate_post_iteration", sortby="iter")]
            embedded_err_values = [me[1] for me in get_sorted(solution_stats, type=f"error_embedded_estimate_post_iteration", sortby="iter")]

            ax_flatten[q].semilogy(x, embedded_err_values, color=colors[key], label=sweeper_labels[sweeper_type])

        ax_flatten[q].set_xlabel("iteration")

        ax_flatten[q].set_ylabel("embedded error estimate")

        ax_flatten[q].set_ylim((1e-15, 1e0))

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)

    plot_name = f"embedded_error_vs_iteration_{num_nodes=}_{dt=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def plot_embedded_error_vs_time(dt, num_nodes, problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"):
    figsize = figsize_by_journal(journal, scale=0.71, ratio=0.6)

    sweeper_types = ["constrainedDAE", "semiImplicitDAE", "fullyImplicitDAE"]
    QI_list = ["IE", "LU", "MIN-SR-S"]

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    hook_class = [LogEmbeddedErrorEstimate]

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    ax_flatten = axs.flatten()

    for q, QI in enumerate(QI_list):
        ax_flatten[q].set_title(QI)

        for sweeper_type in sweeper_types:
            print(f"Running for {sweeper_type} with {QI}..")
            key = f"{sweeper_type}_LU"

            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                Tend,
                num_nodes,
                QI,
                sweeper_type,
                False,
                hook_class=hook_class,
                measure=False,
                e_tol=1e-8,
            )

            x = [me[0] for me in get_sorted(solution_stats, type=f"error_embedded_estimate", sortby="time")]
            embedded_err_values = [me[1] for me in get_sorted(solution_stats, type=f"error_embedded_estimate", sortby="time")]

            ax_flatten[q].semilogy(x, embedded_err_values, color=colors[key], label=sweeper_labels[sweeper_type])

        ax_flatten[q].set_xlabel("time")

        ax_flatten[q].set_ylabel("embedded error estimate")

        ax_flatten[q].set_ylim((1e-15, 1e0))

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)

    plot_name = f"plot_embedded_error_vs_time_{num_nodes=}_{dt=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def plot_algebraic_error_vs_iteration(
        dt, num_nodes, problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing", format="png"
    ):
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.6)
    pair_to_compare = ("constrainedDAE", "MIN-SR-NS") if problem_name in ["ANDREWS-SQUEEZER", "LINEAR-TEST"] else ("constrainedDAE", "MIN-SR-S")

    sweeper_type_QI_pair_list = [
        pair_to_compare,
        # ("constrainedDAE", "IE"),
        # ("constrainedDAE", "LU"),
        ("fullyImplicitDAE", "IE"),
        ("fullyImplicitDAE", "LU"),
        ("fullyImplicitDAE", "MIN-SR-S"),
        ("fullyImplicitDAE", "MIN-SR-NS"),
        # ("semiImplicitDAE", "IE"),
        # ("semiImplicitDAE", "LU"),
        # ("semiImplicitDAE", "MIN-SR-S"),
        # ("semiImplicitDAE", "MIN-SR-NS"),
    ]
    maxiter = 20

    t0 = 0.0

    hook_class = [LogGlobalErrorPostIterAlg]

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()
    linestyles = ["solid", "dashed", "dotted"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for sweeper_type_QI_pair in sweeper_type_QI_pair_list:
        sweeper_type, QI = sweeper_type_QI_pair
        if (sweeper_type, QI) != ("fullyImplicitDAE", "Picard"):
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

            x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]
            err_alg_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            ax.semilogy(x, err_alg_values, color=colors[key], marker=markers[key], label=label)

        ax.set_xlabel(r"iteration $k$")
        ax.set_xticks(np.arange(1, maxiter + 1, 2))

        if problem_name == "ANDREWS-SQUEEZER":
            ax.set_ylabel(r"LTE $||z(t_1) - z^k_{M,t_1}||_{\infty}$")
            ax.set_ylim((1e-12, 1e7))
        elif problem_name == "REACTION-DIFFUSION":
            ax.set_ylabel(r"LTE $||w(t_1) - w^k_{M,t_1}||_{\infty}$")
            ax.set_ylim((1e-16, 1e-3))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    ax.grid(linewidth=0.5)

    plot_name = f"Fig7"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def plot_algebraic_equation_vs_iteration(dt, num_nodes=3, problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"):
    figsize = figsize_by_journal(journal, scale=0.71, ratio=0.6)

    sweeper_type = "constrainedDAE"
    QI_list = ["IE", "LU", "MIN-SR-S"]

    t0 = 0.0

    hook_class = [LogGlobalErrorPostIterAlgebraicEquation]

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, 1, figsize=figsize)

    for q, QI in enumerate(QI_list):
        axs.set_title(QI)

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
            maxiter=2 * num_nodes - 1,
        )

        x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_algebraic_eq_post_iteration", sortby="iter")]
        embedded_err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_eq_post_iteration", sortby="iter")]

        axs.semilogy(x, embedded_err_values, color=colors[key], label=sweeper_labels[sweeper_type] + "-" + f"{QI}")

        axs.set_xlabel("iteration")

        axs.set_ylabel("algebraic equation")

        axs.set_ylim((1e-15, 1e0))

    handles, labels = axs.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)

    plot_name = f"algebraic_equation_vs_iteration_{num_nodes=}_{dt=}_{sweeper_type}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)
