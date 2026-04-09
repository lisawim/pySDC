import numpy as np
from mpi4py import MPI
import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any, Optional
from itertools import combinations
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

from pySDC.core.hooks import Hooks

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.utils import set_correct_sweeper_type
from pySDC.projects.DAE.run.plot_order_iteration import (
    choose_time_step_sizes, sync_xlim, sync_ylim
)

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.mpi_test import build_filename, run_mpi_test
from pySDC.projects.DAE.run.solution import plot_solution_andrews
from pySDC.projects.DAE.misc.methods_config import SDC_METHODS
from pySDC.projects.DAE.run.plots_work_prec import (
    get_method_label, get_metric_key, get_ylabel_based_on_metric
)
from pySDC.projects.DAE.run.plots_scaling import (
    plot_wallclocktime_vs_mpi_ranks,
    plot_mean_iterations_vs_mpi_ranks,
    plot_error_vs_mpi_ranks,
    plot_embedded_error_vs_mpi_ranks,
)


def get_linestyles():
    return {
        "constrainedDAE": "dashed",
        "semiImplicitDAE": "dashdot",
    }


def save_fig(plot, plot_name, problem_name):
    out = Path("data") / problem_name / f"{plot_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(out, dpi=400, bbox_inches="tight")
    plot.close()


def set_nodes_for_plotting(stats_dict, nodes_to_plot):
    try:
        available_nodes = list(stats_dict.keys())
    except AttributeError: 
        available_nodes = list(stats_dict)

    nodes = (
        sorted(available_nodes) if nodes_to_plot is None else [n for n in nodes_to_plot if n in stats_dict]
    )
    return nodes


def compute_factors_one_step(
    all_stats: dict[str, dict[int, dict[str, float]]],
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    ref_QI: str = "LU",
    ref_num_nodes: int = 5,
    **kwargs: Any,
) -> tuple[
    dict[str, dict[str, dict[int, float]]],
    dict[str, dict[str, dict[int, float]]],
    dict[str, list[int]],
]:

    assert ref_QI in QI_serial_methods, f"Reference QI {ref_QI} must be in QI_serial_methods."

    factors = {}
    factors_par_vs_par = {}
    num_processes_by_ref = {}

    for sweeper_type_ser in sweepers:
        sweeper_type_ser_eff = set_correct_sweeper_type(sweeper_type_ser, ref_QI)

        key_ref = f"{sweeper_type_ser_eff}_{ref_QI}"

        if key_ref not in all_stats:
            continue
        if ref_num_nodes not in all_stats[key_ref]:
            continue

        timings_ref = all_stats[key_ref][ref_num_nodes].t_cpu_one_step
        if timings_ref is None:
            continue

        factors.setdefault(key_ref, {})
        available_parallel_nodes = set()

        for sweeper_type_par in sweepers:
            for QI_par in QI_parallel_methods:
                key_par = f"{sweeper_type_par}_{QI_par}"

                if key_par not in all_stats:
                    continue

                factors[key_ref].setdefault(key_par, {})

                for num_nodes, stats_par in sorted(all_stats[key_par].items()):
                    timings_par = stats_par.t_cpu_one_step
                    if timings_par is None:
                        continue

                    f = timings_ref[-1] / timings_par[-1]
                    factors[key_ref][key_par][num_nodes] = f
                    available_parallel_nodes.add(num_nodes)

        num_processes_by_ref[key_ref] = sorted(available_parallel_nodes)

    # parallel vs. parallel
    parallel_keys = []
    for sweeper_type_par in sweepers:
        for QI_par in QI_parallel_methods:
            key_par = f"{sweeper_type_par}_{QI_par}"
            if key_par in all_stats:
                parallel_keys.append(key_par)

    for key_par_1, key_par_2 in combinations(parallel_keys, 2):
        factors_par_vs_par.setdefault(key_par_1, {})
        factors_par_vs_par[key_par_1].setdefault(key_par_2, {})

        num_processes_1 = sorted(all_stats[key_par_1].keys())

        for num_nodes in num_processes_1:
            if num_nodes not in all_stats[key_par_2]:
                continue

            timings_1 = all_stats[key_par_1][num_nodes].t_cpu_one_step
            timings_2 = all_stats[key_par_2][num_nodes].t_cpu_one_step

            factors_par_vs_par[key_par_1][key_par_2][num_nodes] = timings_1[-1] / timings_2[-1]

    return factors, factors_par_vs_par, num_processes_by_ref


def plots_scaling(
    global_comm: MPI.Comm,
    hook_class: list[Hooks],
    problem_name: str,
    dt: float,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    ref_QI: str = "LU",
    ref_num_nodes: int = 5,
    nodes_to_plot: list[int] = range(2, 9),
    num_nodes_per_figure: list[int] = [2, 3, 4, 5],
    filename: str = None,
    **kwargs: Any,
) -> None:
    """Generates scaling plots."""

    global_rank = global_comm.Get_rank()

    base_path = os.path.join("data", problem_name, "results")
    precomputed_files = {
        "ANDREWS-SQUEEZER": f"results_scaling_{dt=}_andrews.pkl",
        "LINEAR-TEST": f"results_scaling_{dt=}_linear.pkl",
        "REACTION-DIFFUSION": f"results_scaling_{dt=}_reaction_diffusion.pkl",
    }

    if filename is not None:
        print("Use precomputed results..\n")

        results_file = filename
    else:
        if problem_name in precomputed_files:
            results_file = precomputed_files[problem_name]
            path = os.path.join(base_path, results_file)
            if not os.path.exists(path):
                run_mpi_test(
                    global_comm,
                    hook_class,
                    problem_name,
                    dt,
                    sweepers,
                    QI_serial_methods,
                    QI_parallel_methods,
                    **kwargs,
                )
                results_file = build_filename(dt, problem_name)

        else:
            run_mpi_test(
                global_comm,
                hook_class,
                problem_name,
                dt,
                sweepers,
                QI_serial_methods,
                QI_parallel_methods,
                **kwargs,
            )
            results_file = build_filename(dt, problem_name)

    path = os.path.join(base_path, results_file)
    if global_rank == 0:
        with open(path, "rb") as f:
            all_stats = dill.load(f)

        factors, factors_par_vs_par, num_processes_by_key_ser = compute_factors_one_step(
            all_stats, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs
        )

        metric_key = get_metric_key(problem_name=problem_name)

        plot_wallclocktime_vs_accuracy(
            all_stats=all_stats,
            problem_name=problem_name,
            sweepers=sweepers,
            QI_serial_methods=QI_serial_methods,
            QI_parallel_methods=QI_parallel_methods,
            metric_key=metric_key,
            nodes_to_plot=nodes_to_plot,
        )

        plot_time_to_accuracy(
            all_stats=all_stats,
            problem_name=problem_name,
            sweepers=sweepers,
            QI_serial_methods=QI_serial_methods,
            QI_parallel_methods=QI_parallel_methods,
            metric_key=metric_key,
            num_nodes_per_figure=num_nodes_per_figure,
        )

        print_speedup_for_one_step(
            problem_name=problem_name,
            factors=factors,
            factors_par_vs_par=factors_par_vs_par,
            sweepers=sweepers,
            nodes_to_plot=nodes_to_plot,
            ref_QI=ref_QI,
            QI_parallel_methods=QI_parallel_methods,
            ref_num_nodes=ref_num_nodes,
        )

        plot_functions = [
            plot_mean_iterations_vs_mpi_ranks,
            plot_error_vs_mpi_ranks,
            plot_embedded_error_vs_mpi_ranks,
        ]
        for plot_function in plot_functions:
            plot_function(
                all_stats,
                problem_name,
                sweepers,
                QI_serial_methods,
                QI_parallel_methods,
                nodes_to_plot=nodes_to_plot,
            )

        # for num_nodes in nodes_to_plot:
        #     for quantity in ["walltime", "error", "increment", "number_iterations"]:
        #         plot_quantity_over_time(
        #             all_stats=all_stats,
        #             dt=dt,
        #             quantity=quantity,
        #             problem_name=problem_name,
        #             sweepers=sweepers,
        #             num_nodes=num_nodes,
        #             QI_serial_methods=QI_serial_methods,
        #             QI_parallel_methods=QI_parallel_methods,
        #             **kwargs,
        #         )

        if problem_name == "ANDREWS-SQUEEZER":
            plot_impact_of_jumps_on_runtime_andrews(
                all_stats=all_stats,
                dt=dt,
                problem_name=problem_name,
                sweepers=sweepers,
                QI_serial_methods=QI_serial_methods,
                QI_parallel_methods=QI_parallel_methods,
                metric_key=metric_key,
                nodes_to_plot=nodes_to_plot,
            )


def plot_wallclocktime_vs_accuracy(
    all_stats: dict[str, dict[int, dict[str, float]]],
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    metric_key: str,
    nodes_to_plot: list[int] = None,
    journal: str = "Springer_Scientific_Computing",
    **kwargs: Any,
) -> None:
    r"""
    Plots wallclock time versus accuracy. The wrapper function is used to do that.

    Parameters
    ----------
    all_stats : dict
        Contains the statistics from the run. It maps from key of the form
        ``f"{sweeper_type}_{QI}"`` to the number of nodes to the statistics
        of dataclass ScalingRunStats.
    problem_name : str
        Name of the problem.
    sweepers : list of Sweeper
        Sweepers.
    QI_serial_methods : list of str
        The QIs indicate the serial methods, i.e., the serial SDC methods,
        the Radau methods, and the Runge-Kutta method(s).
    QI_parallel_methods : list of str
        Contains the QIs that indicate the parallel SDC schemes.
    serial_mode : str
        If set to ``"sdc"`` results are plotted against serial SDC methods.
        If it is set to ``"radau_rk"`` only Radau and Runge-Kutta methods
        are considered in plotting. Default is ``"sdc"``.
    nodes_to_plot : list of int
        The number of nodes that should be plottet. Default is ``None``, i.e.,
        all nodes from the run are plotted.
    journal : str, optional
        Name of the journal to obtain specified scale and height for figsize.
    """

    plot_names = {"LINEAR-TEST": "Fig2", "ANDREWS-SQUEEZER": "Fig6", "REACTION-DIFFUSION": "Fig8"}

    x_of = lambda st: st.t_wall
    if problem_name == "ANDREWS-SQUEEZER":
        metric_key = "q_max_final_error"
        y_of = lambda st: st.qend_error
    else:
        metric_key = "all_max_global_error"
        y_of = lambda st: max(st.e_global_steps)

    figsize = figsize_by_journal(journal, scale=0.45, ratio=0.7)

    my_setup_mpl(fontsize=6)
    colors, markers, _ = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    qi_all = QI_parallel_methods + QI_serial_methods
    qi_all = [qi for qi in qi_all if qi in SDC_METHODS]  # + [qi for qi in qi_all if qi not in SDC_METHODS]

    used_nodes_all = set()

    key_cache = []

    t_min, t_max = [], []
    for QI in qi_all:
        for sweeper_type in sweepers:
            sweeper_type_eff = set_correct_sweeper_type(sweeper_type=sweeper_type, QI=QI)
            key = f"{sweeper_type_eff}_{QI}"

            if key not in all_stats:
                continue

            if key not in key_cache:
                # Adding key to cache to avoid double plotting
                key_cache.append(key)

                nodes = set_nodes_for_plotting(all_stats[key], nodes_to_plot)
                used_nodes_all.update(nodes)

                xs, ys = [], []
                for num_nodes in nodes:
                    stats = all_stats[key][num_nodes]

                    x, y = x_of(stats), y_of(stats)
                    xs.append(x), ys.append(y)

                t_min.append(min(xs))
                t_max.append(max(xs))

                label = get_method_label(sweeper_type, QI)
                ax.loglog(
                    xs,
                    ys,
                    color=colors[key],
                    marker=markers[key],
                    label=label,
                )

    used_nodes_sorted = sorted(used_nodes_all)
    # ax.tick_params(axis="both", which="minor", bottom=True, left=False)
    ax.tick_params(axis="x", which="minor", bottom=True, length=3, width=0.8)
    ax.tick_params(axis="y", which="minor", left=False)
    ax.set_xlabel(r"wall-clock time in s (one run per $M$)")

    if used_nodes_sorted:
        ax.set_xticks(used_nodes_sorted)
        ax.set_xticklabels(used_nodes_sorted)

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)

    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.tick_params(axis="x", which="minor", bottom=True, length=2)
    ax.tick_params(axis="y", which="minor", left=False)

    ax.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
    ax.grid(which="minor", axis="x", linewidth=0.25, alpha=0.10)
    ax.grid(which="major", axis="y", linewidth=0.45, alpha=0.35)

    ylabel = get_ylabel_based_on_metric(metric_key=metric_key)
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 0.04), ncol=2)

    plot_name = plot_names[problem_name]
    save_fig(plt, plot_name, problem_name)


def _get_sorted_filtered_times_and_errors(stats) -> tuple[list[float], list[float]]:
    """Sorts times and errors and filters out non-finite and non-positive values."""

    t = getattr(stats, "t_cpu_one_step", None)
    e = getattr(stats, "e_global_one_step", None)
    if t is None or e is None:
        return None

    if len(e) != len(t):
        raise ValueError("Length of time and error arrays must match.")

    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=float)

    # Sort & filter
    order = np.argsort(t)
    t, e = t[order], e[order]
    mask = np.isfinite(t) & np.isfinite(e) & (t >= 0) & (e > 0)
    t, e = t[mask], e[mask]
    if t.size == 0:
        return None

    return t, e


def plot_time_to_accuracy(
    all_stats: dict[str, dict[int, dict[str, float]]],
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    metric_key: str,
    num_nodes_per_figure: list[int] = [2, 3, 4, 5],
    journal: str = "Springer_Scientific_Computing",
    **kwargs: Any,
):
    
    assert len(num_nodes_per_figure) == 4, "num_nodes_per_figure must have length 4."

    plot_names = {"LINEAR-TEST": "Fig3", "ANDREWS-SQUEEZER": "Fig7", "REACTION-DIFFUSION": "Fig9"}

    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)
    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()
    linestyles = get_linestyles()

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axs_flatten = axs.flatten()

    qi_all = QI_parallel_methods + QI_serial_methods
    qi_all = [qi for qi in qi_all if qi in SDC_METHODS]

    for n, num_nodes in enumerate(num_nodes_per_figure):
        axs_flatten[n].set_title(f"M = {num_nodes}")

        for QI in qi_all:
            s = 2.7
            for sweeper_type in sweepers:
                key = f"{sweeper_type}_{QI}"
                if key not in all_stats:
                    continue
                if num_nodes not in all_stats[key]:
                    continue

                stats = all_stats[key][num_nodes]
                if stats is None:
                    continue

                times, e_vals = _get_sorted_filtered_times_and_errors(stats)
                if len(times) == 0:
                    continue

                label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
                # print(f"Plotting {label} for M={num_nodes} with {len(times)} points.")
                axs_flatten[n].plot(
                    times,
                    e_vals,
                    color=colors[key],
                    marker=markers[key],
                    markersize=s,
                    markeredgecolor=colors[key],
                    markerfacecolor="none",
                    linestyle=linestyles[sweeper_type],
                    linewidth=1.0,
                    label=label,
                )
                s += 2.2

    for ax in axs_flatten:
        ax.set_xlabel("cumulative wall-clock time in s")

        ylabel = get_ylabel_based_on_metric(metric_key=metric_key, type="iter")
        ax.set_ylabel(ylabel)

        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)

        # ax.grid(linewidth=0.5, alpha=0.35, which="major", axis="both")
        ax.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
        ax.grid(which="minor", axis="x", linewidth=0.25, alpha=0.10)
        ax.grid(which="major", axis="y", linewidth=0.45, alpha=0.35)
        ax.tick_params(axis="both", which="minor", bottom=True, left=False)

    axs_flatten = sync_xlim(axs_flatten)
    axs_flatten = sync_ylim(axs_flatten)

    if problem_name == "ANDREWS-SQUEEZER":
        for ax in axs_flatten:
            ax.set_ylim(top=1.5e0)

    handles, labels = axs_flatten[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=len(qi_all))

    plot_name = plot_names[problem_name]
    save_fig(plt, plot_name, problem_name)


def get_method_label_from_key(key: str) -> str:
    sweeper_type, QI = key.rsplit("_", 1)
    return get_method_label(sweeper_type, QI)


def print_speedup_for_one_step(
    problem_name: str,
    factors: dict,
    factors_par_vs_par: dict,
    sweepers: list[str],
    QI_parallel_methods: list[str],
    nodes_to_plot: list[int] = None,
    ref_QI: str = "LU",
    ref_num_nodes: int = 5,
    filename_stem: str = "speedup_one_step",
):

    lines = [
        "Speedup report for one-step timings",
        "==================================",
        "",
        "Definition of the speedup factor:",
        "  factor = timing_ref[-1] / timing_par[-1]",
        "",
        "Here, timing_ref and timing_par are cumulative timings per iteration, i.e.",
        "  timing_ref = np.cumsum(np.asarray(tau_ref, dtype=float))",
        "  timing_par = np.cumsum(np.asarray(tau_par, dtype=float))",
        "",
        "Hence, timing_ref[-1] and timing_par[-1] are the total times to convergence.",
        "Therefore:",
        "  factor > 1   => parallel method is faster",
        "  factor = 1   => both methods have the same runtime",
        "  factor < 1   => parallel method is slower",
        "",
        f"Fixed serial reference: {ref_QI} with M={ref_num_nodes}",
        "",
        "For ANDREWS-SQUEEZER the speedup factors are computed based on the timings",
        "at the final time step, i.e., at Tend = 0.03.",
        "",
    ]

    # --------------------------------------------------
    # Section 1: parallel vs. fixed serial reference
    # --------------------------------------------------
    section = "=== Parallel vs. fixed serial reference ==="
    print(f"\n{section}")
    lines.append(section)
    lines.append("")

    for sweeper_type_ser in sweepers:
        sweeper_type_ser_eff = set_correct_sweeper_type(sweeper_type_ser, ref_QI)
        key_ref = f"{sweeper_type_ser_eff}_{ref_QI}"

        if key_ref not in factors:
            continue

        label_ref = get_method_label(sweeper_type_ser_eff, ref_QI)
        header = f"Reference serial method: {label_ref} with M={ref_num_nodes}"
        print(header)
        lines.append(header)
        lines.append("")

        for sweeper_type_par in sweepers:
            for QI_par in QI_parallel_methods:
                key_par = f"{sweeper_type_par}_{QI_par}"

                if key_par not in factors[key_ref]:
                    continue

                factor_dict = factors[key_ref][key_par]
                if not factor_dict:
                    continue

                label_par = get_method_label(sweeper_type_par, QI_par)
                subheader = f"Comparison with parallel method: {label_par}"
                print(subheader)
                lines.append(subheader)
                lines.append("")

                for num_nodes, factor in sorted(factor_dict.items()):
                    if num_nodes > nodes_to_plot[-1]:
                        continue

                    if factor >= 1.0:
                        line = (
                            f"  M=P={num_nodes}: {label_par} is {factor:.3f}x faster "
                            f"than {label_ref} with M={ref_num_nodes}."
                        )
                    else:
                        line = (
                            f"  M=P={num_nodes}: {label_par} is {1.0 / factor:.3f}x slower "
                            f"than {label_ref} with M={ref_num_nodes}."
                        )

                    print(line)
                    lines.append(line)

                print("")
                lines.append("")

    # --------------------------------------------------
    # Section 2: parallel vs. parallel
    # --------------------------------------------------
    section = "=== Parallel vs. parallel ==="
    print(f"\n{section}")
    lines.append(section)
    lines.append("")

    for key_par_1, comparisons in factors_par_vs_par.items():
        label_1 = get_method_label_from_key(key_par_1)

        for key_par_2, factor_dict in comparisons.items():
            if not factor_dict:
                continue

            label_2 = get_method_label_from_key(key_par_2)

            subheader = f"Comparison: {label_1} vs. {label_2}"
            print(subheader)
            lines.append(subheader)
            lines.append("")

            for num_nodes, factor in sorted(factor_dict.items()):
                if num_nodes > nodes_to_plot[-1]:
                    continue

                if factor >= 1.0:
                    line = (
                        f"  M=P={num_nodes}: {label_1} is {factor:.3f}x faster "
                        f"than {label_2}."
                    )
                else:
                    line = (
                        f"  M=P={num_nodes}: {label_1} is {1.0 / factor:.3f}x slower "
                        f"than {label_2}."
                    )

                print(line)
                lines.append(line)

            print("")
            lines.append("")

    # --------------------------------------------------
    # Save report
    # --------------------------------------------------
    out = Path("data") / problem_name / f"{filename_stem}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_quantity_over_time(
    all_stats: dict[str, dict[int, dict[str, float]]],
    dt: float,
    quantity: str,
    problem_name: str,
    sweepers: list[str],
    num_nodes: int,
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    journal: str = "Springer_Scientific_Computing",
    ax: Optional[Axes] = None,
    return_ax: bool = False,
    **kwargs: Any,
) -> None:
    
    if quantity == "walltime":
        y_of = lambda st: st.t_cpu_steps
    elif quantity == "error":
        y_of = lambda st: st.e_global_steps
    elif quantity == "increment":
        y_of = lambda st: st.e_embedded_steps
    elif quantity == "number_iterations":
        y_of = lambda st: st.niter_steps

    _, Tend = choose_time_step_sizes(problem_name=problem_name)
    t = [i * dt for i in range(1, int(Tend / dt) + 1)]

    colors, _, _ = my_plot_style_config()

    qi_all = QI_parallel_methods + QI_serial_methods
    qi_all = [qi for qi in qi_all if qi in SDC_METHODS]

    created_fig = ax is None

    if created_fig:
        my_setup_mpl(fontsize=7)
        figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    for QI in qi_all:
        for sweeper_type in sweepers:
            key = f"{sweeper_type}_{QI}"
            if key not in all_stats:
                continue

            stats = all_stats[key][num_nodes]
            if stats is None:
                continue

            label = get_method_label(sweeper_type, QI)
            ax.plot(
                t, 
                y_of(stats),
                color=colors[key],
                linewidth=1.0,
                label=label,
            )

    ax.set_xlabel(r"time $t$")

    ylabel = "wall-clock time in s" if quantity == "walltime" else f"{quantity}"
    ax.set_ylabel(ylabel)

    ax.set_yscale("log", base=10)

    ax.grid(linewidth=0.5)

    ax.tick_params(axis="both", which="minor", bottom=False, left=True)

    if created_fig:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

        out = Path("data") / problem_name / f"{quantity}_over_time_{num_nodes=}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=400, bbox_inches="tight")

        if not return_ax:
            plt.close(fig)
            return None

    return ax if return_ax else None


def plot_impact_of_jumps_on_runtime_andrews(
    all_stats: dict[str, dict[int, dict[str, float]]],
    dt: float,
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    ref_QI: str = "LU",
    ref_num_nodes: int = 3,
    journal: str = "Springer_Scientific_Computing",
    return_ax: bool = False,
    **kwargs: Any,
) -> None:
    
    _, Tend = choose_time_step_sizes(problem_name=problem_name)
    t = [i * dt for i in range(1, int(Tend / dt) + 1)]
    
    figsize = figsize_by_journal(journal, scale=1.3, ratio=1.0)
    my_setup_mpl(fontsize=16)

    qi_all = QI_parallel_methods + QI_serial_methods
    qi_all = [qi for qi in qi_all if qi in SDC_METHODS]

    fig, ax = plt.subplots(2, 1, figsize=figsize)

    plot_solution_andrews(dt=dt, num_nodes=ref_num_nodes, QI=ref_QI, ax=ax[0], return_ax=return_ax)

    plot_quantity_over_time(
        all_stats=all_stats,
        dt=dt,
        quantity="walltime",
        problem_name=problem_name,
        sweepers=sweepers,
        num_nodes=ref_num_nodes,
        QI_serial_methods=QI_serial_methods,
        QI_parallel_methods=QI_parallel_methods,
        nodes_to_plot=[ref_num_nodes],
        journal=journal,
        ax=ax[1],
        return_ax=return_ax,
    )

    for ax_obj in ax:
        for line in ax_obj.lines:
            line.set_linewidth(2.5)

        for spine in ax_obj.spines.values():
            spine.set_linewidth(1.5)

    ax[0].tick_params(axis="both", which="major", width=1.5, length=7.0)
    ax[1].tick_params(axis="both", which="major", width=1.5, length=7.0)
    ax[1].tick_params(axis="both", which="minor", width=1.5, length=3.5)

    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=7)
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3)

    for ax_obj in ax:
        ax_obj.set_xlim((dt, 0.03))

        ax_obj.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
        ax_obj.grid(which="minor", axis="x", linewidth=0.25, alpha=0.10)
        ax_obj.grid(which="major", axis="y", linewidth=0.45, alpha=0.35)
        ax_obj.grid(which="minor", axis="y", linewidth=0.25, alpha=0.10)

    out = Path("data") / problem_name / "Fig5.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def make_plots():
    global_comm = MPI.COMM_WORLD
    
    # Plots for LINEAR-TEST
    # print("\nGenerating plots for LINEAR-TEST...\n")
    # config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    # filename = "results_scaling_dt=0.05_linear_#6.pkl"
    # plots_scaling(
    #     global_comm=global_comm, filename=filename, **config_linear
    # )

    # Plots for ANDREWS-SQUEEZER
    print("\nGenerating plots for ANDREWS-SQUEEZER...\n")
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="scaling")
    filename = "results_scaling_dt=0.001_andrews_#8.pkl"
    nodes_to_plot = range(2, 17)
    plots_scaling(
        global_comm=global_comm, filename=filename, nodes_to_plot=nodes_to_plot, **config_andrews
    )

    # Plots for REACTION-DIFFUSION
    # print("\nGenerating plots for REACTION-DIFFUSION...\n")
    # config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="scaling")
    # filename = "results_scaling_dt=0.05_reaction_diffusion_#2.pkl"
    # plots_scaling(
    #     global_comm=global_comm, filename=filename, **config_reacdiff
    # )



if __name__ == "__main__":
    make_plots()
