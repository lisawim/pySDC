import numpy as np
from mpi4py import MPI
import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Callable, Optional
from itertools import combinations

from pySDC.core.hooks import Hooks

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.utils import set_correct_sweeper_type
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.mpi_test import build_filename, run_mpi_test
from pySDC.projects.DAE.misc.methods_config import QI_SERIAL, SDC_METHODS, RADAU_METHODS, RK_METHODS
from pySDC.projects.DAE.run.plots_work_prec import get_method_label, get_metric_key, get_ylabel_based_on_metric


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


def compute_speedups_and_efficiencies(
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

    speedups = {}
    efficiencies = {}
    num_processes_by_ref = {}

    for sweeper_type_ser in sweepers:
        sweeper_type_ser_eff = set_correct_sweeper_type(sweeper_type_ser, ref_QI)
        key_ref = f"{sweeper_type_ser_eff}_{ref_QI}"

        if key_ref not in all_stats:
            continue
        if ref_num_nodes not in all_stats[key_ref]:
            continue

        t_ref = all_stats[key_ref][ref_num_nodes].t_wall
        if t_ref is None:
            continue

        speedups.setdefault(key_ref, {})
        efficiencies.setdefault(key_ref, {})

        available_parallel_nodes = set()

        for sweeper_type_par in sweepers:
            for QI_par in QI_parallel_methods:
                key_par = f"{sweeper_type_par}_{QI_par}"

                if key_par not in all_stats:
                    continue

                speedups[key_ref].setdefault(key_par, {})
                efficiencies[key_ref].setdefault(key_par, {})

                for num_nodes, stats_par in sorted(all_stats[key_par].items()):
                    t_par = stats_par.t_wall
                    if t_par is None:
                        continue

                    s = t_ref / t_par
                    e = s / num_nodes

                    speedups[key_ref][key_par][num_nodes] = s
                    efficiencies[key_ref][key_par][num_nodes] = e
                    available_parallel_nodes.add(num_nodes)

    return speedups, efficiencies, num_processes_by_ref


def get_serial_methods_and_key_builder(
    serial_mode: str, QI_serial_methods: list[str]
) -> tuple[list[str], Callable[[str, str], str]]:
    """Returns serial methods and key builder depending on serial mode."""
    if serial_mode == "sdc":
        serial_methods = [QI for QI in QI_serial_methods if QI in QI_SERIAL]

        def make_serial_key(sweeper_type: str, QI_ser: str) -> str:
            return f"{sweeper_type}_{QI_ser}"

    elif serial_mode == "radau_rk":
        allowed = set(RADAU_METHODS + RK_METHODS)
        serial_methods = [QI for QI in QI_serial_methods if QI in allowed]

        def make_serial_key(sweeper_type: str, QI_ser: str) -> str:
            sweeper_type_eff = set_correct_sweeper_type(sweeper_type, QI_ser)
            return f"{sweeper_type_eff}_{QI_ser}"

    else:
        raise ValueError(f"Unknown serial_mode={serial_mode!r}. " "Use 'sdc' or 'radau_rk'.")

    return serial_methods, make_serial_key


def plots_scaling(
    global_comm: MPI.Comm,
    hook_class: list[Hooks],
    problem_name: str,
    dt: float,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    nodes_to_plot: list[int] = None,
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
        results_file = filename
    else:
        if problem_name in precomputed_files:
            print("Use precomputed results..\n")
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

        metric_key = get_metric_key(problem_name=problem_name)

        speedups, efficiencies, num_processes_by_key_ser = compute_speedups_and_efficiencies(
            all_stats, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs
        )

        factors, factors_par_vs_par, num_processes_by_key_ser = compute_factors_one_step(
            all_stats, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs
        )

        plot_speedup_and_efficiency(
            problem_name,
            speedups,
            efficiencies,
            num_processes_by_key_ser,
            sweepers,
            QI_serial_methods,
            QI_parallel_methods,
            nodes_to_plot=nodes_to_plot,
            **kwargs,
        )

        plot_functions = [
            plot_wallclocktime_vs_mpi_ranks,
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

        plot_time_to_accuracy_for_each_num_nodes(
            all_stats,
            problem_name=problem_name,
            factors=factors,
            factors_par_vs_par=factors_par_vs_par,
            sweepers=sweepers,
            QI_parallel_methods=QI_parallel_methods,
            nodes_to_plot=nodes_to_plot,
            filename_stem="time_to_accuracy",
            format="png",
            metric_key=metric_key,
        )

        plot_wallclocktime_vs_accuracy(
            all_stats=all_stats,
            problem_name=problem_name,
            sweepers=sweepers,
            QI_serial_methods=QI_serial_methods,
            QI_parallel_methods=QI_parallel_methods,
            metric_key=metric_key,
            nodes_to_plot=nodes_to_plot,
        )

        # plot_wallclocktime_over_time(
        #     all_stats=all_stats,
        #     dt=dt,
        #     problem_name=problem_name,
        #     sweepers=sweepers,
        #     QI_serial_methods=QI_serial_methods,
        #     QI_parallel_methods=QI_parallel_methods,
        #     nodes_to_plot=nodes_to_plot,
        # )


def plot_speedup_and_efficiency(
    problem_name: str,
    speedups: dict[str, dict[str, dict[int, float]]],
    efficiencies: dict[str, dict[str, dict[int, float]]],
    num_processes_by_key_ser: dict[str, list[int]],
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    ref_QI: str = "LU",
    ref_num_nodes: int = 5,
    nodes_to_plot: list[int] = None,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    **kwargs: Any,
) -> None:
    r"""
    Plots speedup and efficiency.

    Parameters
    ----------
    problem_name : str
        Name of the problem.
    speedups : dict
        Contains the speedups.
    efficiencies : dict
        Contains the efficiencies.
    num_processes_by_key_ser : dict
        Contains the number of processes that are used in the run.
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
    format : str
        Format of plot. Default is ``"png"``.
    """

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    for sweeper_type_ser in sweepers:
        sweeper_type_ser_eff = set_correct_sweeper_type(sweeper_type_ser, ref_QI)
        key_ref = f"{sweeper_type_ser_eff}_{ref_QI}"

        if key_ref not in speedups:
            continue

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        used_nodes_all = set()

        available_nodes = num_processes_by_key_ser.get(key_ref, [])
        nodes = available_nodes if nodes_to_plot is None else [n for n in nodes_to_plot if n in available_nodes]

        for sweeper_type_par in sweepers:
            for QI_par in QI_parallel_methods:
                key_par = f"{sweeper_type_par}_{QI_par}"
                if key_par not in speedups[key_ref]:
                    continue

                s_map = speedups[key_ref][key_par]
                e_map = efficiencies[key_ref][key_par]

                xs = [n for n in nodes if n in s_map and n in e_map]
                if not xs:
                    continue

                used_nodes_all.update(xs)
                ys_s = [s_map[n] for n in xs]
                ys_e = [e_map[n] for n in xs]

                label = get_method_label(sweeper_type_par, QI_par)

                axs[0].semilogx(xs, ys_s, color=colors[key_par], marker=markers[key_par], label=label)
                axs[1].semilogx(xs, ys_e, color=colors[key_par], marker=markers[key_par])

        used_nodes_sorted = sorted(used_nodes_all)
        for ax in axs:
            ax.tick_params(axis="both", which="minor", bottom=False, left=True)
            ax.set_xlabel(r"number of $\mathtt{MPI}$ ranks")
            ax.set_xticks(used_nodes_sorted)
            ax.set_xticklabels(used_nodes_sorted)
            ax.set_xscale("log", base=2)
            ax.grid(linewidth=0.5)

        axs[0].set_yscale("log", base=10)
        axs[0].set_ylabel("speedup")
        axs[0].set_ylim(bottom=1e0)

        axs[1].set_ylabel("efficiency")
        axs[1].set_ylim((0.0, 1.0))

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

        filename = Path("data") / problem_name / f"scaling_ref_{ref_QI}_M={ref_num_nodes}.{format}"
        filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)


def _plot_metric_vs_mpi_ranks(
    all_stats: dict[str, dict[int, Any]],
    *,
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    y_of: Callable[[Any], float],
    ylabel: str,
    filename_stem: str,
    nodes_to_plot: Optional[list[int]] = None,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    ylim: Optional[tuple[float, float]] = None,
    yscale_log_base: Optional[int] = 10,
) -> None:
    r"""
    Generic plotting routine for y(metric) versus number of MPI ranks. Note that the number of
    collocation nodes is equal to the number of MPI ranks. Since the data is stored as a data
    class, it is only possible to plot the respective attributes from the data class. These
    are then transferred using the ``y_of`` function.

    Parameters
    ----------
    all_stats : dict
        Contains the statistics from the run. It maps from key of the form
        ``f"{sweeper_type}_{QI}"`` to the number of nodes to the statistics
        of dataclass ScalingRunStats.
    problem_name : str
        Name of the problem.
    sweepers : str
        List of sweepers.
    QI_serial_methods : list of str
        The QIs indicate the serial methods, i.e., the serial SDC methods,
        the Radau methods, and the Runge-Kutta method(s).
    QI_parallel_methods : list of str
        Contains the QIs that indicate the parallel SDC schemes.
    y_of : callable
        Function that maps a per-node stats object to a scalar y-value.
    filename_stem : str
        Filename for which the plot is stored. It is saved under
        data/{problem_name}/{filename_stem}.{format}.
    nodes_to_plot : list of int
        The number of nodes that should be plottet. Default is ``None``, i.e.,
        all nodes from the run are plotted.
    journal : str, optional
        Name of the journal to obtain specified scale and height for figsize.
    format : str
        Format of plot. Default is ``"png"``.
    ylim : tuple of float
        Limits for y-axis in plot.
    yscale_log_base : int
        Base of logarithm to scale y-axis.
    """

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    qi_serial_sdc_methods = [qi for qi in QI_serial_methods if qi in QI_SERIAL]
    qi_all = qi_serial_sdc_methods + QI_parallel_methods

    used_nodes_all = set()

    for QI in qi_all:
        for sweeper_type in sweepers:
            key = f"{sweeper_type}_{QI}"
            if key not in all_stats:
                continue

            available_nodes = list(all_stats[key].keys())
            nodes = (
                sorted(available_nodes) if nodes_to_plot is None else [n for n in nodes_to_plot if n in all_stats[key]]
            )

            if not nodes:
                continue

            used_nodes_all.update(nodes)

            ys = []
            xs = []
            for n in nodes:
                # Skip entries that are not available (e.g. None for Andrews')
                stats = all_stats[key][n]
                try:
                    y = float(y_of(stats))
                except (TypeError, ValueError):
                    continue
                if y is None:
                    continue

                xs.append(n)
                ys.append(y)

            if not xs:
                continue

            label = get_method_label(sweeper_type, QI)
            ax.loglog(
                xs,
                ys,
                color=colors[key],
                marker=markers[key],
                label=label,
            )

    used_nodes_sorted = sorted(used_nodes_all)
    ax.tick_params(axis="both", which="minor", bottom=False, left=True)
    ax.set_xlabel(r"number of $\mathtt{MPI}$ ranks/nodes")

    if used_nodes_sorted:
        ax.set_xticks(used_nodes_sorted)
        ax.set_xticklabels(used_nodes_sorted)

    ax.set_xscale("log", base=2)
    if yscale_log_base is not None:
        ax.set_yscale("log", base=yscale_log_base)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(linewidth=0.5)
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    out = Path("data") / problem_name / f"{filename_stem}.{format}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_wallclocktime_vs_mpi_ranks(
    all_stats: dict[str, dict[int, dict[str, float]]],
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    serial_mode: str = "sdc",
    nodes_to_plot: list[int] = None,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    y_of: Callable[[Any], float] = lambda st: st.t_wall,
    **kwargs: Any,
) -> None:
    r"""
    Plots wallclock time versus number of MPI ranks. The wrapper function is used to do that.

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
    format : str
        Format of plot. Default is ``"png"``.
    """

    assert serial_mode == "sdc"  # Only SDC methods

    _plot_metric_vs_mpi_ranks(
        all_stats,
        problem_name=problem_name,
        sweepers=sweepers,
        QI_serial_methods=QI_serial_methods,
        QI_parallel_methods=QI_parallel_methods,
        nodes_to_plot=nodes_to_plot,
        journal=journal,
        format=format,
        y_of=y_of,
        ylabel="wall-clock time in s",
        filename_stem="walltime_vs_mpi_ranks_nodes",
        **kwargs,
    )


def plot_mean_iterations_vs_mpi_ranks(
    all_stats: dict[str, dict[int, dict[str, float]]],
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    serial_mode: str = "sdc",
    nodes_to_plot: list[int] = None,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    y_of: Callable[[Any], float] = lambda st: st.niter_mean,
    **kwargs: Any,
) -> None:
    r"""
    Plots the mean number of iterations versus number of MPI ranks. The wrapper function is used to do that.

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
    format : str
        Format of plot. Default is ``"png"``.
    """

    assert serial_mode == "sdc"  # Only SDC methods

    _plot_metric_vs_mpi_ranks(
        all_stats,
        problem_name=problem_name,
        sweepers=sweepers,
        QI_serial_methods=QI_serial_methods,
        QI_parallel_methods=QI_parallel_methods,
        nodes_to_plot=nodes_to_plot,
        journal=journal,
        format=format,
        y_of=y_of,
        ylabel="mean number of iterations",
        filename_stem="mean_iterations_vs_mpi_ranks_nodes",
        **kwargs,
    )


def plot_error_vs_mpi_ranks(
    all_stats: dict[str, dict[int, dict[str, float]]],
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    serial_mode: str = "sdc",
    nodes_to_plot: list[int] = None,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    **kwargs: Any,
) -> None:
    r"""
    Plots the error versus number of MPI ranks. The wrapper function is used to do that. For
    ``problem_name = "ANDREWS-SQUEEZER"`` the error of ``q``is computed at the end time,
    i.e., ``Tend = 0.03``, otherwise the global error across all unknowns is plotted.

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
    format : str
        Format of plot. Default is ``"png"``.
    """

    assert serial_mode == "sdc"  # Only SDC methods

    if problem_name == "ANDREWS-SQUEEZER":
        metric_key = "q_max_final_error"
        y_of = lambda st: st.qend_error
    else:
        metric_key = "all_max_global_error"
        y_of = lambda st: max(st.e_global_steps)

    ylabel = get_ylabel_based_on_metric(metric_key=metric_key)

    _plot_metric_vs_mpi_ranks(
        all_stats,
        problem_name=problem_name,
        sweepers=sweepers,
        QI_serial_methods=QI_serial_methods,
        QI_parallel_methods=QI_parallel_methods,
        nodes_to_plot=nodes_to_plot,
        journal=journal,
        format=format,
        y_of=y_of,
        ylabel=ylabel,
        filename_stem="error_vs_mpi_ranks_nodes",
        ylim=(1e-15, 1e1),
        **kwargs,
    )


def plot_embedded_error_vs_mpi_ranks(
    all_stats: dict[str, dict[int, dict[str, float]]],
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    serial_mode: str = "sdc",
    nodes_to_plot: list[int] = None,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    y_of: Callable[[Any], float] = lambda st: max(st.e_embedded_steps),
    **kwargs: Any,
) -> None:
    r"""
    Plots the maximum increment across all time steps versus number of MPI ranks.
    The wrapper function is used to do that.

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
    format : str
        Format of plot. Default is ``"png"``.
    """

    assert serial_mode == "sdc"  # Only SDC methods

    _plot_metric_vs_mpi_ranks(
        all_stats,
        problem_name=problem_name,
        sweepers=sweepers,
        QI_serial_methods=QI_serial_methods,
        QI_parallel_methods=QI_parallel_methods,
        nodes_to_plot=nodes_to_plot,
        journal=journal,
        format=format,
        y_of=y_of,
        ylabel="embedded error estimate",
        filename_stem="embedded_error_vs_mpi_ranks_nodes",
        ylim=(1e-15, 1e1),
        **kwargs,
    )


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

def _plot_time_to_accuracy(
    all_stats: dict[str, dict[int, Any]],
    *,
    problem_name: str,
    sweepers: list[str],
    QI_parallel_methods: list[str],
    filename_stem: str,
    num_nodes: int,
    metric_key: str,
    ref_QI: str = "LU",
    ref_num_nodes: int = 5,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    ylim: Optional[tuple[float, float]] = None,
    xlim: Optional[tuple[float, float]] = None,
    xscale_log_base: Optional[int] = 10,
    yscale_log_base: Optional[int] = 10,
    mark_every: int = 0,
) -> None:
    """
    Plot error/metric versus cumulative wallclock time for
    - the fixed serial reference method: ref_QI with M = ref_num_nodes
    - all parallel methods with M = num_nodes
    """

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)
    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # --------------------------------------------------
    # fixed serial reference: LU with M=5
    # --------------------------------------------------
    for sweeper_type in sweepers:
        sweeper_type_ref = set_correct_sweeper_type(sweeper_type, ref_QI)
        key_ref = f"{sweeper_type_ref}_{ref_QI}"

        if key_ref not in all_stats:
            continue
        if ref_num_nodes not in all_stats[key_ref]:
            continue

        stats_ref = all_stats[key_ref][ref_num_nodes]
        if stats_ref is None:
            continue

        times_ref, e_vals_ref = _get_sorted_filtered_times_and_errors(stats_ref)
        if len(times_ref) == 0:
            continue

        label_ref = f"{get_method_label(sweeper_type_ref, ref_QI)} ($M={ref_num_nodes}$)"
        ax.plot(
            times_ref,
            e_vals_ref,
            color=colors[key_ref],
            marker=markers[key_ref],
            markevery=(mark_every if mark_every else None),
            linewidth=1.0,
            label=label_ref,
        )

    # --------------------------------------------------
    # parallel methods with current num_nodes
    # --------------------------------------------------
    for QI in QI_parallel_methods:
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

            label = f"{get_method_label(sweeper_type, QI)} ($M={num_nodes}$)"
            ax.plot(
                times,
                e_vals,
                color=colors[key],
                marker=markers[key],
                markevery=(mark_every if mark_every else None),
                linewidth=1.0,
                label=label,
            )

    ax.set_xlabel(r"cumulative wallclock-time [s] (after each iteration $k$)")

    ylabel = get_ylabel_based_on_metric(metric_key=metric_key, type="iter")
    ax.set_ylabel(ylabel)

    if xscale_log_base is not None:
        ax.set_xscale("log", base=xscale_log_base)
    if yscale_log_base is not None:
        ax.set_yscale("log", base=yscale_log_base)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(linewidth=0.5)
    ax.tick_params(axis="both", which="minor", bottom=True, left=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    out = Path("data") / problem_name / f"{filename_stem}.{format}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_time_to_accuracy_for_each_num_nodes(
    all_stats: dict[str, dict[int, Any]],
    *,
    problem_name: str,
    factors: dict,
    factors_par_vs_par: dict,
    sweepers: list[str],
    QI_parallel_methods: list[str],
    filename_stem: str = "time_to_accuracy",
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    metric_key: str,
    nodes_to_plot: list[int] = None,
    ref_QI: str = "LU",
    ref_num_nodes: int = 5,
    ylim: Optional[tuple[float, float]] = None,
    xlim: Optional[tuple[float, float]] = None,
    xscale_log_base: Optional[int] = 10,
    yscale_log_base: Optional[int] = 10,
    mark_every: int = 0,
) -> None:
    """
    Create one time-to-accuracy plot per parallel M (= num_nodes),
    always compared against the fixed serial reference LU with M=5.
    """

    available_nodes = set()
    for sweeper_type in sweepers:
        for QI in QI_parallel_methods:
            key = f"{sweeper_type}_{QI}"
            if key in all_stats:
                available_nodes.update(all_stats[key].keys())

    nodes = sorted(available_nodes) if nodes_to_plot is None else [n for n in nodes_to_plot if n in available_nodes]

    for num_nodes in nodes:
        _plot_time_to_accuracy(
            all_stats,
            problem_name=problem_name,
            sweepers=sweepers,
            QI_parallel_methods=QI_parallel_methods,
            num_nodes=num_nodes,
            metric_key=metric_key,
            ref_QI=ref_QI,
            ref_num_nodes=ref_num_nodes,
            filename_stem=f"{filename_stem}_{num_nodes=}",
            journal=journal,
            format=format,
            ylim=ylim,
            xlim=xlim,
            xscale_log_base=xscale_log_base,
            yscale_log_base=yscale_log_base,
            mark_every=mark_every,
        )

    print_speedup_for_one_step(
        problem_name=problem_name,
        factors=factors,
        factors_par_vs_par=factors_par_vs_par,
        sweepers=sweepers,
        nodes_to_plot=nodes,
        ref_QI=ref_QI,
        QI_parallel_methods=QI_parallel_methods,
        ref_num_nodes=ref_num_nodes,
    )


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
    _, _, sweeper_labels = my_plot_style_config()

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


def plot_wallclocktime_vs_accuracy(
    all_stats: dict[str, dict[int, dict[str, float]]],
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    metric_key: str,
    nodes_to_plot: list[int] = None,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
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
    format : str
        Format of plot. Default is ``"png"``.
    """

    x_of = lambda st: st.t_wall
    if problem_name == "ANDREWS-SQUEEZER":
        metric_key = "q_max_final_error"
        y_of = lambda st: st.qend_error
    else:
        metric_key = "all_max_global_error"
        y_of = lambda st: max(st.e_global_steps)

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    qi_all = QI_serial_methods + QI_parallel_methods
    qi_all = [qi for qi in qi_all if qi in SDC_METHODS]# + [qi for qi in qi_all if qi not in SDC_METHODS]

    used_nodes_all = set()

    key_cache = []

    for QI in qi_all:
        for sweeper_type in sweepers:
            sweeper_type_eff = set_correct_sweeper_type(sweeper_type=sweeper_type, QI=QI)
            key = f"{sweeper_type_eff}_{QI}"

            if key not in all_stats:
                continue

            if key not in key_cache:
                # Adding key to cache to avoid double plotting
                key_cache.append(key)

                available_nodes = list(all_stats[key].keys())
                nodes = (
                    sorted(available_nodes) if nodes_to_plot is None else [n for n in nodes_to_plot if n in all_stats[key]]
                )

                if not nodes:
                    continue

                used_nodes_all.update(nodes)

                ys = []
                xs = []
                for n in nodes:
                    # Skip entries that are not available (e.g. None for Andrews')
                    stats = all_stats[key][n]
                    try:
                        x = x_of(stats)
                        y = y_of(stats)
                    except (TypeError, ValueError):
                        continue
                    if x is None or y is None:
                        continue

                    xs.append(x)
                    ys.append(y)

                if not xs:
                    continue

                label = get_method_label(sweeper_type, QI)
                ax.loglog(
                    xs,
                    ys,
                    color=colors[key],
                    marker=markers[key],
                    label=label,
                )

    used_nodes_sorted = sorted(used_nodes_all)
    ax.tick_params(axis="both", which="minor", bottom=True, left=False)
    ax.set_xlabel(r"wall-clock time in s (one run per $M$)")

    if used_nodes_sorted:
        ax.set_xticks(used_nodes_sorted)
        ax.set_xticklabels(used_nodes_sorted)

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)

    ax.grid(linewidth=0.5)

    ylabel = get_ylabel_based_on_metric(metric_key=metric_key)
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    filename_stem = "walltime_vs_accuracy"
    out = Path("data") / problem_name / f"{filename_stem}.{format}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_wallclocktime_over_time(
    all_stats: dict[str, dict[int, dict[str, float]]],
    dt: float,
    problem_name: str,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    nodes_to_plot: list[int] = None,
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    **kwargs: Any,
) -> None:
    
    _, Tend = choose_time_step_sizes(problem_name=problem_name)
    t = [i * dt for i in range(1, int(Tend / dt) + 1)]
    
    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)
    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    qi_serial_sdc_methods = [qi for qi in QI_serial_methods if qi in QI_SERIAL]
    qi_all = qi_serial_sdc_methods + QI_parallel_methods

    key = "constrainedDAE_LU"
    available_nodes = list(all_stats[key].keys())
    nodes = (
        sorted(available_nodes) if nodes_to_plot is None else [n for n in nodes_to_plot if n in all_stats[key]]
    )

    for num_nodes in nodes:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

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
                    stats.t_cpu_steps,
                    color=colors[key],
                    linewidth=1.0,
                    label=label,
                )

        ax.set_xlabel(r"time $t$")

        ax.set_ylabel("wallclock time in s")

        ax.set_yscale("log", base=10)

        ax.grid(linewidth=0.5)

        ax.tick_params(axis="both", which="minor", bottom=False, left=True)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

        out = Path("data") / problem_name / f"wallclocktimes_over_time_{num_nodes=}.{format}"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=400, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="scaling")
    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="scaling")

    nodes_to_plot = range(17)

    # filename = None
    # filename = "results_scaling_dt=0.05_linear_#2.pkl"
    filename = "results_scaling_dt=0.001_andrews_#3.pkl"
    # filename = "results_scaling_dt=0.05_reaction_diffusion_#2.pkl"

    plots_scaling(
        global_comm=global_comm, format="png", nodes_to_plot=nodes_to_plot, filename=filename, **config_andrews
    )
