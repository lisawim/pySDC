from mpi4py import MPI
import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Callable, Optional

from pySDC.core.hooks import Hooks

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.utils import set_correct_sweeper_type

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.mpi_test import build_filename, run_mpi_test
from pySDC.projects.DAE.misc.methods_config import QI_SERIAL, RADAU_METHODS, RK_METHODS
from pySDC.projects.DAE.run.plots_work_prec import get_ylabel_based_on_metric


def compute_speedups_and_efficiencies(
    all_stats: dict[str, dict[int, dict[str, float]]],
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    **kwargs: Any,
) -> tuple[
    dict[str, dict[str, dict[int, float]]],
    dict[str, dict[str, dict[int, float]]],
    dict[str, list[int]],
]:
    speedups = {}
    efficiencies = {}
    num_processes_by_key_ser = {}

    for QI_ser in QI_serial_methods:
        for sweeper_type_ser in sweepers:
            sweeper_type_ser_eff = set_correct_sweeper_type(sweeper_type_ser, QI_ser)

            key_ser = f"{sweeper_type_ser_eff}_{QI_ser}"

            speedups.setdefault(key_ser, {})
            efficiencies.setdefault(key_ser, {})

            num_processes = sorted(all_stats[key_ser].keys())
            num_processes_by_key_ser[key_ser] = num_processes

            for sweeper_type_par in sweepers:
                for QI_par in QI_parallel_methods:
                    key_par = f"{sweeper_type_par}_{QI_par}"

                    speedups[key_ser].setdefault(key_par, {})
                    efficiencies[key_ser].setdefault(key_par, {})

                    for num_nodes in num_processes:
                        if num_nodes not in all_stats[key_par]:
                            continue

                        timings_ser = all_stats[key_ser][num_nodes].t_wall
                        timings_par = all_stats[key_par][num_nodes].t_wall

                        s = timings_ser / timings_par
                        speedups[key_ser][key_par][num_nodes] = s

                        e = s / num_nodes
                        efficiencies[key_ser][key_par][num_nodes] = e

    return speedups, efficiencies, num_processes_by_key_ser


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

        speedups, efficiencies, num_processes_by_key_ser = compute_speedups_and_efficiencies(
            all_stats, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs
        )

        for serial_mode in ["sdc", "radau_rk"]:
            plot_speedup_and_efficiency(
                problem_name,
                speedups,
                efficiencies,
                num_processes_by_key_ser,
                sweepers,
                QI_serial_methods,
                QI_parallel_methods,
                serial_mode=serial_mode,
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


def plot_speedup_and_efficiency(
    problem_name: str,
    speedups: dict[str, dict[str, dict[int, float]]],
    efficiencies: dict[str, dict[str, dict[int, float]]],
    num_processes_by_key_ser: dict[str, list[int]],
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

    serial_methods, make_serial_key = get_serial_methods_and_key_builder(
        serial_mode,
        QI_serial_methods,
    )

    for QI_ser in serial_methods:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # collect x-values we actually used (for ticks)
        used_nodes_all = set()

        for sweeper_type in sweepers:
            key_ser = make_serial_key(sweeper_type, QI_ser)

            if key_ser not in speedups:
                continue

            available_nodes = num_processes_by_key_ser.get(key_ser, [])
            nodes = (
                list(available_nodes) if nodes_to_plot is None else [n for n in nodes_to_plot if n in available_nodes]
            )

            if len(nodes) == 0:
                continue

            for QI_par in QI_parallel_methods:
                key_par = f"{sweeper_type}_{QI_par}"

                if key_par not in speedups[key_ser]:
                    continue

                s_map = speedups[key_ser][key_par]
                e_map = efficiencies[key_ser][key_par]

                xs = [n for n in nodes if n in s_map and n in e_map]

                used_nodes_all.update(xs)
                ys_s = [s_map[n] for n in xs]
                ys_e = [e_map[n] for n in xs]

                label = sweeper_labels[sweeper_type] + "-" + f"{QI_par}"
                axs[0].semilogx(
                    xs,
                    ys_s,
                    color=colors[key_par],
                    marker=markers[key_par],
                    label=label,
                )

                axs[1].semilogx(
                    xs,
                    ys_e,
                    color=colors[key_par],
                    marker=markers[key_par],
                )

        used_nodes_sorted = sorted(used_nodes_all)
        for ax in axs:
            ax.tick_params(axis="both", which="minor", bottom=False, left=False)
            ax.set_xlabel(r"number of $\mathtt{MPI}$ ranks")

            ax.set_xticks(used_nodes_sorted)
            ax.set_xticklabels(used_nodes_sorted)

            ax.set_xscale("log", base=2)

            ax.grid(linewidth=0.5)

        axs[0].set_yscale("log", base=10)

        axs[0].set_ylabel("speedup")
        axs[1].set_ylabel("efficiency")
        axs[1].set_ylim((0.0, 1.0))

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

        filename = "data" + "/" + f"{problem_name}" + "/" + f"scaling_{QI_ser}" + "." + format
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

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
    print(filename_stem)
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

            label = f"{sweeper_labels[sweeper_type]}-{QI}"
            print(label, ys, len(ys))
            ax.loglog(
                xs,
                ys,
                color=colors[key],
                marker=markers[key],
                label=label,
            )
    print()
    used_nodes_sorted = sorted(used_nodes_all)
    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
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
        y_of=lambda st: st.t_wall,
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
        y_of=lambda st: st.niter_mean,
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
        y_of = lambda st: st.qend_max_final_error
    else:
        metric_key = "all_max_global_error"
        y_of = lambda st: max(st.e_global_post_step)

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

    y_of = lambda st: max(st.e_emb_post_step)

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


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="scaling")
    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="scaling")

    nodes_to_plot = None  # [2, 4, 8, 16, 32, 64]
    # filename = None
    filename = "results_scaling_dt=0.05_linear.pkl"
    # filename = 'results_scaling_dt=0.001_andrews.pkl'
    # filename = 'results_scaling_dt=0.025_reaction_diffusion.pkl'

    plots_scaling(
        global_comm=global_comm, format="png", nodes_to_plot=nodes_to_plot, filename=filename, **config_linear
    )
