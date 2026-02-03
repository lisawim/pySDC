from mpi4py import MPI
import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Callable

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.utils import set_correct_sweeper_type

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.mpi_test import build_filename, run_mpi_test
from pySDC.projects.DAE.misc.methods_config import QI_SERIAL, RADAU_METHODS, RK_METHODS


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

                        timings_ser = all_stats[key_ser][num_nodes]["t_wall"]
                        timings_par = all_stats[key_par][num_nodes]["t_wall"]

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
    hook_class: list,
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

        # filename = build_filename(dt, problem_name)
        # path = "data" + "/" + f"{problem_name}" + "/" + "results" + "/" + filename
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

        plot_wallclocktime_vs_mpi_ranks(
            all_stats,
            problem_name,
            sweepers,
            QI_serial_methods,
            QI_parallel_methods,
            nodes_to_plot=nodes_to_plot,
        )

        plot_mean_iterations_vs_mpi_ranks(
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
    r"""Plots speedup and efficiency."""

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
            ax.set_xlabel(r"number of $\mathtt{MPI}$ ranks/nodes")

            ax.set_xticks(used_nodes_sorted)
            ax.set_xticklabels(used_nodes_sorted)

            ax.set_xscale("log", base=2)

            ax.grid(linewidth=0.5)

        axs[0].set_yscale("log", base=10)

        axs[0].set_ylabel("speedup")
        axs[1].set_ylabel("efficiency")
        # axs[1].set_ylim((0.0, 1.0))

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

        filename = "data" + "/" + f"{problem_name}" + "/" + f"scaling_{QI_ser}" + "." + format
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
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

    assert serial_mode == "sdc"  # Only SDC methods

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, 1, figsize=figsize)

    qi_serial_sdc_methods = [qi for qi in QI_serial_methods if qi in QI_SERIAL]
    qi_all = qi_serial_sdc_methods + QI_parallel_methods

    for QI in qi_all:
        for sweeper_type in sweepers:
            key = f"{sweeper_type}_{QI}"

            # collect x-values we actually used (for ticks)
            used_nodes_all = set()

            available_nodes = all_stats[key].keys()
            nodes = (
                list(available_nodes) if nodes_to_plot is None else [n for n in nodes_to_plot if n in available_nodes]
            )

            if len(nodes) == 0:
                continue

            used_nodes_all.update(nodes)

            t_wall = [all_stats[key][num_nodes]["t_wall"] for num_nodes in nodes]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            axs.loglog(
                nodes,
                t_wall,
                color=colors[key],
                marker=markers[key],
                label=label,
            )

    used_nodes_sorted = sorted(used_nodes_all)

    axs.tick_params(axis="both", which="minor", bottom=False, left=False)
    axs.set_xlabel(r"number of $\mathtt{MPI}$ ranks/nodes")

    axs.set_xticks(used_nodes_sorted)
    axs.set_xticklabels(used_nodes_sorted)

    axs.set_xscale("log", base=2)
    axs.set_yscale("log", base=10)

    axs.grid(linewidth=0.5)

    axs.set_ylabel("wall-clock time in s")

    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + "walltime_vs_mpi_ranks_nodes" + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


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

    assert serial_mode == "sdc"  # Only SDC methods

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, 1, figsize=figsize)

    qi_serial_sdc_methods = [qi for qi in QI_serial_methods if qi in QI_SERIAL]
    qi_all = qi_serial_sdc_methods + QI_parallel_methods

    for QI in qi_all:
        for sweeper_type in sweepers:
            key = f"{sweeper_type}_{QI}"

            # collect x-values we actually used (for ticks)
            used_nodes_all = set()

            available_nodes = all_stats[key].keys()
            nodes = (
                list(available_nodes) if nodes_to_plot is None else [n for n in nodes_to_plot if n in available_nodes]
            )

            if len(nodes) == 0:
                continue

            used_nodes_all.update(nodes)

            mean_iter = [all_stats[key][num_nodes]["niter_mean"] for num_nodes in nodes]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            axs.loglog(
                nodes,
                mean_iter,
                color=colors[key],
                marker=markers[key],
                label=label,
            )

    used_nodes_sorted = sorted(used_nodes_all)

    axs.tick_params(axis="both", which="minor", bottom=False, left=False)
    axs.set_xlabel(r"number of $\mathtt{MPI}$ ranks/nodes")

    axs.set_xticks(used_nodes_sorted)
    axs.set_xticklabels(used_nodes_sorted)

    axs.set_xscale("log", base=2)
    axs.set_yscale("log", base=10)

    axs.grid(linewidth=0.5)

    axs.set_ylabel("mean number of iterations")

    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + "mean_iterations_vs_mpi_ranks_nodes" + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="scaling")
    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="scaling")

    nodes_to_plot = None  # [2, 4, 8, 16, 32, 64]
    filename = "results_scaling_dt=0.05_linear.pkl"

    plots_scaling(
        global_comm=global_comm, format="png", nodes_to_plot=nodes_to_plot, filename=filename, **config_linear
    )
