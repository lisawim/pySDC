from mpi4py import MPI
import os
import re
import dill
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.misc.dataclasses import ScalingRunStats

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.mpi_grouped_test import build_filename, run_mpi_grouped_breakeven_test


def extract_dt_and_num_nodes_from_filename(filename: str) -> tuple[float, int]:
    m = re.search(r"dt=([0-9.eE+-]+)_P=(\d+)", filename)
    if m is None:
        raise ValueError(f"Cannot parse filename: {filename}")
    dt = float(m.group(1))
    num_nodes = int(m.group(2))
    return dt, num_nodes


def plots_mpi_grouped(
    global_comm: MPI.Comm,
    num_nodes: int,
    problem_name: str,
    dt: float,
    sweepers: list[str],
    QI_parallel_methods: list[str],
    filename: str = None,
    **kwargs: Any,
) -> None:
    """Generates scaling plots."""

    global_rank = global_comm.Get_rank()

    base_path = os.path.join("data", problem_name, "results")

    precomputed_files = {
        "ANDREWS-SQUEEZER": f"results_grouped_{dt=}_{num_nodes=}_andrews.pkl",
        "LINEAR-TEST": f"results_grouped_{dt=}_{num_nodes=}_linear.pkl",
        "REACTION-DIFFUSION": f"results_grouped_{dt=}_{num_nodes=}_reaction_diffusion.pkl",
    }

    if filename is not None:
        results_file = filename
        dt, num_nodes = extract_dt_and_num_nodes_from_filename(filename=filename)
    else:
        if problem_name in precomputed_files:
            print("Use precomputed results..\n")
            results_file = precomputed_files[problem_name]
            path = os.path.join(base_path, results_file)
            if not os.path.exists(path):
                run_mpi_grouped_breakeven_test(
                    global_comm, problem_name, dt, sweepers, QI_parallel_methods, num_nodes, **kwargs
                )
                results_file = build_filename(dt, problem_name, num_nodes)

        else:
            run_mpi_grouped_breakeven_test(
                global_comm, problem_name, dt, sweepers, QI_parallel_methods, num_nodes, **kwargs
            )
            results_file = build_filename(dt, problem_name, num_nodes)

    path = os.path.join(base_path, results_file)
    if global_rank == 0:
        with open(path, "rb") as f:
            all_stats = dill.load(f)

        plot_mpi_grouped_breakeven(
            all_stats=all_stats,
            num_nodes=num_nodes,
            problem_name=problem_name,
            dt=dt,
            sweepers=sweepers,
            QI_parallel_methods=QI_parallel_methods,
            **kwargs,
        )


def plot_mpi_grouped_breakeven(
    all_stats: dict[str, dict[int, ScalingRunStats]],
    num_nodes: int,
    problem_name: str,
    dt: float,
    sweepers: list[str],
    QI_parallel_methods: list[str],
    serial_mode: str = "sdc",
    journal: str = "Springer_Scientific_Computing",
    format: str = "png",
    **kwargs: Any,
) -> None:

    assert serial_mode == "sdc"  # Only SDC methods

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, 1, figsize=figsize)
    axs.set_title(
        rf"{problem_name} with $M={num_nodes}$ collocation nodes",
        pad=6,
    )

    # Vertical line with P = M
    axs.axvline(num_nodes, linestyle="--", color="black", linewidth=0.8)

    all_ranks = set()
    for QI in QI_parallel_methods:
        for sweeper_type in sweepers:
            key = f"{sweeper_type}_{QI}"
            all_ranks.update(list(all_stats[key].keys()))

            ranks = all_stats[key].keys()

            t_wall = [all_stats[key][rank].t_wall for rank in ranks]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            axs.loglog(
                ranks,
                t_wall,
                color=colors[key],
                marker=markers[key],
                label=label,
            )

    xticks = sorted(all_ranks)

    axs.set_xlabel(r"number of MPI ranks $P$")
    axs.set_ylabel(r"wall-clock time [s]")

    axs.set_xscale("log", base=2)
    axs.set_yscale("log", base=10)

    axs.set_xticks(xticks)
    axs.set_xticklabels([str(int(x)) for x in xticks])

    axs.grid(which="major", linewidth=0.5, alpha=0.6)
    axs.tick_params(axis="both", which="minor", bottom=False, left=False)

    # Text for vertical line at P = M (1 rank per collocation node)
    y = (axs.get_ylim()[1] + axs.get_ylim()[0]) / 2
    axs.text(
        num_nodes - 0.5,
        y,
        r" $P=M$",
        rotation=90,
        va="top",
        ha="left",
        fontsize=6,
    )

    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"mpi_grouped_breakeven_{num_nodes=}_{dt=}" + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD

    config_type = "breakeven"
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type=config_type)
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type=config_type)
    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type=config_type)

    num_nodes = global_comm.Get_size()
    # filename = 'results_grouped_dt=5.0e-02_P=64_linear.pkl'
    # filename = 'results_grouped_dt=1.0e-03_P=8_andrews.pkl'
    filename = 'results_grouped_dt=2.5e-02_P=64_reaction_diffusion.pkl'
    plots_mpi_grouped(global_comm=global_comm, num_nodes=num_nodes, format="png", filename=filename, **config_reacdiff)
