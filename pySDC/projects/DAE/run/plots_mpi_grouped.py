from mpi4py import MPI
import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.utils import set_correct_sweeper_type

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.mpi_grouped_test import build_filename, run_mpi_grouped_breakeven_test
from pySDC.projects.DAE.misc.methods_config import QI_SERIAL, RADAU_METHODS, RK_METHODS


def plots_mpi_grouped(
    global_comm,
    num_nodes,
    problem_name,
    dt,
    sweepers,
    QI_serial_methods,
    QI_parallel_methods,
    filename=None,
    **kwargs,
):
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
            all_stats,
            num_nodes,
            problem_name,
            sweepers,
            QI_serial_methods,
            QI_parallel_methods,
            **kwargs,
        )


def plot_mpi_grouped_breakeven(
    all_stats,
    num_nodes,
    problem_name,
    sweepers,
    QI_parallel_methods,
    serial_mode="sdc",
    journal="Springer_Scientific_Computing",
    format="png",
    **kwargs,
):

    assert serial_mode == "sdc"  # Only SDC methods

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, 1, figsize=figsize)

    for QI in QI_parallel_methods:
        for sweeper_type in sweepers:
            key = f"{sweeper_type}_{QI}"

            ranks = all_stats[key].keys()

            t_wall = [all_stats[key][rank]["t_wall"] for rank in ranks]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            axs.loglog(
                ranks,
                t_wall,
                color=colors[key],
                marker=markers[key],
                label=label,
            )

    axs.tick_params(axis="both", which="minor", bottom=False, left=False)
    axs.set_xlabel(r"number of $\mathtt{MPI}$ ranks")

    axs.set_xscale("log", base=2)
    axs.set_yscale("log", base=10)

    axs.grid(linewidth=0.5)

    axs.set_ylabel("wall-clock time in s")

    handles, labels = axs.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"mpi_grouped_breakeven_{num_nodes=}" + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="scaling")
    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="scaling")

    num_nodes = global_comm.Get_size()
    plots_mpi_grouped(global_comm=global_comm, num_nodes=num_nodes, format="png", **config_reacdiff)
