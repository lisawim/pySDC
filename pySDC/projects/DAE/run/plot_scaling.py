import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.mpi_test import build_filename, run_mpi_test
from pySDC.projects.DAE.misc.methods_config import RADAU_METHODS, RK_METHODS


def sweeper_for_serial(QI_ser, sweeper_type):
    if QI_ser in RK_METHODS:
        return "constrainedDAE"
    elif QI_ser in RADAU_METHODS:  # z.B. ["RadauIIA5", "RadauIIA7"]
        return "fullyImplicitDAE"
    else:
        # z.B. LU: hier verwenden wir den sweeper aus der äußeren Liste
        return sweeper_type


def compute_speedups_and_efficiencies(all_stats, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs):
    speedups = {}
    efficiencies = {}

    for QI_ser in QI_serial_methods:
        for sweeper_type_ser in sweepers:
            sweeper_type_ser_eff = sweeper_for_serial(QI_ser, sweeper_type_ser)

            key_ser = f"{sweeper_type_ser_eff}_{QI_ser}"

            speedups[key_ser] = {}
            efficiencies[key_ser] = {}

            num_processes = sorted(all_stats[key_ser].keys())

            for sweeper_type_par in sweepers:
                for QI_par in QI_parallel_methods:
                    key_par = f"{sweeper_type_par}_{QI_par}"

                    speedups[key_ser][key_par] = []
                    efficiencies[key_ser][key_par] = []

                    for num_nodes in num_processes:
                        timings_ser = all_stats[key_ser][num_nodes]
                        timings_par = all_stats[key_par][num_nodes]

                        s = timings_ser / timings_par
                        speedups[key_ser][key_par].append(s)

                        e = s / num_nodes
                        efficiencies[key_ser][key_par].append(e)

    return speedups, efficiencies, num_processes


def plots_scaling(problem_name, dt, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs):
    """Generates scaling plots."""

    base_path = os.path.join("data", problem_name, "results")
    precomputed_files = {
        "ANDREWS-SQUEEZER": f"results_scaling_{dt=}_andrews.pkl",
        "LINEAR-TEST": f"results_scaling_{dt=}_linear.pkl",
        "REACTION-DIFFUSION": f"results_scaling_{dt=}_reaction_diffusion.pkl",
    }

    if problem_name in precomputed_files:
        print("Use precomputed results.. \n")
        filename = precomputed_files[problem_name]
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            run_mpi_test(problem_name, dt, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs)
            filename = build_filename(dt, problem_name)
            path = os.path.join(base_path, filename)
    else:
        run_mpi_test(problem_name, dt, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs)
        build_filename(dt, problem_name)
        path = os.path.join(base_path, filename)

    with open(path, "rb") as f:
        all_stats = dill.load(f)

    filename = build_filename(dt, problem_name)
    path = "data" + "/" + f"{problem_name}" + "/" + "results" + "/" + filename
    with open(path, "rb") as f:
        all_stats = dill.load(f)

    speedups, efficiencies, num_processes = compute_speedups_and_efficiencies(
        all_stats, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs
    )

    plot_speedup_and_efficiency(
        problem_name,
        speedups,
        efficiencies,
        num_processes,
        sweepers,
        QI_parallel_methods,
        QI_ser="RadauIIA7",
        **kwargs,
    )

def plot_speedup_and_efficiency(
    problem_name,
    speedups,
    efficiencies,
    num_processes,
    sweepers,
    QI_parallel_methods,
    QI_ser="RadauIIA7",
    journal="Springer_Scientific_Computing",
    format="eps",
    **kwargs,
):
    r"""
    Plots speedup and efficiency for one single compared with one single serial method ``QI_ser``
    (default is ``"RadauIIA7"``).
    """

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    sweeper_type_eff = sweeper_for_serial(QI_ser, "fullyImplicitDAE")
    key_ser = f"{sweeper_type_eff}_{QI_ser}"

    for sweeper_type in sweepers:
        if key_ser not in speedups:
            continue

        for QI_par in QI_parallel_methods:
            key_par = f"{sweeper_type}_{QI_par}"

            if key_par not in speedups[key_ser]:
                continue

            s_vals = speedups[key_ser][key_par]
            e_vals = efficiencies[key_ser][key_par]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI_par}"
            axs[0].plot(
                num_processes,
                s_vals,
                marker=markers[key_par],
                color=colors[key_par],
                label=label,
            )

            axs[1].plot(
                num_processes,
                e_vals,
                marker=markers[key_par],
                color=colors[key_par],
            )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_xlabel("number of nodes/processes")
        ax.set_xticks(num_processes[::4])
        ax.grid(linewidth=0.5)

    axs[0].set_ylabel("speedup")
    axs[1].set_ylabel("efficiency")

    axs[1].set_ylim((0.0, 1.0))

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.001), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"scaling_{QI_ser}" + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    plots_scaling(format="png", **config_linear)