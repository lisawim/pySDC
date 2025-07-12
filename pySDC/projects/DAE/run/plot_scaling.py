import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import LinearTestWorkPrecision, LinearTestOrderIteration, LinearTestScaling

from pySDC.projects.DAE.run.work_precision import run_all_simulations


def compute_speedups_and_efficiencies(config, results):
    speedups = {}
    efficiencies = {}

    for sweeper_type in config.sweepers:
        for QI_par in config.qDeltas_parallel:
            key_par = f"{sweeper_type}_{QI_par}"

            speedups[key_par] = []
            efficiencies[key_par] = []

    for sweeper_type in config.sweepers:
        key_ser = f"{sweeper_type}_{config.QI_ser}"

        if config.num_processes is None:
            global_size = int(max(val for val in results[key_par].keys()))
            config.set_num_processes(global_size)

        for QI_par in config.qDeltas_parallel:
            key_par = f"{sweeper_type}_{QI_par}"

            for num_nodes in config.num_processes:
                timings_ser = results[key_ser][num_nodes]
                timings_par = results[key_par][num_nodes]
                
                speedup = timings_ser / timings_par
                speedups[key_par].append(speedup)

                efficiency = speedup / num_nodes
                efficiencies[key_par].append(efficiency)

    return speedups, efficiencies


def plot_scaling(config):
    """Plots the speedup and the efficiency."""

    path = "data" + "/" + f"{config.problem_name}" + "/" + "results" + "/" + f"results_scaling.pkl"
    with open(path, "rb") as f:
        results = dill.load(f)

    speedups, efficiencies = compute_speedups_and_efficiencies(config, results)

    my_setup_mpl()
    colors, markers, sweeper_labels = my_plot_style_config()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for QI_par in config.qDeltas_parallel:
        for sweeper_type in config.sweepers:
            key_par = f"{sweeper_type}_{QI_par}"

            label = sweeper_labels[sweeper_type] + "-" + f"{QI_par}"
            axs[0].plot(
                config.num_processes,
                speedups[key_par],
                marker=markers[key_par],
                color=colors[key_par],
                label=label,
            )

            axs[1].plot(
                config.num_processes,
                efficiencies[key_par],
                marker=markers[key_par],
                color=colors[key_par],
            )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_xlabel("number of nodes/processes")
        ax.set_xticks(config.num_processes[::4])

    axs[0].set_ylabel("speedup")
    axs[1].set_ylabel("efficiency")

    axs[1].set_ylim((0.0, 1.0))

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.001), ncol=2)

    filename = "data" + "/" + f"{config.problem_name}" + "/" + f"scaling.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    config_scaling = LinearTestScaling()
    plot_scaling(config_scaling)