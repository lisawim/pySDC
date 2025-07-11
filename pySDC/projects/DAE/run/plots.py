import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import LinearTestWorkPrecision, LinearTestScaling

from pySDC.projects.DAE.run.work_precision import run_all_simulations


def plot_work_vs_error(config):
    path = "data" + "/" + f"{config.problem_name}" + "/" + "results" + "/" + f"results_experiment_{config.num_nodes}.pkl"
    if not os.path.isfile(path):
        run_all_simulations(config)

        with open(path, "rb") as f:
            all_stats = dill.load(f)
    else:
        with open(path, "rb") as f:
            all_stats = dill.load(f)

    plot_work_vs_error_single(all_stats, config)

    plot_work_vs_error_sdc_variants(all_stats, config)

    plot_work_vs_error_best_vs_radau(all_stats, config)


def plot_order_iteration(all_stats, config, sweeper_type="constrainedDAE"):
    """Plots order in each iteration for one single SDC variant."""

    my_setup_mpl()
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for QI in [q for q in config.test_methods if not q.startswith("RadauIIA")]:
        key = f"{sweeper_type}_{QI}"
        stats = all_stats[key]

        dt_list = stats["dt_list"]
        max_errors_y = stats["max_errors_y"]
        max_errors_z = stats["max_errors_z"]

        axs[0].loglog(dt_list, max_errors_y, marker=markers[key], color=colors[key], label=f"{QI}")
        axs[1].loglog(dt_list, max_errors_y, marker=markers[key], color=colors[key], label=f"{QI}")


def plot_work_vs_error_single(all_stats, config, sweeper_type="constrainedDAE"):
    """Plots work vs error for one single SDC variant (default is SDC-C)."""

    my_setup_mpl()
    colors, markers, _ = my_plot_style_config()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for QI in [q for q in config.test_methods if not q.startswith("RadauIIA")]:
        key = f"{sweeper_type}_{QI}"
        stats = all_stats[key]

        wc_times = stats["wc_times"]
        max_errors = stats["max_errors"]

        ax.loglog(wc_times, max_errors, marker=markers[key], color=colors[key], label=f"{QI}")

    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("global error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    filename = "data" + "/" + f"{config.problem_name}" + "/" + f"work_vs_error_single.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_work_vs_error_sdc_variants(
        all_stats, config, qDelta_best=["LU", "MIN-SR-NS"]
    ):
    """Plots work vs error for all SDC-variants with best observed qDelta (default is "LU" and "MIN-SR-NS")."""

    my_setup_mpl()
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for QI in qDelta_best:
        for sweeper_type in config.sweepers:
            key = f"{sweeper_type}_{QI}"
            stats = all_stats[key]

            wc_times = stats["wc_times"]
            max_errors = stats["max_errors"]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            ax.loglog(
                wc_times,
                max_errors,
                marker=markers[key],
                color=colors[key],
                label=label,
            )

    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("global error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    filename = "data" + "/" + f"{config.problem_name}" + "/" + f"work_vs_error_sdc_variants.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def plot_work_vs_error_best_vs_radau(
        all_stats,
        config,
        sweeper_type_best=["constrainedDAE", "fullyImplicitDAE"],
        qDelta_best=["LU", "MIN-SR-NS"],
        radau_methods_plot=["RadauIIA5", "RadauIIA7"],
):
    """Plots best SDC-variants against Radau solvers."""

    key_cache = []

    my_setup_mpl()
    colors, markers, sweeper_labels = my_plot_style_config()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    qDelta_best_vs_radau = qDelta_best + radau_methods_plot
    for QI in qDelta_best_vs_radau:
        for sweeper_type in sweeper_type_best:
            key = f"{sweeper_type}_{QI}" if QI in qDelta_best else f"fullyImplicitDAE_{QI}"

            if key not in key_cache:
                stats = all_stats[key]

                # Adding key to cache to avoid double plotting
                key_cache.append(key)

                wc_times = stats["wc_times"]
                max_errors = stats["max_errors"]

                label = sweeper_labels[sweeper_type] + "-" + f"{QI}" if QI in qDelta_best else f"{QI}"
                ax.loglog(
                    wc_times,
                    max_errors,
                    marker=markers[key],
                    color=colors[key],
                    label=label,
                )

    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("global error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    filename = "data" + "/" + f"{config.problem_name}" + "/" + f"work_vs_error_best_vs_radau.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

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
    config = LinearTestWorkPrecision()
    plot_work_vs_error(config)

    # config_scaling = LinearTestScaling()
    # plot_scaling(config_scaling)
