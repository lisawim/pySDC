import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import LinearTestWorkPrecision, LinearTestOrderIteration

from pySDC.projects.DAE.run.work_precision import run_all_simulations


def plots_work_vs_error(config):
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

    path = "data" + "/" + f"{config.problem_name}" + "/" + "results" + "/" + f"results_experiment_{config.num_nodes}.pkl"
    run_all_simulations(config)

    with open(path, "rb") as f:
        all_stats = dill.load(f)

    my_setup_mpl()
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for QI in [q for q in config.test_methods if not q.startswith("RadauIIA")]:
        key = f"{sweeper_type}_{QI}"
        stats = all_stats[key]

        dt_list = stats["dt_list"]
        errors_y_iter = stats["errors_y_iter"]
        errors_z_iter = stats["errors_z_iter"]

        axs[0].loglog(dt_list, errors_y_iter, marker=markers[key], color=colors[key], label=f"{QI}")
        axs[1].loglog(dt_list, errors_z_iter, marker=markers[key], color=colors[key], label=f"{QI}")


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


if __name__ == "__main__":
    config_work_prec = LinearTestWorkPrecision()
    plots_work_vs_error(config_work_prec)

    config_order_iter = LinearTestOrderIteration()
