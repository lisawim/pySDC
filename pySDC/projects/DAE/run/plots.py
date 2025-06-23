import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import QDELTAS, ExperimentConfig, my_setup_mpl, my_plot_style_config

from pySDC.projects.DAE.run.work_precision import run_all_simulations


def plot_work_vs_error(problem_name, config=ExperimentConfig):
    path = "data" + "/" + f"{problem_name}" + "/" + "results" + "/" + f"results_experiment_{config.num_nodes}.pkl"
    if not os.path.isfile(path):
        run_all_simulations(problem_name=first_problem)

        with open(path, "rb") as f:
            all_stats = dill.load(f)
    else:
        with open(path, "rb") as f:
            all_stats = dill.load(f)

    plot_work_vs_error_single(all_stats, config.qDelta_list, problem_name)

    plot_work_vs_error_sdc_variants(all_stats, config.sweeper_type_list, problem_name)

    plot_work_vs_error_best_vs_radau(all_stats, config, problem_name)


def plot_work_vs_error_single(all_stats, qDelta_list, problem_name, sweeper_type="constrainedDAE"):
    """Plots work vs error for one single SDC variant (default is SDC-C)."""

    qDelta_sdc = [QI for QI in qDelta_list if QI in QDELTAS]

    my_setup_mpl()
    colors, markers, _ = my_plot_style_config()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for QI in qDelta_sdc:
        key = f"{sweeper_type}_{QI}"
        stats = all_stats[key]

        wc_times = stats["wc_times"]
        max_errors = stats["max_errors"]

        ax.loglog(wc_times, max_errors, marker=markers[key], color=colors[key], label=f"{QI}")

    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("global error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"work_vs_error_single.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_work_vs_error_sdc_variants(
        all_stats, sweeper_type_list, problem_name, qDelta_best=["LU", "MIN-SR-NS"]
    ):
    """Plots work vs error for all SDC-variants with best observed qDelta (default is "LU" and "MIN-SR-NS")."""

    my_setup_mpl()
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for QI in qDelta_best:
        for sweeper_type in sweeper_type_list:
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

    filename = "data" + "/" + f"{problem_name}" + "/" + f"work_vs_error_sdc_variants.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def plot_work_vs_error_best_vs_radau(
        all_stats,
        config,
        problem_name,
        sweeper_type_best=["constrainedDAE", "fullyImplicitDAE"],
        qDelta_best=["LU", "MIN-SR-NS"],
):
    """Plots best SDC-variants against Radau solvers."""

    key_cache = []

    my_setup_mpl()
    colors, markers, sweeper_labels = my_plot_style_config()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    qDelta_best_vs_radau = [q for q in config.qDelta_list if q in qDelta_best or q.startswith("RadauIIA")]
    for QI in qDelta_best_vs_radau:
        for sweeper_type in sweeper_type_best:
            key = f"{sweeper_type}_{QI}" if QI in qDelta_best else f"fullyImplicitDAE_{QI}"

            if key not in key_cache:
                stats = all_stats[key]

                # Adding key to cache
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

    filename = "data" + "/" + f"{problem_name}" + "/" + f"work_vs_error_best_vs_radau.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    first_problem = "LINEAR-TEST"

    plot_work_vs_error(problem_name=first_problem)
