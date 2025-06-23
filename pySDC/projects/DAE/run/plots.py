import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import QDELTAS, ExperimentConfig, my_setup_mpl

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


def plot_work_vs_error_single(all_stats, qDelta_list, problem_name, sweeper_type="constrainedDAE"):
    """Plots work vs error for one single SDC variant (default is SDC-C)."""

    colors = {
        "IE": "gold",
        "LU": "orange",
        "MIN-SR-NS": "firebrick",
        "MIN-SR-S": "purple",
        "Picard": "dodgerblue",
    }
    markers = {"IE": "o", "LU": "s", "MIN-SR-NS": "^", "MIN-SR-S": "d", "Picard": "*"}

    qDelta_sdc = [QI for QI in qDelta_list if QI in QDELTAS]

    my_setup_mpl()
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for QI in qDelta_sdc:
        key = f"{sweeper_type}_{QI}"
        stats = all_stats[key]

        wc_times = stats["wc_times"]
        max_errors = stats["max_errors"]

        ax.loglog(wc_times, max_errors, marker=markers[QI], color=colors[QI], label=f"{QI}")

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

    colors = {
        "constrainedDAE": {
            "IE": "gold",
            "LU": "orange",
            "MIN-SR-NS": "firebrick",
            "MIN-SR-S": "purple",
            "Picard": "dodgerblue",
        },
        "embeddedDAE": {
            "IE": "royalblue",
            "LU": "green",
            "MIN-SR-NS": "plum",
            "MIN-SR-S": "coral",
            "Picard": "darkcyan",
        },
        "fullyImplicitDAE": {
            "IE": "limegreen",
            "LU": "darkturquoise",
            "MIN-SR-NS": "slategrey",
            "MIN-SR-S": "pink",
            "Picard": "sandybrown",
        },
        "semiImplicitDAE": {
            "IE": "yellow",
            "LU": "darkmagenta",
            "MIN-SR-NS": "mediumseagreen",
            "MIN-SR-S": "khaki",
            "Picard": "red",
        },
    }

    markers = {
        "constrainedDAE": {"IE": "o", "LU": "s", "MIN-SR-NS": "^", "MIN-SR-S": "d", "Picard": "*"},
        "embeddedDAE": {"IE": "D", "LU": "<", "MIN-SR-NS": "H", "MIN-SR-S": "o", "Picard": "v"},
        "fullyImplicitDAE": {"IE": "s", "LU": "p", "MIN-SR-NS": "X", "MIN-SR-S": "*", "Picard": "<"},
        "semiImplicitDAE": {"IE": "d", "LU": "8", "MIN-SR-NS": "s", "MIN-SR-S": "^", "Picard": "D"},
    }

    sweeper_labels = {
        "constrainedDAE": "SDC-C",
        "embeddedDAE": "SDC-E",
        "fullyImplicitDAE": "FI-SDC",
        "semiImplicitDAE": "SI-SDC",
    }

    my_setup_mpl()
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
                marker=markers[sweeper_type][QI],
                color=colors[sweeper_type][QI],
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

# def plot_work_vs_error_best_vs_radau(all_stats):




if __name__ == '__main__':
    first_problem = "LINEAR-TEST"

    plot_work_vs_error(problem_name=first_problem)
