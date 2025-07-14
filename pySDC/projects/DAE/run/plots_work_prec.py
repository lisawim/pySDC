import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.misc.configurations import LinearTestWorkPrecision

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

    plot_work_vs_error_sdc_radau(all_stats, config)


def plot_work_vs_error_single(all_stats, config, sweeper_type="constrainedDAE", journal="Springer_Scientific_Computing"):
    """Plots work vs error for one single SDC variant (default is SDC-C)."""

    figsize = figsize_by_journal(journal, scale=0.6, ratio=0.9)

    my_setup_mpl(fontsize=10)
    colors, markers, _ = my_plot_style_config()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for QI in [q for q in config.test_methods if not q.startswith("RadauIIA")]:
        key = f"{sweeper_type}_{QI}"
        stats = all_stats[key]

        wc_times = stats["wc_times"]
        max_errors = stats["max_errors"]

        ax.loglog(
            wc_times,
            max_errors,
            marker=markers[key],
            markersize=6,
            markeredgewidth=1.0,
            color=colors[key],
            linewidth=1.5,
            label=f"{QI}",
        )

    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel("global error")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3)

    plot_name = "Fig3.eps"  # f"work_vs_error_single.eps"
    filename = "data" + "/" + f"{config.problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_work_vs_error_sdc_radau(
        all_stats,
        config,
        qDelta_best=["LU", "MIN-SR-NS"],
        sweeper_type_best=["constrainedDAE", "fullyImplicitDAE"],
        radau_methods_plot=["RadauIIA5", "RadauIIA7"],
        journal="Springer_Scientific_Computing",
    ):
    """Plots work vs error for all SDC-variants with best observed qDelta and Radau methods."""

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for QI in qDelta_best:
        for sweeper_type in config.sweepers:
            key = f"{sweeper_type}_{QI}"
            stats = all_stats[key]

            wc_times = stats["wc_times"]
            max_errors = stats["max_errors"]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            axs[0].loglog(
                wc_times,
                max_errors,
                marker=markers[key],
                color=colors[key],
                label=label,
            )

    key_cache = []

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
                axs[1].loglog(
                    wc_times,
                    max_errors,
                    marker=markers[key],
                    color=colors[key],
                    label=label,
                )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_xlabel("wall-clock time")
        ax.set_ylabel("global error")

    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[1].get_legend_handles_labels()
    handles = handles0 + handles1
    labels = labels0 + labels1

    unique = dict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    plot_name = "Fig4.eps"
    filename = "data" + "/" + f"{config.problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight", format="eps")
    plt.close(fig)


if __name__ == "__main__":
    config_work_prec = LinearTestWorkPrecision()
    plots_work_vs_error(config_work_prec)
