import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.plot_order_iteration import sync_ylim

from pySDC.projects.DAE.run.work_precision import run_all_simulations


def get_ylabel_based_on_metric(metric_key):
    if metric_key == "q_max_final_error":
        return "absolute error in q at end time"
    elif metric_key == "all_max_global_error":
        return "global error"


def plots_work_vs_error(
        hook_class, num_nodes, problem_name, sweepers, test_methods, metric_key="all_max_global_error", **kwargs
    ):
    """Generates plots for work vs error study."""

    base_path = os.path.join("data", problem_name, "results")
    precomputed_files = {
        "LINEAR-TEST": f"results_experiment_{num_nodes}_linear.pkl",
        "ANDREWS-SQUEEZER": f"results_experiment_{num_nodes}_andrews.pkl"
    }

    if problem_name in precomputed_files:
        filename = precomputed_files[problem_name]
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            run_all_simulations(hook_class, num_nodes, problem_name, sweepers, test_methods, **kwargs)
            path = os.path.join(base_path, f"results_experiment_{num_nodes}.pkl")
    else:
        run_all_simulations(hook_class, num_nodes, problem_name, sweepers, test_methods, **kwargs)
        path = os.path.join(base_path, f"results_experiment_{num_nodes}.pkl")

    with open(path, "rb") as f:
        all_stats = dill.load(f)

    plot_work_vs_error_single(all_stats, metric_key, problem_name, test_methods, **kwargs)

    plot_work_vs_error_sdc_radau(all_stats, metric_key, problem_name, sweepers, **kwargs)


def plot_work_vs_error_single(
        all_stats,
        metric_key,
        problem_name,
        test_methods,
        sweeper_type="constrainedDAE",
        journal="Springer_Scientific_Computing",
        format="eps",
    ):
    """Plots work vs error for one single SDC variant (default is SDC-C)."""

    plot_names = {"LINEAR-TEST": "Fig4", "ANDREWS-SQUEEZER": "Fig7"}

    figsize = figsize_by_journal(journal, scale=0.5, ratio=0.9)

    ylabel = get_ylabel_based_on_metric(metric_key)

    my_setup_mpl(fontsize=8)
    colors, markers, _ = my_plot_style_config()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for QI in [q for q in test_methods if not q.startswith("RadauIIA")]:
        key = f"{sweeper_type}_{QI}"
        stats = all_stats[key]

        wc_times = stats["wc_times"]
        metric_values = stats[metric_key]

        ax.loglog(
            wc_times,
            metric_values,
            marker=markers[key],
            markersize=6,
            markeredgewidth=1.0,
            color=colors[key],
            linewidth=1.5,
            label=f"{QI}",
        )

    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.set_xlabel("wall-clock time")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

    plot_name = plot_names[problem_name]
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_work_vs_error_sdc_radau(
        all_stats,
        metric_key,
        problem_name,
        sweepers,
        qDelta_best=["LU", "MIN-SR-NS"],
        sweeper_type_best=["constrainedDAE", "semiImplicitDAE"],
        radau_methods_plot=["RadauIIA5", "RadauIIA7"],
        journal="Springer_Scientific_Computing",
        format="eps",
    ):
    """Plots work vs error for all SDC-variants with best observed qDelta and Radau methods."""

    plot_names = {"LINEAR-TEST": "Fig5", "ANDREWS-SQUEEZER": "Fig8"}

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)

    ylabel = get_ylabel_based_on_metric(metric_key)

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for QI in qDelta_best:
        for sweeper_type in sweepers:
            key = f"{sweeper_type}_{QI}"
            stats = all_stats[key]

            wc_times = stats["wc_times"]
            metric_values = stats[metric_key]

            label = sweeper_labels[sweeper_type] + "-" + f"{QI}"
            axs[0].loglog(
                wc_times,
                metric_values,
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
                metric_values = stats[metric_key]

                label = sweeper_labels[sweeper_type] + "-" + f"{QI}" if QI in qDelta_best else f"{QI}"
                axs[1].loglog(
                    wc_times,
                    metric_values,
                    marker=markers[key],
                    color=colors[key],
                    label=label,
                )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_xlabel("wall-clock time")
        ax.set_ylabel(ylabel)

    axs = sync_ylim(axs, min_y_set=1e-15)

    handles0, labels0 = axs[0].get_legend_handles_labels()
    handles1, labels1 = axs[1].get_legend_handles_labels()
    handles = handles0 + handles1
    labels = labels0 + labels1

    unique = dict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    plot_name = plot_names[problem_name]
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="work_precision")
    plots_work_vs_error(metric_key="all_max_global_error", format="png", **config_linear)

    # config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="work_precision")
    # plots_work_vs_error(metric_key="q_max_final_error", format="png", **config_andrews)
