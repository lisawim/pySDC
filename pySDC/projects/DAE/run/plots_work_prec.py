import numpy as np
import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from typing import Any

from pySDC.core.hooks import Hooks
from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.plot_order_iteration import sync_xlim, sync_ylim
from pySDC.projects.DAE.misc.methods_config import RADAU_METHODS, RK_METHODS, SDC_METHODS

from pySDC.projects.DAE.run.work_precision import run_all_simulations


def get_method_label(sweeper_type: str, QI: str) -> str:
    _, _, sweeper_labels = my_plot_style_config()
    return sweeper_labels[sweeper_type] + "-" + f"{QI}" if QI in SDC_METHODS else f"{QI}"


def get_metric_key(problem_name: str) -> str:
    """
    Returns the key for the metric to plot based on the problem name.

    Parameters
    ----------
    problem_name : str
        Name of problem. Can be 'ANDREWS-SQUEEZER', 'LINEAR-TEST', or 'REACTION-DIFFUSION'.

    Returns
    -------
    metric_key : str
        Key for metric to plot.
    """

    if problem_name == "ANDREWS-SQUEEZER":
        return "q_max_final_error"
    elif problem_name in ["LINEAR-TEST", "REACTION-DIFFUSION"]:
        return "all_max_global_error"
    else:
        raise ValueError(f"Unknown problem name: {problem_name}")


def get_ylabel_based_on_metric(metric_key: str, type: str = "step") -> str:
    """
    Returns labels for y-axis indicating the correct kind of error.

    Parameters
    ----------
    metric_key : str

    Returns
    -------
    ylabel : str
        Label for plotting.
    """

    if metric_key == "q_max_final_error":
        if type == "step":
            return r"error $||q(T) - q^{\tilde{k}}_M||_{\infty}$"
        elif type == "iter":
            return r"error $||q(T) - q^{k}_M||_{\infty}$"

    elif metric_key == "all_max_global_error":
        if type == "step":
            return r"$L_\infty$ error"
        elif type == "iter":
            return r"$L_\infty$ error after iteration $k$"


def get_sorted_handles_and_labels(
    ax: Axes,
    label_order: list[str],
) -> tuple[tuple[str, ...], tuple[Artist, ...]]:
    """
    Sorts handles and labels for legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis of plot.
    label_order: sequence of str
        Includes label in its correct order.

    Returns
    -------
    labels_sorted : tuple of str
        Sorted labels.
    handles_sorted : tuple of str
        Sorted handles.
    """

    handles, labels = ax.get_legend_handles_labels()

    labels_handles_sorted = sorted(
        zip(labels, handles), key=lambda x: label_order.index(x[0]) if x[0] in label_order else 999
    )

    labels_sorted, handles_sorted = zip(*labels_handles_sorted)
    return labels_sorted, handles_sorted


def plots_work_vs_error(
    hook_class: list[Hooks],
    num_nodes: int,
    problem_name: str,
    sweepers: list[str],
    setup: str,
    nsweeps: int,
    test_methods: list[str],
    qDelta_best: list[str] = ["LU", "MIN-SR-NS", "MIN-SR-S"],
    include_dopri: bool = True,
    filename: str = None,
    **kwargs: Any,
) -> None:
    """
    Generates plots for work vs error study.

    Parameters
    ----------
    hook_class : list
        Contains hook classes for logging.
    num_nodes : int
        Number of collocation nodes.
    problem_name : str
        Name of problem. Can be 'ANDREWS-SQUEEZER', 'LINEAR-TEST', or 'REACTION-DIFFUSION'.
    sweepers : list
        Contains sweeper types.
    test_methods : list
        Contains methods to test, i.e., QIs for SDC and also the strings for RK-methods.
    metric_key : str, optional
        Indicates which kind of error is considered.
    qDelta_best : list, optional
        Best performing QIs in tests.
    include_dopri : bool, optional
        Indicates if half-explicit RK method using Dormand & Prince coefficients should be used.
    """

    base_path = os.path.join("data", problem_name, "results")
    precomputed_files = {
        "ANDREWS-SQUEEZER": f"results_experiment_{num_nodes}_andrews.pkl",
        "LINEAR-TEST": f"results_experiment_{num_nodes}_linear.pkl",
        "REACTION-DIFFUSION": f"results_experiment_{num_nodes}_reaction_diffusion.pkl",
    }

    if filename is not None:
        print("Use precomputed results..\n")

        results_file = filename
    else:
        if problem_name in precomputed_files:
            print("Use precomputed results.. \n")
            results_file = precomputed_files[problem_name]
            path = os.path.join(base_path, results_file)
            if not os.path.exists(path):
                run_all_simulations(hook_class, num_nodes, nsweeps, problem_name, sweepers, setup, test_methods, **kwargs)
                results_file = f"results_experiment_{num_nodes}_{nsweeps}.pkl"
        else:
            run_all_simulations(hook_class, num_nodes, nsweeps, problem_name, sweepers, setup, test_methods, **kwargs)
            results_file = f"results_experiment_{num_nodes}_{nsweeps}.pkl"

    path = os.path.join(base_path, results_file)
    with open(path, "rb") as f:
        all_stats = dill.load(f)

    metric_key = get_metric_key(problem_name)

    plot_work_vs_error_single(
        all_stats=all_stats,
        metric_key=metric_key,
        problem_name=problem_name,
        test_methods=test_methods,
        **kwargs,
    )

    plot_work_vs_error_sdc_radau(
        all_stats,
        metric_key,
        problem_name,
        sweepers,
        qDelta_best=qDelta_best,
        include_dopri=include_dopri,
        **kwargs,
    )


def plot_work_vs_error_single(
    all_stats: dict[str, dict[str, list[float]]],
    metric_key: str,
    problem_name: str,
    test_methods: list[str],
    sweeper_type: str = "constrainedDAE",
    journal: str = "Springer_Scientific_Computing",
) -> None:
    r"""
    Plots work versus error (kind of error is indicated by ``metric_key``) for one single SDC variant
    ``sweeper_type`` (default is SDC-C).

    Parameters
    ----------
    all_stats : dict
        Contains statistics from tests.
    metric_key : str
        Indicates kind of error to plot.
    problem_name : str
        Name of problem. Can be 'ANDREWS-SQUEEZER', 'LINEAR-TEST', or 'REACTION-DIFFUSION'.
    test_methods : list of str
        Contains methods to test, i.e., QIs for SDC and also the strings for RK-methods.
    sweeper_type : str
        Type of sweeper to plot results for. Default is 'constrainedDAE' (SDC-C).
    journal : str, optional
        Name of the journal to obtain specified scale and height for figsize.
    """

    sweeper_types = ["constrainedDAE", "semiImplicitDAE"]

    figsize = figsize_by_journal(journal, scale=0.85, ratio=0.47)

    ylabel = get_ylabel_based_on_metric(metric_key)

    label_order = []

    my_setup_mpl(fontsize=7.5)

    colors, markers, sweeper_labels = my_plot_style_config()
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for s, sweeper_type in enumerate(sweeper_types):
        axs[s].set_title(f"{sweeper_labels[sweeper_type]}")

        for QI in [q for q in test_methods if q not in RADAU_METHODS + RK_METHODS]:
            key_stat = f"{sweeper_type}_{QI}"
            stats = all_stats[key_stat]

            wc_times = stats["wc_times"]
            metric_values = stats[metric_key]

            label = f"{QI}"
            key_plot = f"constrainedDAE_{QI}"
            axs[s].loglog(
                wc_times,
                metric_values,
                marker=markers[key_plot],
                color=colors[key_plot],
                linewidth=1.1,
                markersize=3.5,
                markeredgewidth=0.6,
                label=label,
            )

            label_order.append(label)

    for ax in axs:
        ax.tick_params(axis="both", which="major", length=3.0)
        ax.tick_params(axis="both", which="minor", bottom=True, left=False, length=1.5)
        ax.set_xlabel("wall-clock time in s")
        ax.set_ylabel(ylabel)

        ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.5)
        ax.grid(axis="both", which="minor", linewidth=0.2, alpha=0.15)

    axs = sync_xlim(axs, min_x_set=1e-15)
    axs = sync_ylim(axs, min_y_set=1e-15)

    labels_sorted, handles_sorted = get_sorted_handles_and_labels(ax, label_order)

    fig.legend(handles_sorted, labels_sorted, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"work_vs_error_single.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_work_vs_error_sdc_radau(
    all_stats: dict[str, dict[str, list[float]]],
    metric_key: str,
    problem_name: str,
    sweepers: list[str],
    qDelta_best: list[str] = ["LU", "MIN-SR-NS", "MIN-SR-S", "MIN-SR-FLEX"],
    sweeper_type_best: list[str] = ["constrainedDAE", "semiImplicitDAE"],
    radau_methods_plot: list[str] = ["RadauIIA5", "RadauIIA7"],
    journal: str = "Springer_Scientific_Computing",
    include_dopri: bool = True,
) -> None:
    r"""
    Plots work vs error for all SDC-variants with best observed qDelta and Radau methods.

    Parameters
    ----------
    all_stats : dict
        Contains statistics (wallclock times and errors) from tests.
    metric_key : str
        Indicates kind of error to plot.
    problem_name : str
        Name of problem. Can be 'ANDREWS-SQUEEZER', 'LINEAR-TEST', or 'REACTION-DIFFUSION'.
    sweepers : list of str
        Contains sweeper types.
    qDelta_best : list, optional
        Best performing QIs in tests.
    sweeper_type_best : list, optional
        Best performing SDC sweepers in tests.
    radau_methods_plot : list of str
        Radau methods that are plotted.
    journal : str, optional
        Name of the journal to obtain specified scale and height for figsize.
    include_dopri : bool, optional
        Indicates if half-explicit RK method using Dormand & Prince coefficients should be used.
    """

    figsize = figsize_by_journal(journal, scale=0.45, ratio=0.6)

    ylabel = get_ylabel_based_on_metric(metric_key)

    label_order = []

    my_setup_mpl(fontsize=4)
    plt.rcParams['axes.linewidth'] = 0.45
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    key_cache = []

    qDelta_best_vs_radau_vs_rk = qDelta_best + radau_methods_plot
    if include_dopri:
        qDelta_best_vs_radau_vs_rk += ["DOPRI5"]

    for QI in qDelta_best_vs_radau_vs_rk:
        for sweeper_type in sweepers:
            if QI in radau_methods_plot:
                key = f"fullyImplicitDAE_{QI}"
            elif QI in ["DOPRI5"]:
                key = f"constrainedDAE_{QI}"
            else:
                key = f"{sweeper_type}_{QI}"

            if key not in key_cache:
                stats = all_stats[key]

                # Adding key to cache to avoid double plotting
                key_cache.append(key)

                wc_times = stats["wc_times"]
                metric_values = stats[metric_key]

                label = get_method_label(sweeper_type, QI)
                ax.loglog(
                    wc_times,
                    metric_values,
                    marker=markers[key],
                    color=colors[key],
                    linewidth=0.7,
                    markersize=2.0,
                    markeredgewidth=0.4,
                    label=label,
                )

                label_order.append(label)

    ax.tick_params(axis="both", which="major", length=2.5, width=0.4)
    ax.tick_params(axis="both", which="minor", bottom=True, left=False, length=1.5, width=0.4)

    ax.set_xlabel("wall-clock time in s")
    ax.set_ylabel(ylabel)

    ax.grid(axis="both", which="major", linewidth=0.35, alpha=0.5)
    ax.grid(axis="both", which="minor", linewidth=0.2, alpha=0.15)

    labels_sorted, handles_sorted = get_sorted_handles_and_labels(ax, label_order)

    fig.legend(handles_sorted, labels_sorted, loc="upper center", bbox_to_anchor=(0.55, 0.07), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + "work_vs_error_sdc_radau.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def default_keys_for_comparison(
    qDelta_best: list[str] = ["LU", "MIN-SR-NS"],
    sweepers: list[str] = ["constrainedDAE", "semiImplicitDAE"],
    radau_methods: list[str] = ["RadauIIA5", "RadauIIA7"],
    include_dopri: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Builds sdc keys and baselines.

    Parameters
    ----------
    qDelta_best : list, optional
        Best performing QIs in tests.
    sweepers : list of str
        Contains sweeper types.
    radau_methods_plot : list of str
        Radau methods that are plotted.
    include_dopri : bool, optional
        Indicates if half-explicit RK method using Dormand & Prince coefficients should be used.

    Returns
    -------
    sdc_keys : list of str
        Contains keys for SDC methods.
    baseline : list of str
        Contains keys for baseline methods, e.g. Radau and DOPRI5.
    """

    sdc_keys = [f"{sw}_{qi}" for sw in sweepers for qi in qDelta_best]
    baseline = [f"fullyImplicitDAE_{rm}" for rm in radau_methods]
    if include_dopri:
        baseline.append("constrainedDAE_DOPRI5")
    return sdc_keys, baseline


def compute_speedups_vs_error(
    all_stats: dict[str, dict[str, list[float]]],
    sdc_keys: list[str],
    baseline_keys: list[str],
    metric_key: str,
    time_key: str = "wc_times",
) -> dict[str, dict[str, Any]]:
    """
    Computes speedups for equal or better accuracy.
        speedup(e_j) = t_baseline(e_j) / min_{SDC}( t_SDC(e <= e_j) )

    For every baseline (that are Radau- and RK-methods) it will be:
      - iterated along all data points j,
      - taken the fastest data point with error <= error_baseline[j] for every SDC method,
      - build speedup values and other statistics based on the data.

    Returns a dictionary of the form:
        result[baseline_key] = {
            "records": [ {baseline, method, dt_baseline, err_baseline,
                        t_baseline, t_method_best, err_method_best, speedup}, ... ],
            "summary": [ {method, n, max, best_at_error, best_speedup}, ... ]
        }

    Parameters
    ----------
    all_stats : dict
        Contains statistics (wallclock times and errors) from tests.
    sdc_keys : list of str
        Contains keys for SDC methods.
    baseline : list of str
        Contains keys for baseline methods, e.g. Radau and DOPRI5.
    metric_key : str
        Indicates kind of error to plot.
    time_key : str, optional
        Denotes the key where wallclock times are stored in ``all_stats``.
    """

    result = {}

    for base in baseline_keys:
        if base not in all_stats:
            continue

        stats_base = all_stats[base]
        t_base = np.asarray(stats_base[time_key], dtype=float)
        err_base = np.asarray(stats_base[metric_key], dtype=float)
        # falls du dt pro Baseline mitloggst:
        dt_base = np.asarray(stats_base.get("dts", np.full_like(t_base, np.nan)), dtype=float)

        # Gültigkeitsmaske für Baseline
        mask_base = np.isfinite(t_base) & (t_base > 0.0) & np.isfinite(err_base) & (err_base > 0.0)

        per_baseline_records = []
        summaries = []

        for method in sdc_keys:
            if method not in all_stats:
                continue

            stats_m = all_stats[method]
            t_m = np.asarray(stats_m[time_key], dtype=float)
            err_m = np.asarray(stats_m[metric_key], dtype=float)

            # Gültigkeitsmaske für Methode
            mask_m_valid = np.isfinite(t_m) & (t_m > 0.0) & np.isfinite(err_m) & (err_m > 0.0)

            if not np.any(mask_m_valid):
                summaries.append(
                    {
                        "method": method,
                        "n": 0,
                        "max": None,
                        "best_at_error": None,
                        "best_speedup": None,
                    }
                )
                continue

            t_m_valid = t_m[mask_m_valid]
            err_m_valid = err_m[mask_m_valid]

            recs = []

            # über alle baseline-Punkte
            for dtb, tb, eb in zip(dt_base[mask_base], t_base[mask_base], err_base[mask_base]):
                mask_better_or_equal = err_m_valid <= eb
                if not np.any(mask_better_or_equal):
                    continue

                t_candidates = t_m_valid[mask_better_or_equal]
                err_candidates = err_m_valid[mask_better_or_equal]

                i_best = int(np.argmin(t_candidates))
                t_best = float(t_candidates[i_best])
                e_best = float(err_candidates[i_best])

                speedup = float(tb / t_best)

                recs.append(
                    {
                        "baseline": base,
                        "method": method,
                        "dt_baseline": float(dtb),
                        "err_baseline": float(eb),
                        "t_baseline": float(tb),
                        "t_method": t_best,
                        "err_method": e_best,
                        "speedup": speedup,
                    }
                )

            per_baseline_records.extend(recs)

            if len(recs) > 0:
                speeds = np.array([r["speedup"] for r in recs], dtype=float)
                err_vals = np.array([r["err_baseline"] for r in recs], dtype=float)
                i_best = int(np.argmax(speeds))

                summaries.append(
                    {
                        "method": method,
                        "n": int(len(speeds)),
                        "max": float(np.max(speeds)),
                        "best_at_error": float(err_vals[i_best]),
                        "best_speedup": float(speeds[i_best]),
                    }
                )
            else:
                summaries.append(
                    {
                        "method": method,
                        "n": 0,
                        "max": None,
                        "best_at_error": None,
                        "best_speedup": None,
                    }
                )

        result[base] = {"records": per_baseline_records, "summary": summaries}

    return result


def plot_speedup_vs_error(
    all_stats: dict[str, dict[str, list[float]]],
    problem_name: str,
    sdc_keys: list[str],
    baseline_keys: list[str],
    metric_key: str,
    journal: str = "Springer_Scientific_Computing",
) -> None:
    """
    Plots speedup(e) = t_baseline(e) / t_sdc(e) for all baseline keys,
    where e is the error of the baseline method.

    Parameters
    ----------
    all_stats : dict
        Contains statistics (wallclock times and errors) from tests.
    problem_name : str
        Name of problem. Can be 'ANDREWS-SQUEEZER', 'LINEAR-TEST', or 'REACTION-DIFFUSION'.
    sdc_keys : list of str
        Contains keys for SDC methods.
    baseline : list of str
        Contains keys for baseline methods, e.g. Radau and DOPRI5.
    metric_key : str
        Indicates kind of error to plot.
    journal : str, optional
        Name of the journal to obtain specified scale and height for figsize.
    """

    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)
    my_setup_mpl(fontsize=7)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, len(baseline_keys), figsize=figsize, sharey=True)
    if len(baseline_keys) == 1:
        axs = [axs]

    res = compute_speedups_vs_error(all_stats, sdc_keys, baseline_keys, metric_key)

    for ax, base in zip(axs, baseline_keys):
        if base not in res:
            continue
        records = res[base]["records"]
        if not records:
            continue

        grouped = {}
        for r in records:
            grouped.setdefault(r["method"], []).append(r)

        for method, recs in grouped.items():
            recs = sorted(recs, key=lambda d: d["err_baseline"])
            errs = [d["err_baseline"] for d in recs]
            speedups = [d["speedup"] for d in recs]

            if "_" in method:
                sweeper, QI = method.split("_", 1)
                label = sweeper_labels.get(sweeper, sweeper) + "-" + QI
            else:
                label = method

            key = method
            ax.loglog(
                errs,
                speedups,
                marker=markers.get(key, "o"),
                color=colors.get(key, None),
                label=label,
            )

            print_speedup_factors(base, errs, problem_name, speedups, label)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)

        xlabel = get_ylabel_based_on_metric(metric_key)
        ax.set_xlabel(xlabel)

        title = base.replace("fullyImplicitDAE_", "").replace("constrainedDAE_", "")
        ax.set_title(title)
        ax.grid(linewidth=0.5)
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)

    axs[0].set_ylabel("speedup vs baseline")

    axs = sync_xlim(axs, min_x_set=1e-15)

    handles_all, labels_all = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles_all += h
        labels_all += l
    unique = {}
    for h, l in zip(handles_all, labels_all):
        if l not in unique:
            unique[l] = h
    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=2)

    plot_names = {
        "LINEAR-TEST": "Fig_speedup_vs_error_linear",
        "ANDREWS-SQUEEZER": "Fig_speedup_vs_error_andrews",
        "REACTION-DIFFUSION": "Fig_speedup_vs_error_reacdiff",
    }
    plot_name = plot_names.get(problem_name, "Fig_speedup_vs_error")
    filename = f"data/{problem_name}/{plot_name}.png"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def print_speedup_factors(base, errs, problem_name, speedups, label):
    speedups = np.asarray(speedups)
    errs = np.asarray(errs)
    mask_better_one = speedups >= 1.0
    speedups_better_one = speedups[mask_better_one]

    if len(speedups_better_one) > 0:
        idx_min = np.argmin(speedups[mask_better_one])
        idx_max = np.argmax(speedups[mask_better_one])

        min_speedup = float(speedups_better_one[idx_min])
        max_speedup = float(speedups_better_one[idx_max])

        err_min = float(errs[mask_better_one][idx_min])
        err_max = float(errs[mask_better_one][idx_max])

        line = (
            f"[Baseline method: {base}] {label}: \n"
            f"SDC faster at {len(speedups_better_one)} points — \n"
            f"min = {min_speedup:.2f}x (with error={err_min:.2e}), \n"
            f"max = {max_speedup:.2f}x (with error={err_max:.2e})\n"
        )
    else:
        line = f"[Baseline method: {base}] {label}: No points, where SDC ist faster."

    print(line)


if __name__ == "__main__":
    """
    Generates plots for paper 'On the analysis of spectral deferred corrections for differential-algebraic
    equations of index one'.
    """

    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="work_precision")
    filename = "results_experiment_6_linear_thesis.pkl"

    plots_work_vs_error(filename=filename, journal="BUW_thesis", **config_linear)

    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="work_precision")
    filename = "results_experiment_6_andrews_thesis.pkl"

    plots_work_vs_error(filename=filename, journal="BUW_thesis", **config_andrews)

    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="work_precision")
    qDelta_best = ["LU", "MIN-SR-S"]
    include_dopri = False
    filename = "results_experiment_6_reaction_diffusion_thesis.pkl"

    plots_work_vs_error(
        include_dopri=include_dopri,
        qDelta_best=qDelta_best,
        filename=filename,
        journal="BUW_thesis",
        **config_reacdiff,
    )
