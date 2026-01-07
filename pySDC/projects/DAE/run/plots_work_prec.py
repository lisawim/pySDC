import numpy as np
import os
import dill
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.plotting.plot_svd import sync_xlim
from pySDC.projects.DAE.misc.methods_config import RADAU_METHODS, RK_METHODS

from pySDC.projects.DAE.run.work_precision import run_all_simulations


def get_ylabel_based_on_metric(metric_key):
    if metric_key == "q_max_final_error":
        return r"error $||q(T) - q^{\tilde{k}}_M||_{\infty}$"
    elif metric_key == "all_max_global_error":
        return r"$L_\infty$ error"

def get_sorted_handles_and_labels(ax, label_order):
    """Sorts handles and labels for legend."""

    handles, labels = ax.get_legend_handles_labels()

    labels_handles_sorted = sorted(
        zip(labels, handles),
        key=lambda x: label_order.index(x[0]) if x[0] in label_order else 999
    )

    labels_sorted, handles_sorted = zip(*labels_handles_sorted)
    return labels_sorted, handles_sorted


def plots_work_vs_error(
        hook_class,
        num_nodes,
        problem_name,
        sweepers,
        test_methods,
        metric_key="all_max_global_error",
        qDelta_best=["LU", "MIN-SR-NS"],
        include_dopri=True,
        **kwargs,
    ):
    """Generates plots for work vs error study."""

    base_path = os.path.join("data", problem_name, "results")
    precomputed_files = {
        "ANDREWS-SQUEEZER": f"results_experiment_{num_nodes}_andrews.pkl",
        "LINEAR-TEST": f"results_experiment_{num_nodes}_linear.pkl",
        "REACTION-DIFFUSION": f"results_experiment_{num_nodes}_reaction_diffusion.pkl",
    }

    if problem_name in precomputed_files:
        print("Use precomputed results.. \n")
        filename = precomputed_files[problem_name]
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            run_all_simulations(
                hook_class, num_nodes, problem_name, sweepers, test_methods, **kwargs
            )
            path = os.path.join(base_path, f"results_experiment_{num_nodes}.pkl")
    else:
        run_all_simulations(
            hook_class, num_nodes, problem_name, sweepers, test_methods, **kwargs
        )
        path = os.path.join(base_path, f"results_experiment_{num_nodes}.pkl")

    with open(path, "rb") as f:
        all_stats = dill.load(f)

    plot_work_vs_error_single(all_stats, metric_key, problem_name, test_methods, **kwargs)

    plot_work_vs_error_sdc_radau(
        all_stats,
        metric_key,
        problem_name,
        sweepers,
        qDelta_best=qDelta_best,
        include_dopri=include_dopri,
        **kwargs,
    )

    # sdc_keys, baseline_keys = default_keys_for_comparison()

    # plot_speedup_vs_error(all_stats, problem_name, sdc_keys, baseline_keys, metric_key, **kwargs)


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

    plot_names = {"LINEAR-TEST": "Fig4", "ANDREWS-SQUEEZER": "Fig8", "REACTION-DIFFUSION": "Fig11"}

    figsize = figsize_by_journal(journal, scale=0.55, ratio=0.7)

    ylabel = get_ylabel_based_on_metric(metric_key)

    label_order = []

    my_setup_mpl(fontsize=8)

    colors, markers, _ = my_plot_style_config()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for QI in [q for q in test_methods if q not in RADAU_METHODS + RK_METHODS]:
        key = f"{sweeper_type}_{QI}"
        stats = all_stats[key]

        wc_times = stats["wc_times"]
        metric_values = stats[metric_key]

        label = f"{QI}"
        ax.loglog(
            wc_times,
            metric_values,
            marker=markers[key],
            color=colors[key],
            label=label,
        )

        label_order.append(label)

    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.set_xlabel("wall-clock time in s")
    ax.set_ylabel(ylabel)
    ax.grid(linewidth=0.5)

    labels_sorted, handles_sorted = get_sorted_handles_and_labels(ax, label_order)

    fig.legend(handles_sorted, labels_sorted, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3)

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
        include_dopri=True,
    ):
    """Plots work vs error for all SDC-variants with best observed qDelta and Radau methods."""

    plot_names = {"LINEAR-TEST": "Fig5", "ANDREWS-SQUEEZER": "Fig9", "REACTION-DIFFUSION": "Fig12"}

    figsize = figsize_by_journal(journal, scale=0.55, ratio=0.7)

    ylabel = get_ylabel_based_on_metric(metric_key)

    label_order = []

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(1, 1, figsize=figsize)

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

                label = sweeper_labels[sweeper_type] + "-" + f"{QI}" if QI in qDelta_best else f"{QI}"
                axs.loglog(
                    wc_times,
                    metric_values,
                    marker=markers[key],
                    color=colors[key],
                    label=label,
                )

                label_order.append(label)

    axs.tick_params(axis="both", which="minor", bottom=False, left=False)
    axs.set_xlabel("wall-clock time in s")
    axs.set_ylabel(ylabel)
    axs.grid(linewidth=0.5)

    labels_sorted, handles_sorted = get_sorted_handles_and_labels(axs, label_order)

    fig.legend(handles_sorted, labels_sorted, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3)

    plot_name = plot_names[problem_name]
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + "." + format
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def default_keys_for_comparison(
    qDelta_best=("LU", "MIN-SR-NS"),
    sweepers=("constrainedDAE", "semiImplicitDAE"),
    radau_methods=("RadauIIA5", "RadauIIA7"),
    include_dopri=True,
):
    """
    Baut SDC-Keys und Baselines, analog zu deiner Plot-Funktion.
    """
    sdc_keys = [f"{sw}_{qi}" for sw in sweepers for qi in qDelta_best]
    baseline = [f"fullyImplicitDAE_{rm}" for rm in radau_methods]
    if include_dopri:
        baseline.append("constrainedDAE_DOPRI5")
    return sdc_keys, baseline

def compute_speedups_vs_error(
    all_stats,
    sdc_keys,
    baseline_keys,
    metric_key,
    time_key="wc_times",
):
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
        mask_base = (
            np.isfinite(t_base) & (t_base > 0.0) &
            np.isfinite(err_base) & (err_base > 0.0)
        )

        per_baseline_records = []
        summaries = []

        for method in sdc_keys:
            if method not in all_stats:
                continue

            stats_m = all_stats[method]
            t_m = np.asarray(stats_m[time_key], dtype=float)
            err_m = np.asarray(stats_m[metric_key], dtype=float)

            # Gültigkeitsmaske für Methode
            mask_m_valid = (
                np.isfinite(t_m) & (t_m > 0.0) &
                np.isfinite(err_m) & (err_m > 0.0)
            )

            if not np.any(mask_m_valid):
                summaries.append({
                    "method": method,
                    "n": 0,
                    "median": None, "q25": None, "q75": None, "max": None,
                    "best_at_error": None, "best_speedup": None,
                })
                continue

            t_m_valid = t_m[mask_m_valid]
            err_m_valid = err_m[mask_m_valid]

            recs = []

            # über alle baseline-Punkte
            for dtb, tb, eb in zip(dt_base[mask_base], t_base[mask_base], err_base[mask_base]):
                # SDC-Punkte, die mindestens so genau sind wie baseline (Fehler <= eb)
                mask_better_or_equal = err_m_valid <= eb
                if not np.any(mask_better_or_equal):
                    # keine SDC-Lösung mit dieser oder besserer Genauigkeit
                    continue

                t_candidates = t_m_valid[mask_better_or_equal]
                err_candidates = err_m_valid[mask_better_or_equal]

                i_best = int(np.argmin(t_candidates))
                t_best = float(t_candidates[i_best])
                e_best = float(err_candidates[i_best])

                speedup = float(tb / t_best)

                recs.append({
                    "baseline": base,
                    "method": method,
                    "dt_baseline": float(dtb),
                    "err_baseline": float(eb),
                    "t_baseline": float(tb),
                    "t_method": t_best,
                    "err_method": e_best,
                    "speedup": speedup,
                })

            per_baseline_records.extend(recs)

            if len(recs) > 0:
                speeds = np.array([r["speedup"] for r in recs], dtype=float)
                err_vals = np.array([r["err_baseline"] for r in recs], dtype=float)
                i_best = int(np.argmax(speeds))

                summaries.append({
                    "method": method,
                    "n": int(len(speeds)),
                    "max": float(np.max(speeds)),
                    "best_at_error": float(err_vals[i_best]),
                    "best_speedup": float(speeds[i_best]),
                })
            else:
                summaries.append({
                    "method": method,
                    "n": 0,
                    "max": None,
                    "best_at_error": None, "best_speedup": None,
                })

        result[base] = {"records": per_baseline_records, "summary": summaries}

    return result

def plot_speedup_vs_error(
    all_stats,
    problem_name,
    sdc_keys,
    baseline_keys,
    metric_key,
    journal="Springer_Scientific_Computing",
    format="eps",
):
    """
    Plots speedup(e) = t_baseline(e) / t_sdc(e) for all baseline keys,
    where e is the error of the baseline method.
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

        # gruppieren nach methode
        grouped = {}
        for r in records:
            grouped.setdefault(r["method"], []).append(r)

        for method, recs in grouped.items():
            # Für schöne Linien nach Fehler sortieren (x-Achse: Fehler der Baseline)
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
                errs, speedups,
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

    # Legende (einzigartige Labels)
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
    filename = f"data/{problem_name}/{plot_name}.{format}"
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
        line = (f"[Baseline method: {base}] {label}: No points, where SDC ist faster.")

    print(line)
