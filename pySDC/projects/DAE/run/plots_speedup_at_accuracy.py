from mpi4py import MPI
import os
import dill
import matplotlib.pyplot as plt
from typing import Any

from pySDC.core.hooks import Hooks

from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.utils import set_correct_sweeper_type

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.speedup_at_accuracy_test import build_filename, run_speedup_at_accuracy_test
from pySDC.projects.DAE.run.plots_scaling import save_fig, set_nodes_for_plotting
from pySDC.projects.DAE.run.plots_work_prec import get_method_label


def compute_speedups(
    all_stats: dict[str, dict[int, dict[str, float]]],
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    ref_QI: str = "IE",
    **kwargs: Any,
) -> tuple[
    dict[str, dict[str, dict[int, float]]],
    dict[str, dict[str, dict[int, float]]],
    dict[str, list[int]],
]:

    assert ref_QI in QI_serial_methods, f"Reference QI {ref_QI} must be in QI_serial_methods."

    speedups = {}
    efficiencies = {}
    num_processes_by_ref = {}

    for sweeper_type_ser in sweepers:
        sweeper_type_ser_eff = set_correct_sweeper_type(sweeper_type_ser, ref_QI)
        key_ref = f"{sweeper_type_ser_eff}_{ref_QI}"

        speedups.setdefault(key_ref, {})
        efficiencies.setdefault(key_ref, {})

        available_parallel_nodes = set()

        for sweeper_type_par in sweepers:
            for QI_par in QI_parallel_methods:
                key_par = f"{sweeper_type_par}_{QI_par}"

                if key_par not in all_stats:
                    continue

                speedups[key_ref].setdefault(key_par, {})
                efficiencies[key_ref].setdefault(key_par, {})

                for num_nodes, stats_par in sorted(all_stats[key_par].items()):
                    t_ref = all_stats[key_ref][num_nodes].t_wall_stop_at_acc

                    t_par = stats_par.t_wall_stop_at_acc
                    if t_par is None:
                        continue

                    s = t_ref / t_par
                    speedups[key_ref][key_par][num_nodes] = s

                    e = s / num_nodes
                    efficiencies[key_ref][key_par][num_nodes] = e

                    available_parallel_nodes.add(num_nodes)

        num_processes_by_ref[key_ref] = sorted(available_parallel_nodes)

    return speedups, efficiencies, num_processes_by_ref


def split_method_key(key: str) -> tuple[str, str]:
    return key.rsplit("_", 1)


def get_sweeper_type_from_key(key: str) -> str:
    sweeper_type, _ = split_method_key(key)
    return sweeper_type


def plots_speedup_at_accuracy(
    global_comm: MPI.Comm,
    hook_class: list[Hooks],
    problem_name: str,
    dt: float,
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    stop_at_accuracy_for_speedup: bool,
    ref_QI: str = "IE",
    nodes_to_plot: list[int] = range(2, 9),
    filename: str = None,
    **kwargs: Any,
) -> None:
    """Generates scaling plots."""

    global_rank = global_comm.Get_rank()

    base_path = os.path.join("data", problem_name, "results")
    precomputed_files = {
        "ANDREWS-SQUEEZER": f"results_speedup_at_acc_{dt=}_andrews.pkl",
        "LINEAR-TEST": f"results_speedup_at_acc_{dt=}_linear.pkl",
        "REACTION-DIFFUSION": f"results_speedup_at_acc_{dt=}_reaction_diffusion.pkl",
    }

    if filename is not None:
        print("Use precomputed results..\n")

        results_file = filename
    else:
        if problem_name in precomputed_files:
            results_file = precomputed_files[problem_name]
            path = os.path.join(base_path, results_file)
            if not os.path.exists(path):
                run_speedup_at_accuracy_test(
                    global_comm,
                    hook_class,
                    problem_name,
                    dt,
                    sweepers,
                    QI_serial_methods,
                    QI_parallel_methods,
                    stop_at_accuracy_for_speedup,
                    **kwargs,
                )
                results_file = build_filename(dt, problem_name)

        else:
            run_speedup_at_accuracy_test(
                global_comm,
                hook_class,
                problem_name,
                dt,
                sweepers,
                QI_serial_methods,
                QI_parallel_methods,
                stop_at_accuracy_for_speedup,
                **kwargs,
            )
            results_file = build_filename(dt, problem_name)

    path = os.path.join(base_path, results_file)
    if global_rank == 0:
        with open(path, "rb") as f:
            all_stats = dill.load(f)

        speedups, efficiencies, num_processes_by_ref = compute_speedups(
            all_stats, sweepers, QI_serial_methods, QI_parallel_methods, ref_QI=ref_QI, **kwargs
        )

        plot_speedups_and_efficiencies(
            problem_name,
            speedups,
            efficiencies,
            num_processes_by_ref,
            sweepers,
            QI_serial_methods,
            QI_parallel_methods,
            nodes_to_plot=nodes_to_plot,
            ref_QI=ref_QI,
            **kwargs,
        )

        # for num_nodes in nodes_to_plot:
        #     for quantity in ["walltime", "error", "number_iterations"]:
        #         plot_quantity_over_time(
        #             all_stats=all_stats,
        #             dt=dt,
        #             quantity=quantity,
        #             problem_name=problem_name,
        #             sweepers=sweepers,
        #             num_nodes=num_nodes,
        #             QI_serial_methods=QI_serial_methods,
        #             QI_parallel_methods=QI_parallel_methods,
        #             **kwargs,
        #         )

        # plot_error_vs_mpi_ranks(
        #     all_stats,
        #     problem_name,
        #     sweepers,
        #     QI_serial_methods,
        #     QI_parallel_methods,
        #     nodes_to_plot=nodes_to_plot,
        # )


def plot_speedups_and_efficiencies(
    problem_name: str,
    speedups: dict[str, dict[str, dict[int, float]]],
    efficiencies: dict[str, dict[str, dict[int, float]]],
    num_processes_by_ref: dict[str, list[int]],
    sweepers: list[str],
    QI_serial_methods: list[str],
    QI_parallel_methods: list[str],
    ref_QI: str = "IE",
    nodes_to_plot: list[int] = None,
    journal: str = "SIAM_Scientific_Computing",
    **kwargs: Any,
) -> None:
    r"""
    Plots speedup and efficiency.

    Parameters
    ----------
    problem_name : str
        Name of the problem.
    speedups : dict
        Contains the speedups.
    efficiencies : dict
        Contains the efficiencies.
    num_processes_by_key_ser : dict
        Contains the number of processes that are used in the run.
    sweepers : list of Sweeper
        Sweepers.
    QI_serial_methods : list of str
        The QIs indicate the serial methods, i.e., the serial SDC methods,
        the Radau methods, and the Runge-Kutta method(s).
    QI_parallel_methods : list of str
        Contains the QIs that indicate the parallel SDC schemes.
    serial_mode : str
        If set to ``"sdc"`` results are plotted against serial SDC methods.
        If it is set to ``"radau_rk"`` only Radau and Runge-Kutta methods
        are considered in plotting. Default is ``"sdc"``.
    nodes_to_plot : list of int
        The number of nodes that should be plottet. Default is ``None``, i.e.,
        all nodes from the run are plotted.
    journal : str, optional
        Name of the journal to obtain specified scale and height for figsize.
    """

    figsize = figsize_by_journal(journal, scale=0.65, ratio=0.45)
    fontsize = 5.5

    my_setup_mpl(fontsize=fontsize)
    plt.rcParams["axes.linewidth"] = 0.5
    colors, markers, _ = my_plot_style_config()
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    s_min, s_max = [], []
    for sweeper_type_ser in sweepers:
        sweeper_type_ser_eff = set_correct_sweeper_type(sweeper_type_ser, ref_QI)
        key_ref = f"{sweeper_type_ser_eff}_{ref_QI}"

        if key_ref not in speedups:
            continue

        used_nodes_all = set()

        nodes = set_nodes_for_plotting(num_processes_by_ref[key_ref], nodes_to_plot)

        for sweeper_type_par in sweepers:
            for QI_par in QI_parallel_methods:
                key_par = f"{sweeper_type_par}_{QI_par}"
                if key_par not in speedups[key_ref]:
                    continue

                if get_sweeper_type_from_key(key_par) != get_sweeper_type_from_key(key_ref):
                    continue

                s_map = speedups[key_ref][key_par]
                e_map = efficiencies[key_ref][key_par]

                xs = [n for n in nodes if n in s_map]

                used_nodes_all.update(xs)
                ys_s = [s_map[n] for n in xs]
                ys_e = [e_map[n] for n in xs]

                s_min.append(min(ys_s))
                s_max.append(max(ys_s))

                label = get_method_label(sweeper_type_par, QI_par)
                axs[0].semilogx(
                    xs,
                    ys_s,
                    color=colors[key_par],
                    marker=markers[key_par],
                    label=label,
                    linewidth=0.9,
                    linestyle="solid" if sweeper_type_par == "constrainedDAE" else "dashdot",
                    markersize=3.0,
                    markeredgewidth=0.35,
                )
                print(f"Plotted speedup for {key_par}: {ys_s}")
                axs[1].semilogx(
                    xs,
                    ys_e,
                    color=colors[key_par],
                    marker=markers[key_par],
                    label=label,
                    linewidth=0.9,
                    linestyle="solid" if sweeper_type_par == "constrainedDAE" else "dashdot",
                    markersize=3.0,
                    markeredgewidth=0.35,
                )
                print(f"Plotted maximum speedup for {key_par}: {max(ys_s)}")

    used_nodes_sorted = sorted(used_nodes_all)
    for ax in axs:
        ax.tick_params(axis="both", which="major", length=2.0, width=0.3)

        ax.set_xlabel(r"number of $\mathtt{MPI}$ processes")

        ax.set_xscale("linear")
        ax.set_xticks(used_nodes_sorted)
        ax.set_xticklabels(used_nodes_sorted)
        ax.grid(axis="both", which="major", linewidth=0.3, alpha=0.5)

        ax.set_yscale("linear")

        ax.tick_params(axis="both", which="minor", bottom=True, left=False)

        ax.set_xlim((nodes[0] - 0.12, nodes[-1] + 0.12))

    axs[0].set_ylabel("speedup")
    axs[1].set_ylabel("efficiency")
    ymax = max(s_max)
    ymin = min(s_min)
    if problem_name == "REACTION-DIFFUSION":
        axs[0].set_ylim(max(1.0, ymin - 0.4), ymax + 0.4)
    else:
        axs[0].set_ylim(bottom=0.87)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 0.08), ncol=2)

    plot_name = "speedup_at_accuracy"
    save_fig(plt, plot_name, problem_name)


def make_plots():
    global_comm = MPI.COMM_WORLD
    
    # Plots for LINEAR-TEST
    # print("\nGenerating plots for LINEAR-TEST...\n")
    # config_linear = get_configs(problem_name="LINEAR-TEST", config_type="speedup_at_accuracy")
    # filename = "results_speedup_at_acc_dt=0.05_linear_#1.pkl"
    # plots_speedup_at_accuracy(
    #     global_comm=global_comm, filename=filename, **config_linear
    # )

    print("\nGenerating plots for REACTION-DIFFUSION...\n")
    config_reacdiff = get_configs(problem_name="REACTION-DIFFUSION", config_type="speedup_at_accuracy")
    # filename = "results_speedup_at_acc_dt=0.05_reaction_diffusion_#2.pkl"
    filename = "results_speedup_at_acc_dt=0.05_reaction_diffusion_#8.pkl"
    plots_speedup_at_accuracy(
        global_comm=global_comm, filename=filename, **config_reacdiff
    )


if __name__ == "__main__":
    make_plots()
