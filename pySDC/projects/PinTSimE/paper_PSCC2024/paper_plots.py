from pathlib import Path
import numpy as np
import dill
import os
import matplotlib.pyplot as plt

from pySDC.core.errors import ParameterError

from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE
from pySDC.projects.DAE.problems.discontinuousTestDAE import DiscontinuousTestDAE
from pySDC.projects.DAE.problems.wscc9BusSystem import WSCC9BusSystem

from pySDC.projects.PinTSimE.battery_model import generateDescription
from pySDC.projects.PinTSimE.battery_model import controllerRun
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import my_setup_mpl
from pySDC.helpers.plot_helper import figsize_by_journal

from pySDC.projects.PinTSimE.paper_PSCC2024.log_event import LogEventDiscontinuousTestDAE, LogEventWSCC9
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
from pySDC.implementations.hooks.log_restarts import LogRestarts


def make_plots_for_test_DAE(journal="BUW_thesis"):  # pragma: no cover
    """
    Makes the plot for the discontinuous test DAE, i.e.,

        - error over time for fixed time step size for different number of collocation nodes in
          comparison with use of switch detection and not,
        - error norm for different step sizes and different number of collocation nodes in
          comparison with use of switch detection and not, additionally with number of restarts
          for each case,
        - absolute value of state function at end time for different number of collocation nodes
          and different step sizes in comparison with use of switch detection and not,
        - event error to exact event time for differen step sizes and different number of
          collocation nodes,
        - plots event time error of all founded events (not necessarily satisfying the tolerances)
          and the maximum value of the state function in this time step for different number of
          collocation nodes in comparison with use of switch detection and not.

    Thus, this function contains all the parameters used in the paper for this numerical example.
    """
    problem_name = "DISC-TEST"

    output_dir = Path("data") / problem_name / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = "results_data.pkl"
    base_path = output_dir

    results_path = os.path.join(base_path, results_file)
    if not os.path.exists(results_path):
        print(f"Results file {results_path} does not exist. Running the simulations to generate the results.")

        problem_class = DiscontinuousTestDAE

        sweeper = FullyImplicitDAE
        nnodes = [2, 3, 4, 5]
        quad_type = 'RADAU-RIGHT'
        QI = 'LU'
        maxiter = 45
        tol_hybr = 1e-6
        restol = 1e-13

        hook_class = [LogGlobalErrorPostStep, LogEventDiscontinuousTestDAE, LogRestarts]

        problem_params = dict()
        problem_params['newton_tol'] = tol_hybr

        use_detection = [False, True]
        max_restarts = 200
        epsilon_SE = 1e-10
        alpha = 0.95

        t0 = 3.0
        Tend = 5.4

        dt_list = [1 / (2**m) for m in range(2, 9)]
        dt_fix = 1 / (2**7)

        recomputed = False

        results_error_over_time = {}
        results_error_norm = {}
        results_state_function = {}
        results_event_error = {}
        results_event_error_restarts = {}

        for M in nnodes:
            results_error_over_time[M], results_error_norm[M] = {}, {}
            results_state_function[M], results_event_error[M] = {}, {}
            results_event_error_restarts[M] = {}

            for dt in dt_list:
                results_error_over_time[M][dt], results_error_norm[M][dt] = {}, {}
                results_state_function[M][dt], results_event_error[M][dt] = {}, {}
                results_event_error_restarts[M][dt] = {}

                for use_SE in use_detection:
                    results_error_over_time[M][dt][use_SE], results_error_norm[M][dt][use_SE] = {}, {}
                    results_state_function[M][dt][use_SE], results_event_error[M][dt][use_SE] = {}, {}
                    results_event_error_restarts[M][dt][use_SE] = {}

                    description, controller_params, controller = generateDescription(
                        dt,
                        problem_class,
                        sweeper,
                        M,
                        quad_type,
                        QI,
                        hook_class,
                        False,
                        use_SE,
                        problem_params,
                        restol,
                        maxiter,
                        max_restarts,
                        epsilon_SE,
                        alpha,
                    )

                    stats, t_switch_exact = controllerRun(
                        description, controller_params, controller, t0, Tend, exact_event_time_avail=True
                    )

                    err_val = get_sorted(stats, type='e_global_post_step', sortby='time', recomputed=recomputed)
                    results_error_over_time[M][dt][use_SE] = err_val

                    err_norm = max([item[1] for item in err_val])
                    results_error_norm[M][dt][use_SE] = err_norm

                    h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
                    h_abs = abs([item[1] for item in h_val][-1])
                    results_state_function[M][dt][use_SE]['h_abs'] = h_abs

                    if use_SE:
                        switches = get_sorted(stats, type='switch', sortby='time', recomputed=recomputed)

                        t_switch = [item[1] for item in switches][-1]
                        results_event_error[M][dt][use_SE] = abs(t_switch_exact - t_switch)

                        restarts = get_sorted(stats, type='restart', sortby='time', recomputed=None)
                        sum_restarts = sum([item[1] for item in restarts])
                        results_state_function[M][dt][use_SE]['restarts'] = sum_restarts

                        switches_all = get_sorted(stats, type='switch_all', sortby='time', recomputed=None)
                        t_switches_all = [item[1] for item in switches_all]
                        event_error_all = [abs(t_switch_exact - t_switch) for t_switch in t_switches_all]
                        results_event_error_restarts[M][dt][use_SE]['event_error_all'] = event_error_all
                        h_val_all = get_sorted(stats, type='h_all', sortby='time', recomputed=None)
                        results_event_error_restarts[M][dt][use_SE]['h_max_event'] = [item[1] for item in h_val_all]

        results = {
            "results_error_over_time": results_error_over_time,
            "results_error_norm": results_error_norm,
            "results_state_function": results_state_function,
            "results_event_error": results_event_error,
            "results_event_error_restarts": results_event_error_restarts,
            "metadata": {
                "dt_fix": dt_fix,
                "nnodes": nnodes,
                "dt_list": dt_list,
                "use_detection": use_detection,
                "QI": QI,
                "quad_type": quad_type,
            },
        }

        with results_path.open("wb") as f:
            dill.dump(results, f)

    else:
        results_path = os.path.join(base_path, results_file)
        with open(results_path, "rb") as f:
            loaded_results = dill.load(f)

        results_error_over_time = loaded_results["results_error_over_time"]
        results_error_norm = loaded_results["results_error_norm"]
        results_state_function = loaded_results["results_state_function"]
        results_event_error = loaded_results["results_event_error"]
        results_event_error_restarts = loaded_results["results_event_error_restarts"]

        dt_fix = loaded_results["metadata"]["dt_fix"]

    plot_functions_over_time(
        results_error_over_time, problem_name, r"global error $|y(t) - y_{ex}(t)|$", dt_fix, journal
    )
    plot_error_norm(results_error_norm, problem_name, journal)
    plot_state_function_detection(
        results_state_function, problem_name, r"absolute value of state function $|h(y(T))|$", journal
    )
    plot_event_time_error(results_event_error, problem_name, journal)
    plot_event_time_error_before_restarts(results_event_error_restarts, problem_name, dt_fix, journal)


def make_plots_for_WSCC9_test_case(journal="BUW_thesis"):  # pragma: no cover
    """
    Generates the plots for the WSCC 9-bus test case, i.e.,

        - the values of the state function over time for different number of collocation nodes in comparison
          with event detection and not,
        - the values of the state function at end time for different number of collocation nodes and
          different step sizes.

    Thus, this function contains all the parameters used for this numerical example.

    Parameters
    ----------
    cwd : str, optional
        Current working directory.
    """

    problem_name = "WSCC9"

    output_dir = Path("data") / problem_name / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = "results_data.pkl"
    results_path = output_dir / results_file

    problem_class = WSCC9BusSystem
    prob_class_name = WSCC9BusSystem.__name__

    sweeper = FullyImplicitDAE
    nnodes = [2, 3, 4, 5]
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    maxiter = 50
    tol_hybr = 1e-10
    restol = 5e-13

    hook_class = [LogEventWSCC9, LogRestarts]

    problem_params = dict()
    problem_params['newton_tol'] = tol_hybr

    use_detection = [False, True]
    max_restarts = 400
    epsilon_SE = 1e-10
    alpha = 0.95

    t0 = 0.0
    Tend = 0.7

    dt_list = [1 / (2**m) for m in range(5, 11)]
    dt_fix = 1 / (2**8)

    recomputed = False

    if not results_path.exists():
        print(f"Results file {results_path} does not exist. Collecting/generating results.")
        results_state_function_over_time = {}
        results_state_function_detection = {}
        for M in nnodes:
            results_state_function_over_time[M], results_state_function_detection[M] = {}, {}

            for dt in dt_list:
                results_state_function_over_time[M][dt], results_state_function_detection[M][dt] = {}, {}

                for use_SE in use_detection:
                    results_state_function_over_time[M][dt][use_SE], results_state_function_detection[M][dt][use_SE] = (
                        {},
                        {},
                    )

                    description, controller_params, controller = generateDescription(
                        dt,
                        problem_class,
                        sweeper,
                        M,
                        quad_type,
                        QI,
                        hook_class,
                        False,
                        use_SE,
                        problem_params,
                        restol,
                        maxiter,
                        max_restarts,
                        epsilon_SE,
                        alpha,
                    )

                    stats_file = output_dir / f"WSCC9BusSystem_{M=}_{dt=}_{use_SE=}.dat"
                    if stats_file.is_file():
                        print("Load file")
                        with stats_file.open("rb") as f:
                            stats = dill.load(f)
                    else:
                        print("Compute results")
                        stats, _ = controllerRun(description, controller_params, controller, t0, Tend)

                        with stats_file.open("wb") as f:
                            dill.dump(stats, f)

                    h_val = get_sorted(stats, type="state_function", sortby="time", recomputed=recomputed)
                    results_state_function_over_time[M][dt][use_SE] = h_val

                    h_abs_end = abs(h_val[-1][1])
                    results_state_function_detection[M][dt][use_SE]["h_abs"] = h_abs_end

                    if use_SE:
                        restarts = get_sorted(stats, type="restart", sortby="time", recomputed=None)
                        sum_restarts = sum(item[1] for item in restarts)
                        results_state_function_detection[M][dt][use_SE]["restarts"] = sum_restarts

        results = {
            "results_state_function_over_time": results_state_function_over_time,
            "results_state_function_detection": results_state_function_detection,
            "metadata": {
                "problem_name": problem_name,
                "prob_class_name": prob_class_name,
                "dt_fix": dt_fix,
                "nnodes": nnodes,
                "dt_list": dt_list,
                "use_detection": use_detection,
                "QI": QI,
                "quad_type": quad_type,
                "maxiter": maxiter,
                "restol": restol,
                "max_restarts": max_restarts,
                "epsilon_SE": epsilon_SE,
                "alpha": alpha,
                "t0": t0,
                "Tend": Tend,
            },
        }

        with results_path.open("wb") as f:
            dill.dump(results, f)

    else:
        print("Load results from file")
        with results_path.open("rb") as f:
            loaded_results = dill.load(f)

        results_state_function_over_time = loaded_results["results_state_function_over_time"]
        results_state_function_detection = loaded_results["results_state_function_detection"]
        dt_fix = loaded_results["metadata"]["dt_fix"]
        prob_class_name = loaded_results["metadata"]["prob_class_name"]

    plot_functions_over_time(
        results_state_function_over_time,
        prob_class_name,
        r'Absolute value of $h$ $|h(P_{SV,0}(t))|$',
        dt_fix,
        journal,
    )
    plot_state_function_detection(
        results_state_function_detection, prob_class_name, r'Absolute value of $h$ $|h(P_{SV,0}(T))|$', journal
    )


def plot_styling_stuff(problem_name):  # pragma: no cover
    """
    Implements all the stuff needed for making the plots more pretty.
    """

    colors = {
        2: 'limegreen',
        3: 'firebrick',
        4: 'deepskyblue',
        5: 'purple',
    }

    markers = {
        2: 's',
        3: 'o',
        4: '*',
        5: 'd',
    }

    if problem_name == "DISC-TEST":
        xytext = {
            2: (-8.0, 9.5),
            3: (-2.0, 55),
            4: (-13.0, -27),
            5: (-1.0, -40),
        }
    elif problem_name == 'WSCC9BusSystem':
        xytext = {
            2: (-13.0, 16),
            3: (-13.0, 30),
            4: (-13.0, -17),
            5: (-1.0, -38),
        }
    else:
        raise ParameterError(f"For {problem_name} no dictionary for position of data points is set up!")

    return colors, markers, xytext


def plot_functions_over_time(
    results_function_over_time, problem_name, y_label, dt_fix=None, journal="BUW_thesis"
):  # pragma: no cover
    """
    Plots the functions over time for different numbers of collocation nodes in comparison with detection
    and not.

    Parameters
    ----------
    results_function_over_time : dict
        Results of some function over time for different number of coll.nodes.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    y_label : str
        y-label used for the plot.
    loc_legend : str
        Location of the legend in the plot.
    dt_fix : bool, optional
        If it is set to a considered step size, only one plot will generated.
    """

    my_setup_mpl(fontsize=5)
    figsize = figsize_by_journal(journal, scale=0.55, ratio=0.73)

    colors, _, _ = plot_styling_stuff(problem_name)
    x0 = 3.5 if problem_name == "DISC-TEST" else 0.5

    M_key = list(results_function_over_time.keys())[0]
    dt_list = [dt_fix] if dt_fix is not None else results_function_over_time[M_key].keys()
    for dt in dt_list:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for M in results_function_over_time.keys():
            for use_SE in results_function_over_time[M][dt].keys():
                err_val = results_function_over_time[M][dt][use_SE]
                t, err = [item[0] for item in err_val], [abs(item[1]) for item in err_val]

                linestyle_detection = "solid" if not use_SE else "dashdot"
                (line,) = ax.plot(t, err, color=colors[M], linestyle=linestyle_detection)

                if not use_SE:
                    line.set_label(rf"$M$ = {M}")

                if M == 5:  # dummy plot for more pretty legend
                    ax.plot(x0, 0, color="black", linestyle=linestyle_detection, label=f"Detection: {use_SE}")

        ax.set_xlabel(r"time $t$")
        ax.set_ylabel(y_label)

        ax.set_xlim((t[0], t[-1]))
        ax.set_ylim(1e-15, 1e1)
        ax.set_yscale("log", base=10)

        ax.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
        ax.grid(which="minor", axis="x", linewidth=0.25, alpha=0.10)
        ax.grid(which="major", axis="y", linewidth=0.45, alpha=0.35)

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

        if problem_name == "DISC-TEST":
            filename = f"test_DAE_error_over_time_{dt=}"
        elif problem_name == "WSCC9BusSystem":
            filename = f"wscc9_state_function_over_time_{dt=}"
        
        filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)


def plot_error_norm(results_error_norm, problem_name, journal="BUW_thesis"):  # pragma: no cover
    """
    Plots the error norm for different step sizes and different number of collocation nodes in comparison
    with detection and not.

    Parameters
    ----------
    results_error_norm : dict
        Statistics containing the error norms and sum of restarts for all considered coll. nodes.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    """

    colors, markers, xytext = plot_styling_stuff(problem_name)

    my_setup_mpl(fontsize=5)
    figsize = figsize_by_journal(journal, scale=0.55, ratio=0.73)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for M in results_error_norm.keys():
        dt = list(results_error_norm[M].keys())
        for use_SE in results_error_norm[M][dt[0]].keys():
            err_norm_dt = [results_error_norm[M][k][use_SE] for k in dt]

            linestyle_detection = "solid" if not use_SE else "dashdot"
            (line,) = ax.loglog(
                dt,
                err_norm_dt,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
            )

            if not use_SE:
                line.set_label(rf"$M$ = {M}")

            if M == 5:  # dummy plot for more pretty legend
                ax.plot(0, 0, color="black", linestyle=linestyle_detection, label=f"Detection: {use_SE}")

    ax.tick_params(axis="both", which="major")

    ax.set_ylim(1e-15, 1e3)

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)

    ax.set_xlabel(r"time step size $\Delta t$")
    ax.set_ylabel(r"error norm $||y(t) - \tilde{y}(t)||_\infty$")

    ax.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
    ax.grid(which="minor", axis="x", linewidth=0.25, alpha=0.10)
    ax.grid(which="major", axis="y", linewidth=0.45, alpha=0.35)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + "test_DAE_error_norms.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_state_function_detection(results_state_function, problem_name, y_label, journal="BUW_thesis"):  # pragma: no cover
    """
    Plots the absolute value of the state function after the event which denotes how close it is to the zero.

    Parameters
    ----------
    results_state_function : dict
        Contains the absolute value of the state function for each number of coll. nodes, each step size and
        detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    y_label : str
        y-label used for the plot.
    loc_legend : str
        Location of the legend in the plot.
    """

    colors, markers, _ = plot_styling_stuff(problem_name)

    my_setup_mpl(fontsize=5)
    figsize = figsize_by_journal(journal, scale=0.75, ratio=0.5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for M in results_state_function.keys():
        dt = list(results_state_function[M].keys())
        for use_SE in results_state_function[M][dt[0]].keys():
            h_abs = [results_state_function[M][k][use_SE]['h_abs'] for k in dt]

            linestyle_detection = "solid" if not use_SE else "dashdot"
            (line0,) = axs[0].loglog(
                dt,
                h_abs,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
            )

            if not use_SE:
                line0.set_label(rf"$M$ = {M}")

            if use_SE:
                sum_restarts = [results_state_function[M][k][use_SE]["restarts"] for k in dt]
                axs[1].semilogx(
                    dt,
                    sum_restarts,
                    color=colors[M],
                    linestyle=linestyle_detection,
                    marker=markers[M],
                )

            if M == 5:  # dummy plot for more pretty legend
                axs[0].plot(0, 0, color="black", linestyle=linestyle_detection, label=f"Detection: {use_SE}")

    for ax in axs:
        ax.set_xlabel(r"time step size $\Delta t$")

        ax.tick_params(axis="both", which="major")

        ax.set_xscale("log", base=10)

        ax.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
        ax.grid(which="minor", axis="x", linewidth=0.25, alpha=0.10)
        ax.grid(which="major", axis="y", linewidth=0.45, alpha=0.35)

    axs[0].set_ylim(1e-17, 1e3)

    axs[0].set_yscale("log", base=10)

    axs[0].set_ylabel(y_label)
    axs[1].set_ylabel("number of restarts")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + "state_function_detection.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_event_time_error(results_event_error, problem_name, journal="BUW_thesis"):  # pragma: no cover
    """
    Plots the error between event time founded by detection and exact event time.

    Parameters
    ----------
    results_event_error : dict
        Contains event time error for each considered number of coll. nodes, step size and
        event detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    """

    colors, markers, _ = plot_styling_stuff(problem_name)

    my_setup_mpl(fontsize=5)
    figsize = figsize_by_journal(journal, scale=0.45, ratio=1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for M in results_event_error.keys():
        dt = list(results_event_error[M].keys())
        for use_SE in [True]:
            event_error = [results_event_error[M][k][use_SE] for k in dt]

            linestyle_detection = "solid" if not use_SE else "dashdot"
            ax.loglog(
                dt,
                event_error,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
                label=rf"$M$ = {M}",
            )

    ax.tick_params(axis="both", which="major")

    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)

    ax.set_ylim(1e-15, 1e1)

    ax.set_xlabel(r"time step size $\Delta t$")
    ax.set_ylabel(r"event time error $|t^*_{ex} - t^*_{SE}|$")

    ax.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
    ax.grid(which="minor", axis="x", linewidth=0.25, alpha=0.10)
    ax.grid(which="major", axis="y", linewidth=0.45, alpha=0.35)
    ax.grid(which="minor", axis="y", linewidth=0.25, alpha=0.10)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + "test_DAE_event_time_error.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_event_time_error_before_restarts(results_event_error_restarts, prob_class, dt_fix=None, journal="BUW_thesis"):  # pragma: no cover
    """
    Plots all events founded by switch estimation, not necessarily satisfying the conditions.

    Parameters
    ----------
    results_event_error_restarts : dict
        Contains all events for each considered number of coll. nodes, step size and
        event detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    dt_fix : float, optional
        Step size considered.
    """

    my_setup_mpl(fontsize=6)
    figsize = figsize_by_journal(journal, scale=0.62, ratio=0.68)

    colors, markers, _ = plot_styling_stuff(prob_class)

    M_key = list(results_event_error_restarts.keys())[0]
    dt_list = [dt_fix] if dt_fix is not None else results_event_error_restarts[M_key].keys()
    for dt in dt_list:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        h_ax = ax.twinx()
        for M in results_event_error_restarts.keys():
            for use_SE in results_event_error_restarts[M][dt].keys():
                if use_SE:
                    event_error_all = results_event_error_restarts[M][dt][use_SE]['event_error_all']

                    (line,) = ax.semilogy(
                        np.arange(1, len(event_error_all) + 1),
                        event_error_all,
                        color=colors[M],
                        linestyle='solid',
                        # marker=markers[M],
                    )

                    line.set_label(r'$M={}$'.format(M))

                    h_max_event = results_event_error_restarts[M][dt][use_SE]['h_max_event']
                    h_ax.semilogy(
                        np.arange(1, len(h_max_event) + 1),
                        h_max_event,
                        color=colors[M],
                        linestyle='dashdot',
                        marker=markers[M],
                        markersize=5,
                        alpha=0.4,
                    )

                    if M == 5:  # dummy plot for more pretty legend
                        ax.plot(
                            1, event_error_all[0], color='black', linestyle='solid', label=r'$|t^*_{ex} - t^*_{SE}|$'
                        )
                        ax.plot(
                            1,
                            1e2,
                            color='black',
                            linestyle='dashdot',
                            marker=markers[M],
                            markersize=5,
                            alpha=0.4,
                            label=r'$||h(t)||_\infty$',
                        )

        # h_ax.tick_params(labelsize=16)
        h_ax.set_ylim(1e-11, 1e0)
        h_ax.set_yscale('log', base=10)
        h_ax.set_ylabel(r'Maximum value of h $||h(t)||_\infty$')
        h_ax.minorticks_off()

        ax.tick_params(axis='both', which='major')
        ax.set_ylim(1e-11, 1e-1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'Restarted steps $n_{restart}$')
        ax.set_ylabel(r'Event time error $|t^*_{ex} - t^*_{SE}|$')
        ax.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
        ax.grid(which="minor", axis="x", linewidth=0.25, alpha=0.10)
        ax.grid(which="major", axis="y", linewidth=0.45, alpha=0.35)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=3)

        fig.savefig('data/test_DAE_event_time_error_restarts_dt{}.png'.format(dt), dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    # make_plots_for_test_DAE()
    make_plots_for_WSCC9_test_case()
