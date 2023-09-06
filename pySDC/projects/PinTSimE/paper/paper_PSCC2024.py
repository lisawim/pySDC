from pathlib import Path
import numpy as np
import pickle
from matplotlib.collections import LineCollection

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.problems.IEEE9BusSystem import IEEE9BusSystem

from pySDC.projects.PinTSimE.battery_model import generate_description, get_recomputed, controller_run
from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.DAE.run.ieee9 import LogEvent
from pySDC.implementations.hooks.log_restarts import LogRestarts


def make_plots_for_IEEE9_test_case():
    """
    Generates the plots for the IEEE9 bus test case, i.e.,

        - the values of the state function over time for different number of collocation nodes in comparison
        with event detection and not,
        - the values of the state function at end time for different number of collocation nodes and
          different step sizes.

    Thus, this function contains all the parameters used for this numerical example.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    problem_class = IEEE9BusSystem

    sweeper = fully_implicit_DAE
    nnodes = [2, 3, 4, 5]
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    maxiter = 50
    tol_hybr = 1e-10
    restol = 5e-13

    hookclass = [LogEvent, LogRestarts]

    problem_params = dict()
    problem_params['newton_tol'] = tol_hybr

    use_detection = [False, True]
    max_restarts = 400
    epsilon_SE = 1e-10
    alpha = 0.95

    t0 = 0.0
    Tend = 0.7

    dt_list = [1 / (2 ** 8)]  # [1 / (2 ** m) for m in range(5, 11)]
    dt_fix = 1 / (2 ** 8)

    recomputed = False

    results_state_function_over_time = {}
    results_state_function_detection = {}
    for M in nnodes:
        results_state_function_over_time[M], results_state_function_detection[M] = {}, {}

        for dt in dt_list:
            results_state_function_over_time[M][dt], results_state_function_detection[M][dt] = {}, {}

            for use_SE in use_detection:
                print(M, dt, use_SE)
                results_state_function_over_time[M][dt][use_SE], results_state_function_detection[M][dt][use_SE] = {}, {}

                description, controller_params = generate_description(
                    dt,
                    problem_class,
                    sweeper,
                    M,
                    quad_type,
                    QI,
                    hookclass,
                    False,
                    use_SE,
                    problem_params,
                    restol,
                    maxiter,
                    max_restarts,
                    epsilon_SE,
                    alpha,
                )

                stats = controller_run(description, controller_params, False, use_SE, t0, Tend)

                h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
                results_state_function_over_time[M][dt][use_SE] = h_val

                h_abs_end = abs([me[1] for me in h_val][-1])
                results_state_function_detection[M][dt][use_SE]['err'] = h_abs_end

                if use_SE:
                    restarts = get_sorted(stats, type='restart', sortby='time', recomputed=None)
                    sum_restarts = sum([me[1] for me in restarts])
                    results_state_function_detection[M][dt][use_SE]['restarts'] = sum_restarts

    plot_state_function_over_time(results_state_function_over_time, dt_fix)
    plot_state_function_detection(results_state_function_detection)


def plot_styling_stuff():
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

    xytext = {
        2: (-13.0, 16),
        3: (-13.0, 30),  
        4: (-13.0, -17),
        5: (-1.0, -38),
    }

    return colors, markers, xytext


def plot_state_function_over_time(results_state_function_over_time, dt_fix=None):
    """
    Plots the errors over time for different numbers of collocation nodes in comparison with detection
    and not.

    Parameters
    ----------
    results_state_function_over_time : dict
        Results of the error for different number of coll.nodes.
    dt_fix : bool, optional
        If it is set to a considered step size, only one plot will generated.
    """

    colors, _, _ = plot_styling_stuff()

    dt_list = [dt_fix] if dt_fix is not None else results_state_function_over_time[2].keys()
    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
        for M in results_state_function_over_time.keys():
            for use_SE in results_state_function_over_time[M][dt].keys():
                err_val = results_state_function_over_time[M][dt][use_SE]
                t, err = [item[0] for item in err_val], [abs(item[1]) for item in err_val]

                linestyle_detection = 'solid' if not use_SE else 'dashdot'
                line, = ax.plot(t, err, color=colors[M], linestyle=linestyle_detection)

                if not use_SE:
                    line.set_label(r'$M={}$'.format(M))

                if M == 5:  # dummy plot for more pretty legend
                    ax.plot(0.5, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))


        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-15, 1e+1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'$t$', fontsize=16)
        ax.set_ylabel(r'$|h(P_{SV,0}(t))|$', fontsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=True, fontsize=12, loc='lower left')
        ax.minorticks_off()

        fig.savefig('data/ieee9_state_function_over_time_dt{}.png'.format(dt), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_state_function_detection(results_state_function):
    """
    Plots the absolute value of the state function after the event which denotes how close it is to the zero.

    Parameters
    ----------
    results_state_function : dict
        Contains the absolute value of the state function for each number of coll. nodes, each step size and
        detection and not.
    """

    colors, markers, xytext = plot_styling_stuff()

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_state_function.keys():
        dt = list(results_state_function[M].keys())
        for use_SE in results_state_function[M][dt[0]].keys():
            h_abs = [results_state_function[M][k][use_SE]['err'] for k in dt]

            linestyle_detection = 'solid' if not use_SE else 'dashdot'
            line,  = ax.loglog(
                dt,
                h_abs,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
            )

            if use_SE:
                sum_restarts = [results_state_function[M][k][use_SE]['restarts'] for k in dt]
                for m in range(len(dt)):
                    ax.annotate(
                        sum_restarts[m],
                        (dt[m], h_abs[m]),
                        xytext=xytext[M],
                        textcoords="offset points",
                        color=colors[M],
                        fontsize=16,
                    )

            if not use_SE:
                line.set_label(r'$M={}$'.format(M))

            if M == 5:  # dummy plot for more pretty legend
                ax.plot(0, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-17, 1e+3)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'$\Delta t$', fontsize=16)
    ax.set_ylabel(r'$|h(P_{SV,0}(T))|$', fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc='upper right')

    fig.savefig('data/ieee9_state_function_detection.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    make_plots_for_IEEE9_test_case()