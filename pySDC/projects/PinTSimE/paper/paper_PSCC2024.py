from pathlib import Path
import numpy as np
import pickle
from matplotlib.collections import LineCollection

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE

from pySDC.projects.PinTSimE.battery_model import generate_description, get_recomputed
from pySDC.projects.PinTSimE.discontinuous_test_ODE import controller_run
from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.DAE.run.discontinuous_test_DAE import LogEvent, LogGlobalErrorPostStepAlgebraicVariable
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
from pySDC.implementations.hooks.log_restarts import LogRestarts


def make_plots_for_test_DAE():
    """
    Makes the plot for the discontinuous test DAE, i.e.,

        - error over time for fixed time step size for different number of collocation nodes in
          comparison with use of switch detection and not,
        - global error for different step sizes and different number of collocation nodes in
          comparison with use of switch detection and not, additionally with number of restarts
          for each case,
        - event error to exact event time for differen step sizes and different number of
          collocation nodes.

    Thus, this function contains all the parameters used for this numerical example. Note that the 
    hook class "LogGlobalErrorPostStep" only logs the error of the differential variable. Hence, also
    the hook class "LogGlobalErrorPostStepAlgebraicVariable" is necessary.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    problem_class = DiscontinuousTestDAE

    sweeper = fully_implicit_DAE
    nnodes = [2, 3, 4, 5]
    quad_type = 'RADAU-RIGHT'
    maxiter = 45
    tol_hybr = 1e-6
    restol = 1e-13

    hookclass = [LogGlobalErrorPostStep, LogGlobalErrorPostStepAlgebraicVariable, LogEvent, LogRestarts]

    problem_params = dict()
    problem_params['newton_tol'] = tol_hybr

    use_detection = [False, True]
    max_restarts = 200
    epsilon_SE = 1e-10
    alpha = 0.95

    t0 = 3.0
    Tend = 5.4

    dt_list = [1 / (2 ** m) for m in range(2, 9)]
    dt_fix = 1 / (2 ** 7)

    recomputed = False

    results_error_over_time = {}
    results_error_norm = {}
    results_state_function = {}
    results_event_error = {}

    for M in nnodes:
        results_error_over_time[M], results_error_norm[M] = {}, {}
        results_state_function[M], results_event_error[M] = {}, {}

        for dt in dt_list:
            results_error_over_time[M][dt], results_error_norm[M][dt] = {}, {}
            results_state_function[M][dt], results_event_error[M][dt] = {}, {}

            for use_SE in use_detection:
                results_error_over_time[M][dt][use_SE], results_error_norm[M][dt][use_SE] = {}, {}
                results_state_function[M][dt][use_SE], results_event_error[M][dt][use_SE] = {}, {}

                description, controller_params = generate_description(
                    dt,
                    problem_class,
                    sweeper,
                    M,
                    quad_type,
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

                stats, t_switch_exact = controller_run(t0, Tend, controller_params, description)

                err_val = get_sorted(stats, type='e_global_post_step', sortby='time', recomputed=recomputed)
                results_error_over_time[M][dt][use_SE] = err_val

                err_norm = max([item[1] for item in err_val])
                results_error_norm[M][dt][use_SE] = err_norm

                h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
                h_abs = abs([item[1] for item in h_val][-1])
                results_state_function[M][dt][use_SE] = h_abs

                if use_SE:
                    switches = get_recomputed(stats, type='switch', sortby='time')

                    t_switch = [item[1] for item in switches][-1]
                    results_event_error[M][dt][use_SE]['err'] = abs(t_switch_exact - t_switch)

                    restarts = get_sorted(stats, type='restart', sortby='time', recomputed=None)
                    sum_restarts = sum([item[1] for item in restarts])
                    results_event_error[M][dt][use_SE]['restarts'] = sum_restarts

    plot_errors_over_time(results_error_over_time, dt_fix)
    plot_error_norm(results_error_norm)
    plot_state_function_detection(results_state_function)
    plot_event_time_error(results_event_error)


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
        3: (-13.0, 10),  
        4: (-13.0, -15),
        5: (-1.0, -34),
    }

    return colors, markers, xytext


def plot_errors_over_time(results_error_over_time, dt_fix=None):
    """
    Plots the errors over time for different numbers of collocation nodes in comparison with detection
    and not.

    Parameters
    ----------
    results_error_over_time : dict
        Results of the error for different number of coll.nodes.
    dt_fix : bool, optional
        If it is set to a considered step size, only one plot will generated.
    """

    colors, _, _ = plot_styling_stuff()

    dt_list = [dt_fix] if dt_fix is not None else results_error_over_time[2].keys()
    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
        for M in results_error_over_time.keys():
            for use_SE in results_error_over_time[M][dt].keys():
                err_val = results_error_over_time[M][dt][use_SE]
                t, err = [item[0] for item in err_val], [item[1] for item in err_val]

                linestyle_detection = 'solid' if not use_SE else 'dashdot'
                line, = ax.plot(t, err, color=colors[M], linestyle=linestyle_detection)

                if not use_SE:
                    line.set_label(r'$M={}$'.format(M))

                if M == 5:  # dummy plot for more pretty legend
                    ax.plot(3.5, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))


        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-15, 1e+1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'$t$', fontsize=16)
        ax.set_ylabel(r'$|y(t) - \tilde{y}(t)|$', fontsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=True, fontsize=12, loc='upper left')
        ax.minorticks_off()

        fig.savefig('data/test_DAE_error_over_time_dt{}.png'.format(dt), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_error_norm(results_error_norm):
    """
    Plots the error norm for different step sizes and different number of collocation nodes in comparison
    with detection and not.

    Parameters
    ----------
    results_error_norm : dict
        Statistics containing the error norms and sum of restarts for all considered coll. nodes.
    """

    colors, markers, xytext = plot_styling_stuff()

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_error_norm.keys():
        dt = list(results_error_norm[M].keys())
        for use_SE in results_error_norm[M][dt[0]].keys():
            err_norm_dt = [results_error_norm[M][k][use_SE] for k in dt]

            linestyle_detection = 'solid' if not use_SE else 'dashdot'
            line, = ax.loglog(
                dt,
                err_norm_dt,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
            )

            if not use_SE:
                line.set_label(r'$M={}$'.format(M))

            if M == 4:  # dummy plot for more pretty legend
                ax.plot(0, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-15, 1e+3)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'$\Delta t$', fontsize=16)
    ax.set_ylabel(r'$||y(t) - \tilde{y}(t)||_\infty$', fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc='lower right')

    fig.savefig('data/test_DAE_error_norms.png', dpi=300, bbox_inches='tight')
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

    colors, markers, _ = plot_styling_stuff()

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_state_function.keys():
        dt = list(results_state_function[M].keys())
        for use_SE in results_state_function[M][dt[0]].keys():
            h_abs = [results_state_function[M][k][use_SE] for k in dt]

            linestyle_detection = 'solid' if not use_SE else 'dashdot'
            line,  = ax.loglog(
                dt,
                h_abs,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
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
    ax.set_ylabel(r'$|h(y(T))|$', fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc='lower right')

    fig.savefig('data/test_DAE_state_function.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_event_time_error(results_event_error):
    """
    Plots the error between event time founded by detection and exact event time.

    Parameters
    ----------
    results_event_error : dict
        Contains event time error for each considered number of coll. nodes, step size and
        event detection and not.
    """

    colors, markers, xytext = plot_styling_stuff()

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_event_error.keys():
        dt = list(results_event_error[M].keys())
        for use_SE in [True]:
            event_error = [results_event_error[M][k][use_SE]['err'] for k in dt]

            linestyle_detection = 'solid' if not use_SE else 'dashdot'
            ax.loglog(
                dt,
                event_error,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
                label=r'$M={}$'.format(M),
            )

            if use_SE:
                sum_restarts = [results_event_error[M][k][use_SE]['restarts'] for k in dt]
                for m in range(len(dt)):
                    ax.annotate(
                        sum_restarts[m],
                        (dt[m], event_error[m]),
                        xytext=xytext[M],
                        textcoords="offset points",
                        color=colors[M],
                        fontsize=16,
                    )

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-15, 1e+1)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'$\Delta t$', fontsize=16)
    ax.set_ylabel(r'$|t^*_{ex} - t^*_{SE}|$', fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc='lower right')

    fig.savefig('data/test_DAE_event_time_error.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    make_plots_for_test_DAE()