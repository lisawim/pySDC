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
                results_state_function[M][dt][use_SE]['h_abs'] = h_abs

                if use_SE:
                    switches = get_recomputed(stats, type='switch', sortby='time')

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
                    h_max_event = []
                    print('h_all', len([item[0] for item in h_val_all]))
                    print()
                    print('switches_all', len([item[0] for item in switches_all]))
                    # for item in h_val_all:
                        # for item2 in switches_all:
                            # if abs(item[0] - item2[0]) <= 1e-14:
                                # print(item[0], t_switch)
                                # h_max_event.append(item[1])
                    results_event_error_restarts[M][dt][use_SE]['h_max_event'] = [item[1] for item in h_val_all]  # h_max_event

                    print(dt, M)

    plot_errors_over_time(results_error_over_time, dt_fix)
    plot_error_norm(results_error_norm)
    plot_state_function_detection(results_state_function)
    plot_event_time_error(results_event_error)
    plot_event_time_error_before_restarts(results_event_error_restarts, dt_fix)


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
        2: (-15.0, 16.5),
        3: (-2.0, 55),  
        4: (-13.0, -27),
        5: (-1.0, -40),
    }

    return colors, markers, xytext


def plot_errors_over_time(results_error_over_time, dt_fix=None):
    """
    Plots the errors over time for different numbers of collocation nodes in comparison with detection
    and not.

    Parameters
    ----------
    results_event_error_restarts : dict
        Results of the error for different number of coll.nodes.
    dt_fix : bool, optional
        If it is set to a considered step size, only one plot will generated.
    """

    colors, _, _ = plot_styling_stuff()

    M_key = list(results_error_over_time.keys())[0]
    dt_list = [dt_fix] if dt_fix is not None else results_error_over_time[M_key].keys()
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

            if M == 5:  # dummy plot for more pretty legend
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

    colors, markers, xytext = plot_styling_stuff()

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_state_function.keys():
        dt = list(results_state_function[M].keys())
        for use_SE in results_state_function[M][dt[0]].keys():
            h_abs = [results_state_function[M][k][use_SE]['h_abs'] for k in dt]

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
    ax.legend(frameon=True, fontsize=12, loc='upper left')

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

    colors, markers, _ = plot_styling_stuff()

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_event_error.keys():
        dt = list(results_event_error[M].keys())
        for use_SE in [True]:
            event_error = [results_event_error[M][k][use_SE] for k in dt]

            linestyle_detection = 'solid' if not use_SE else 'dashdot'
            ax.loglog(
                dt,
                event_error,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
                label=r'$M={}$'.format(M),
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


def plot_event_time_error_before_restarts(results_event_error_restarts, dt_fix=None):
    """
    Plots all events founded by switch estimation, not necessarily satisfying the conditions.

    Parameters
    ----------
    results_event_error_restarts : dict
        Contains all events for each considered number of coll. nodes, step size and
        event detection and not.
    dt_fix : float, optional
        Step size considered.
    """

    colors, markers, _ = plot_styling_stuff()

    M_key = list(results_event_error_restarts.keys())[0]
    dt_list = [dt_fix] if dt_fix is not None else results_event_error_restarts[M_key].keys()
    for dt in dt_list:
        lines, labels = [], []
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
        h_ax = ax.twinx()
        for M in results_event_error_restarts.keys():
            for use_SE in results_event_error_restarts[M][dt].keys():
                if use_SE:
                    event_error_all = results_event_error_restarts[M][dt][use_SE]['event_error_all']

                    event_error_label = r'$M={}$'.format(M)
                    h_val_event_label = r'State function $M={}$'.format(M)

                    line, = ax.semilogy(
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
                        ax.plot(1, event_error_all[0], color='black', linestyle='solid', label=r'$|t^*_{ex} - t^*_{SE}|$')
                        ax.plot(
                            1,
                            1e+2,
                            color='black',
                            linestyle='dashdot',
                            marker=markers[M],
                            markersize=5,
                            alpha=0.4,
                            label=r'$||h(t)||_\infty$',
                        )

        h_ax.tick_params(labelsize=16)
        h_ax.set_ylim(1e-11, 1e+0)
        h_ax.set_yscale('log', base=10)
        h_ax.set_ylabel(r'$||h(t)||_\infty$', fontsize=21)
        h_ax.minorticks_off()

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-11, 1e-1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel('Founded events', fontsize=16)
        ax.set_ylabel(r'$|t^*_{ex} - t^*_{SE}|$', fontsize=16)
        ax.grid(visible=True)
        ax.minorticks_off()
        ax.legend(frameon=True, fontsize=12, loc='upper right')

        fig.savefig('data/test_DAE_event_time_error_restarts_dt{}.png'.format(dt), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


if __name__ == "__main__":
    make_plots_for_test_DAE()