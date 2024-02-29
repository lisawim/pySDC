from pathlib import Path
import numpy as np
import dill

from pySDC.core.Errors import ParameterError

from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.Runge_Kutta_DAE import EDIRK4DAE, TrapezoidalRuleDAE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
from pySDC.projects.DAE.problems.WSCC9BusSystem import WSCC9BusSystem

from pySDC.projects.PinTSimE.battery_model import generateDescription
from pySDC.projects.PinTSimE.battery_model import controllerRun
from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.paper_PSCC2024.log_event import LogEventDiscontinuousTestDAE, LogEventWSCC9
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable
from pySDC.implementations.hooks.log_restarts import LogRestarts


def make_plots_for_test_DAE():  # pragma: no cover
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

    Path("data").mkdir(parents=True, exist_ok=True)

    problem_class = DiscontinuousTestDAE
    prob_class_name = DiscontinuousTestDAE.__name__

    nnodes, dt_list, use_detection, alphas = get_dict_keys(prob_class_name)

    sweeper_classes = [fully_implicit_DAE]
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    maxiter = 60
    tol_hybr = 1e-6
    restol = 1e-13

    hook_class = [LogGlobalErrorPostStepDifferentialVariable, LogEventDiscontinuousTestDAE, LogRestarts]

    problem_params = dict()
    problem_params['newton_tol'] = tol_hybr

    max_restarts = 200
    epsilon_SE = 1e-10
    typeFD = 'backward'

    alpha_fix = alphas[0]#alphas[4]

    t0 = 3.5
    Tend = 5.0

    dt_fix = dt_list[-2]

    recomputed = False

    results_error_over_time = {}
    results_error_norm = {}
    results_state_function = {}
    results_event_error = {}
    results_event_error_restarts = {}

    dt_list = [dt_list] if isinstance(dt_list, float) else dt_list
    for sweeper in sweeper_classes:
        sweeper_cls_name = sweeper.__name__
        results_error_over_time[sweeper_cls_name][M], results_error_norm[sweeper_cls_name][M] = {}, {}
        results_state_function[sweeper_cls_name][M], results_event_error[sweeper_cls_name][M] = {}, {}
        results_event_error_restarts[sweeper_cls_name][M] = {}

        for M in nnodes:
            results_error_over_time[sweeper_cls_name][M], results_error_norm[sweeper_cls_name][M] = {}, {}
            results_state_function[sweeper_cls_name][M], results_event_error[sweeper_cls_name][M] = {}, {}
            results_event_error_restarts[sweeper_cls_name][M] = {}

            for dt in dt_list:
                results_error_over_time[sweeper_cls_name][M][dt], results_error_norm[sweeper_cls_name][M][dt] = {}, {}
                results_state_function[sweeper_cls_name][M][dt], results_event_error[sweeper_cls_name][M][dt] = {}, {}
                results_event_error_restarts[sweeper_cls_name][M][dt] = {}

                for use_SE in use_detection:
                    results_error_over_time[sweeper_cls_name][M][dt][use_SE], results_error_norm[sweeper_cls_name][M][dt][use_SE] = {}, {}
                    results_state_function[sweeper_cls_name][M][dt][use_SE], results_event_error[sweeper_cls_name][M][dt][use_SE] = {}, {}
                    results_event_error_restarts[sweeper_cls_name][M][dt][use_SE] = {}

                    for alpha in alphas:
                        results_error_over_time[sweeper_cls_name][M][dt][use_SE][alpha], results_error_norm[sweeper_cls_name][M][dt][use_SE][alpha] = {}, {}
                        results_state_function[sweeper_cls_name][M][dt][use_SE][alpha], results_event_error[sweeper_cls_name][M][dt][use_SE][alpha] = {}, {}
                        results_event_error_restarts[M][dt][use_SE][alpha] = {}[sweeper_cls_name]

                        description, controller_params, controller = generateDescription(
                            dt=dt,
                            problem=problem_class,
                            sweeper=sweeper,
                            num_nodes=M,
                            quad_type=quad_type,
                            QI=QI,
                            hook_class=hook_class,
                            use_adaptivity=False,
                            use_switch_estimator=use_SE,
                            problem_params=problem_params,
                            restol=restol,
                            maxiter=maxiter,
                            max_restarts=max_restarts,
                            tol_event=epsilon_SE,
                            alpha=alpha,
                            typeFD=typeFD,
                        )

                        stats, t_switch_exact = controllerRun(
                            description, controller_params, controller, t0, Tend, exact_event_time_avail=True
                        )

                        err_val = get_sorted(stats, type='e_global_differential_post_step', sortby='time', recomputed=recomputed)
                        results_error_over_time[sweeper_cls_name][M][dt][use_SE][alpha] = err_val

                        err_norm = max([item[1] for item in err_val])
                        results_error_norm[sweeper_cls_name][M][dt][use_SE][alpha] = err_norm

                        h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
                        h_abs = abs([item[1] for item in h_val][-1])
                        results_state_function[sweeper_cls_name][M][dt][use_SE][alpha]['h_abs'] = h_abs
                        print(M, dt, use_SE, alpha)
                        if use_SE:
                            switches = get_sorted(stats, type='switch', sortby='time', recomputed=recomputed)

                            t_switch = [item[1] for item in switches][-1]
                            results_event_error[sweeper_cls_name][M][dt][use_SE][alpha] = abs(t_switch_exact - t_switch)

                            restarts = get_sorted(stats, type='restart', sortby='time', recomputed=None)
                            sum_restarts = sum([item[1] for item in restarts])
                            results_state_function[sweeper_cls_name][M][dt][use_SE][alpha]['restarts'] = sum_restarts

                            switches_all = get_sorted(stats, type='switch_all', sortby='time', recomputed=None)
                            t_switches_all = [item[1] for item in switches_all]
                            event_error_all = [abs(t_switch_exact - t_switch) for t_switch in t_switches_all]
                            results_event_error_restarts[sweeper_cls_name][M][dt][use_SE][alpha]['event_error_all'] = event_error_all
                            h_val_all = get_sorted(stats, type='h_all', sortby='time', recomputed=None)
                            results_event_error_restarts[sweeper_cls_name][M][dt][use_SE][alpha]['h_max_event'] = [item[1] for item in h_val_all]

    plot_functions_over_time(
        results_error_over_time, prob_class_name, r'Global error $|y(t) - y_{ex}(t)|$', 'upper left', dt_fix, alpha_fix
    )
    plot_error_norm(results_error_norm, prob_class_name, alpha_fix)
    plot_state_function_detection(
        results_state_function, prob_class_name, r'Absolute value of $h$ $|h(y(T))|$', 'lower left', alpha_fix
    )
    plot_event_time_error(results_event_error, prob_class_name, alpha_fix)
    plot_event_time_error_before_restarts(results_event_error_restarts, prob_class_name, dt_fix, alpha_fix)
    plot_compare_alpha(results_state_function, results_event_error, prob_class_name)


def make_plots_for_test_DAE_numerical_comparison():
    """
    Generates the plots for the discontinuous test DAE where SDC is compared with RK methods. Here, SDC is compared with

        - EDIRK4: DIRK with of order 4,
        - Trapezoidal rule: famous trapezoidal rule often used in engineering.

    This function contains all important parameters for the comparison.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    problem_class = DiscontinuousTestDAE
    prob_class_name = DiscontinuousTestDAE.__name__

    nnodes, dt_list, use_detection, alphas = get_dict_keys(prob_class_name)

    sweeper_classes = [fully_implicit_DAE, EDIRK4DAE, TrapezoidalRuleDAE]
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'

    hook_class = [LogGlobalErrorPostStepDifferentialVariable, LogEventDiscontinuousTestDAE, LogRestarts]

    max_restarts = 200
    epsilon_SE = 1e-10
    typeFD = 'backward'

    alpha_fix = alphas[0]#alphas[4]

    t0 = 3.5
    Tend = 5.0

    dt_fix = dt_list#[-2]

    recomputed = False

    results_error_over_time = {}
    results_error_norm = {}
    results_state_function = {}
    results_event_error = {}

    dt_list = [dt_list] if isinstance(dt_list, float) else dt_list
    for sweeper in sweeper_classes:
        sweeper_cls_name = sweeper.__name__
        results_error_over_time[sweeper_cls_name], results_error_norm[sweeper_cls_name] = {}, {}
        results_state_function[sweeper_cls_name], results_event_error[sweeper_cls_name] = {}, {}

        for M in nnodes:
            results_error_over_time[sweeper_cls_name][M], results_error_norm[sweeper_cls_name][M] = {}, {}
            results_state_function[sweeper_cls_name][M], results_event_error[sweeper_cls_name][M] = {}, {}

            for dt in dt_list:
                results_error_over_time[sweeper_cls_name][M][dt], results_error_norm[sweeper_cls_name][M][dt] = {}, {}
                results_state_function[sweeper_cls_name][M][dt], results_event_error[sweeper_cls_name][M][dt] = {}, {}

                for use_SE in use_detection:
                    results_error_over_time[sweeper_cls_name][M][dt][use_SE], results_error_norm[sweeper_cls_name][M][dt][use_SE] = {}, {}
                    results_state_function[sweeper_cls_name][M][dt][use_SE], results_event_error[sweeper_cls_name][M][dt][use_SE] = {}, {}

                    for alpha in alphas:
                        results_error_over_time[sweeper_cls_name][M][dt][use_SE][alpha], results_error_norm[sweeper_cls_name][M][dt][use_SE][alpha] = {}, {}
                        results_state_function[sweeper_cls_name][M][dt][use_SE][alpha], results_event_error[sweeper_cls_name][M][dt][use_SE][alpha] = {}, {}

                        maxiter = 60 if sweeper_cls_name == 'fully_implicit_DAE' else 1
                        tol_hybr = 1e-6 if sweeper_cls_name == 'fully_implicit_DAE' else 1e-12
                        restol = 1e-13 if sweeper_cls_name == 'fully_implicit_DAE' else -1

                        problem_params = dict()
                        problem_params['newton_tol'] = tol_hybr

                        description, controller_params, controller = generateDescription(
                            dt=dt,
                            problem=problem_class,
                            sweeper=sweeper,
                            num_nodes=M,
                            quad_type=quad_type,
                            QI=QI,
                            hook_class=hook_class,
                            use_adaptivity=False,
                            use_switch_estimator=use_SE,
                            problem_params=problem_params,
                            restol=restol,
                            maxiter=maxiter,
                            max_restarts=max_restarts,
                            tol_event=epsilon_SE,
                            alpha=alpha,
                            typeFD=typeFD,
                        )

                        stats, t_switch_exact = controllerRun(
                            description, controller_params, controller, t0, Tend, exact_event_time_avail=True
                        )

                        err_val = get_sorted(stats, type='e_global_differential_post_step', sortby='time', recomputed=recomputed)
                        results_error_over_time[sweeper_cls_name][M][dt][use_SE][alpha] = err_val

                        err_norm = max([item[1] for item in err_val])
                        results_error_norm[sweeper_cls_name][M][dt][use_SE][alpha] = err_norm

                        h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
                        h_abs = abs([item[1] for item in h_val][-1])
                        results_state_function[sweeper_cls_name][M][dt][use_SE][alpha]['h_abs'] = h_abs
                        print(sweeper_cls_name, M, dt, use_SE, alpha)
                        if use_SE:
                            switches = get_sorted(stats, type='switch', sortby='time', recomputed=recomputed)
                            t_switches = [item[1] for item in switches]

                            if len(t_switches) >= 1:
                                t_switch = t_switches[-1]
                                results_event_error[sweeper_cls_name][M][dt][use_SE][alpha] = abs(t_switch_exact - t_switch)
                            else:
                                results_event_error[sweeper_cls_name][M][dt][use_SE][alpha] = 1.0

                            restarts = get_sorted(stats, type='restart', sortby='time', recomputed=None)
                            sum_restarts = sum([item[1] for item in restarts])
                            results_state_function[sweeper_cls_name][M][dt][use_SE][alpha]['restarts'] = sum_restarts

    
    plot_comparison_event_time_state_function(results_state_function, results_event_error, prob_class_name)


def make_plots_for_WSCC9_test_case(cwd='./'):  # pragma: no cover
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

    Path("data").mkdir(parents=True, exist_ok=True)

    problem_class = WSCC9BusSystem
    prob_class_name = WSCC9BusSystem.__name__

    nnodes, dt_list, use_detection, alphas = get_dict_keys(prob_class_name)

    sweeper_cls_name = [fully_implicit_DAE]
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    maxiter = 50
    tol_hybr = 1e-10
    restol = 5e-13

    hook_class = [LogEventWSCC9, LogRestarts]

    problem_params = dict()
    problem_params['newton_tol'] = tol_hybr

    max_restarts = 400
    epsilon_SE = 1e-10

    alpha_fix = alphas[0]

    t0 = 0.0
    Tend = 0.7

    dt_list = [1 / (2**m) for m in range(5, 11)]
    dt_fix = dt_list[-3]

    recomputed = False

    results_state_function_over_time = {}
    results_state_function_detection = {}
    for sweeper in sweeper_cls_name:
        sweeper_cls_name = sweeper.__name__
        results_state_function_over_time[sweeper_cls_name] = {}
        results_state_function_detection[sweeper_cls_name] = {}

        for M in nnodes:
            results_state_function_over_time[sweeper_cls_name][M], results_state_function_detection[sweeper_cls_name][M] = {}, {}

            for dt in dt_list:
                results_state_function_over_time[sweeper_cls_name][M][dt], results_state_function_detection[sweeper_cls_name][M][dt] = {}, {}

                for use_SE in use_detection:
                    results_state_function_over_time[sweeper_cls_name][M][dt][use_SE], results_state_function_detection[sweeper_cls_name][M][dt][use_SE] = (
                            {},
                            {},
                        )

                    for alpha in alphas:
                        results_state_function_over_time[sweeper_cls_name][M][dt][use_SE][alpha], results_state_function_detection[sweeper_cls_name][M][dt][use_SE][alpha] = (
                            {},
                            {},
                        )

                        description, controller_params = generateDescription(
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

                        # ---- either solution is computed or it is loaded from .dat file already created ----
                        path = Path('data/{}_M={}_dt={}_useSE={}.dat'.format(prob_class_name, M, dt, use_SE))
                        if path.is_file():
                            f = open(cwd + 'data/{}_M={}_dt={}_useSE={}.dat'.format(prob_class_name, M, dt, use_SE), 'rb')
                            stats = dill.load(f)
                            f.close()
                        else:
                            stats, _ = controllerRun(description, controller_params, t0, Tend)

                            fname = 'data/{}_M={}_dt={}_useSE={}.dat'.format(prob_class_name, M, dt, use_SE)
                            f = open(fname, 'wb')
                            dill.dump(stats, f)
                            f.close()

                        h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
                        results_state_function_over_time[sweeper_cls_name][M][dt][use_SE][alpha] = h_val

                        h_abs_end = abs([me[1] for me in h_val][-1])
                        results_state_function_detection[sweeper_cls_name][M][dt][use_SE][alpha]['h_abs'] = h_abs_end

                        if use_SE:
                            restarts = get_sorted(stats, type='restart', sortby='time', recomputed=None)
                            sum_restarts = sum([me[1] for me in restarts])
                            results_state_function_detection[sweeper_cls_name][M][dt][use_SE][alpha]['restarts'] = sum_restarts

    plot_functions_over_time(
        results_state_function_over_time,
        prob_class_name,
        r'Absolute value of $h$ $|h(P_{SV,0}(t))|$',
        'lower left',
        dt_fix,
        alpha_fix,
    )
    plot_state_function_detection(
        results_state_function_detection, prob_class_name, r'Absolute value of $h$ $|h(P_{SV,0}(T))|$', 'upper right', alpha_fix
    )


def get_dict_keys(prob_class):
    """
    Returns keys for dict for numerical solution.

    Parameters
    ----------
    prob_class : str
        Name of problem class.

    Returns
    -------
    nnodes : list
        List of number of collocation nodes.
    dt_list : list
        List of step sizes.
    use_detection : list
        List of using event detection or not.
    alphas : list
        List of factor alpha.
    """

    nnodes = [5]#[3, 4, 5]
    use_detection = [False, True]

    if prob_class == 'DiscontinuousTestDAE':
        dt_list = np.logspace(-3.0, 0.0, num=8)#[1 / (2**m) for m in range(2, 9)]
        alphas = [0.96] #np.arange(0.9, 1.02, 0.02)
    elif prob_class == 'WSCC9BusSystem':
        dt_list = [1 / (2**m) for m in range(5, 11)]
        alphas = [0.95]
    else:
        raise NotImplementedError()

    return nnodes, dt_list, use_detection, alphas
    


def plot_styling_stuff(prob_class, color_type='M'):  # pragma: no cover
    """
    Implements all the stuff needed for making the plots more pretty.
    """

    if color_type == 'M':
        colors = {
            2: 'limegreen',
            3: 'tomato',
            4: 'deepskyblue',
            5: 'orchid',
        }

        markers = {
            2: 'X',#'s',
            3: 'X',#'o',
            4: 'X',#'*',
            5: 'X',#'d',
        }
    elif color_type == 'sweeper':
        colors = {
            'fully_implicit_DAE': 'deepskyblue',
            'EDIRK4DAE': 'forestgreen',
            'TrapezoidalRuleDAE': 'darkorange',
        }

        markers = {
            'fully_implicit_DAE': 'o',
            'EDIRK4DAE': 'o',
            'TrapezoidalRuleDAE': 'o',
        }

    if prob_class == 'DiscontinuousTestDAE':
        xytext = {
            2: (-15.0, 16.5),
            3: (-2.0, 55),
            4: (-13.0, -27),
            5: (-1.0, -40),
        }
    elif prob_class == 'WSCC9BusSystem':
        xytext = {
            2: (-13.0, 16),
            3: (-13.0, 30),
            4: (-13.0, -17),
            5: (-1.0, -38),
        }
    else:
        raise ParameterError(f"For {prob_class} no dictionary for position of data points is set up!")

    return colors, markers, xytext


def plot_functions_over_time(
    results_function_over_time, prob_class, y_label, loc_legend, dt_fix=None, alpha_fix=None
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
    dt_fix : float, optional
        If it is set to a considered step size, only one plot will generated.
    alpha_fix : float, optional
        If it is set to a specific alpha, only one plot will generated.
    """

    colors, _, _ = plot_styling_stuff(prob_class)
    x0 = 3.5 if prob_class == 'DiscontinuousTestDAE' else 0.5

    nnodes, dt_list, use_detection, alphas = get_dict_keys(prob_class)
    dt_list = [dt_fix] if dt_fix is not None else dt_list
    alphas = [alpha_fix] if alpha_fix is not None else alphas
    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 6))
        for M in nnodes:
            for use_SE in use_detection:
                for alpha in alphas:
                    err_val = results_function_over_time['fully_implicit_DAE'][M][dt][use_SE][alpha]
                    t, err = [item[0] for item in err_val], [abs(item[1]) for item in err_val]

                    linestyle_detection = 'solid' if not use_SE else 'dashdot'
                    (line,) = ax.plot(t, err, color=colors[M], linestyle=linestyle_detection, linewidth=0.9)

                    if not use_SE:
                        line.set_label(r'$M={}$'.format(M))

                    if M == 5:  # dummy plot for more pretty legend
                        ax.plot(x0, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-15, 1e1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'Time $t$', fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=True, fontsize=12, loc=loc_legend)
        ax.minorticks_off()
        ax.set_facecolor('#D3D3D3')

        if prob_class == 'DiscontinuousTestDAE':
            file_name = 'data/test_DAE_error_over_time_dt{}.png'.format(dt)
        elif prob_class == 'WSCC9BusSystem':
            file_name = 'data/wscc9_state_function_over_time_dt{}.png'.format(dt)
        else:
            raise ParameterError(f"For {prob_class} no file name is implemented!")

        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_error_norm(results_error_norm, prob_class, alpha_fix=None):  # pragma: no cover
    """
    Plots the error norm for different step sizes and different number of collocation nodes in comparison
    with detection and not.

    Parameters
    ----------
    results_error_norm : dict
        Statistics containing the error norms and sum of restarts for all considered coll. nodes.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    alpha_fix : float, optional
        Factor alpha considered.
    """

    colors, markers, xytext = plot_styling_stuff(prob_class)

    nnodes, dt_list, use_detection, alphas = get_dict_keys(prob_class)

    alphas = [alpha_fix] if alpha_fix is not None else alphas
    dt_list = [dt_list] if isinstance(dt_list, float) else dt_list

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 6))
    for M in nnodes:
        for use_SE in use_detection:
            for alpha in alphas:
                err_norm_dt = [results_error_norm['fully_implicit_DAE'][M][k][use_SE][alpha] for k in dt_list]

                linestyle_detection = 'solid' if not use_SE else 'dashdot'
                (line,) = ax.loglog(
                    dt_list,
                    err_norm_dt,
                    color=colors[M],
                    linestyle=linestyle_detection,
                    linewidth=0.9,
                    markersize=15,
                    markeredgecolor='k',
                    marker=markers[M],
                )

                if not use_SE:
                    line.set_label(r'$M={}$'.format(M))

                if M == 5:  # dummy plot for more pretty legend
                    ax.plot(0, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-15, 1e3)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Step size $\Delta t$', fontsize=16)
    ax.set_ylabel(r'Error norm $||y(t) - \tilde{y}(t)||_\infty$', fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc='lower right')
    ax.set_facecolor('#D3D3D3')

    fig.savefig('data/test_DAE_error_norms.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_state_function_detection(results_state_function, prob_class, y_label, loc_legend, alpha_fix=None):  # pragma: no cover
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
    alpha_fix : float, optional
        Considered factor alpha.
    """

    colors, markers, xytext = plot_styling_stuff(prob_class)

    nnodes, dt_list, use_detection, alphas = get_dict_keys(prob_class)

    alphas = [alpha_fix] if alpha_fix is not None else alphas
    dt_list = [dt_list] if isinstance(dt_list, float) else dt_list

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 6))
    for M in nnodes:
        for use_SE in use_detection:
            for alpha in alphas:
                h_abs = [results_state_function['fully_implicit_DAE'][M][k][use_SE][alpha]['h_abs'] for k in dt_list]

                linestyle_detection = 'solid' if not use_SE else 'dashdot'
                (line,) = ax.loglog(
                    dt_list,
                    h_abs,
                    color=colors[M],
                    linestyle=linestyle_detection,
                    linewidth=0.9,
                    markersize=15,
                    markeredgecolor='k',
                    marker=markers[M],
                )

                if not use_SE:
                    line.set_label(r'$M={}$'.format(M))

                if use_SE:
                    sum_restarts = [results_state_function['fully_implicit_DAE'][M][k][use_SE][alpha]['restarts'] for k in dt_list]
                    for m in range(len(dt_list)):
                        ax.annotate(
                            sum_restarts[m],
                            (dt_list[m], h_abs[m]),
                            xytext=xytext[M],
                            textcoords="offset points",
                            color=colors[M],
                            fontsize=16,
                        )

            if M == 5:  # dummy plot for more pretty legend
                ax.plot(0, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-17, 1e3)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Step size $\Delta t$', fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc=loc_legend)
    ax.set_facecolor('#D3D3D3')

    if prob_class == 'DiscontinuousTestDAE':
        file_name = 'data/test_DAE_state_function.png'
    elif prob_class == 'WSCC9BusSystem':
        file_name = 'data/wscc9_state_function_detection.png'
    else:
        raise ParameterError(f"For {prob_class} no file name is set up!")

    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_event_time_error(results_event_error, prob_class, alpha_fix=None):  # pragma: no cover
    """
    Plots the error between event time founded by detection and exact event time.

    Parameters
    ----------
    results_event_error : dict
        Contains event time error for each considered number of coll. nodes, step size and
        event detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    alpha_fix : float, optional
        Considered factor alpha.
    """

    colors, markers, _ = plot_styling_stuff(prob_class)

    nnodes, dt_list, use_detection, alphas = get_dict_keys(prob_class)
    dt_list = [dt_list] if isinstance(dt_list, float) else dt_list

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 6))
    for M in nnodes:
        for use_SE in [True]:
            for alpha in alphas:
                event_error = [results_event_error['fully_implicit_DAE'][M][k][use_SE][alpha] for k in dt_list]

                linestyle_detection = 'solid' if not use_SE else 'dashdot'
                ax.loglog(
                    dt_list,
                    event_error,
                    color=colors[M],
                    linestyle=linestyle_detection,
                    marker=markers[M],
                    markersize=15,
                    markeredgecolor='k',
                    label=r'$M={}$'.format(M),
                )

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-15, 1e1)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Step size $\Delta t$', fontsize=16)
    ax.set_ylabel(r'Event time error $|t^*_{ex} - t^*_{SE}|$', fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc='lower right')
    ax.set_facecolor('#D3D3D3')

    fig.savefig('data/test_DAE_event_time_error.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_event_time_error_before_restarts(results_event_error_restarts, prob_class, dt_fix=None, alpha_fix=None):  # pragma: no cover
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
    alpha_fix : float, optional
        Factor alpha considered.
    """

    colors, markers, _ = plot_styling_stuff(prob_class)

    nnodes, dt_list, use_detection, alphas = get_dict_keys(prob_class)

    dt_list = [dt_list] if isinstance(dt_list, float) else dt_list
    dt_list = [dt_fix] if dt_fix is not None else dt_list
    alphas = [alpha_fix] if alpha_fix is not None else alphas
    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 6))
        h_ax = ax.twinx()
        for M in nnodes:
            for use_SE in use_detection:
                for alpha in alphas:
                    if use_SE:
                        event_error_all = results_event_error_restarts['fully_implicit_DAE'][M][dt][use_SE][alpha]['event_error_all']

                        (line,) = ax.semilogy(
                            np.arange(1, len(event_error_all) + 1),
                            event_error_all,
                            color=colors[M],
                            linestyle='solid',
                            # marker=markers[M],
                        )

                        line.set_label(r'$M={}$'.format(M))

                        h_max_event = results_event_error_restarts['fully_implicit_DAE'][M][dt][use_SE][alpha]['h_max_event']
                        h_ax.semilogy(
                            np.arange(1, len(h_max_event) + 1),
                            h_max_event,
                            color=colors[M],
                            linestyle='dashdot',
                            linewidth=0.9,
                            marker='o',
                            markeredgecolor='k',
                            markersize=7,
                            alpha=0.5,
                        )

                        if M == nnodes[-1]:  # dummy plot for more pretty legend
                            ax.plot(
                                1, event_error_all[0], color='black', linestyle='solid', label=r'$|t^*_{ex} - t^*_{SE}|$'
                            )
                            ax.plot(
                                1,
                                1e2,
                                color='black',
                                linestyle='dashdot',
                                linewidth=0.9,
                                marker='o',
                                markeredgecolor='k',
                                markersize=7,
                                alpha=0.5,
                                label=r'$||h(t)||_\infty$',
                            )

        h_ax.tick_params(labelsize=16)
        h_ax.set_ylim(1e-15, 1e0)
        h_ax.set_yscale('log', base=10)
        h_ax.set_ylabel(r'Maximum value of h $||h(t)||_\infty$', fontsize=16)
        h_ax.minorticks_off()

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-15, 1e-1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'Restarted steps $n_{restart}$', fontsize=16)
        ax.set_ylabel(r'Event time error $|t^*_{ex} - t^*_{SE}|$', fontsize=16)
        ax.grid(visible=True)
        ax.minorticks_off()
        ax.legend(frameon=True, fontsize=12, loc='lower left')
        ax.set_facecolor('#D3D3D3')

        fig.savefig('data/test_DAE_event_time_error_restarts_dt{}.png'.format(dt), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_compare_alpha(results_state_function, results_event_error, prob_class, dt_fix=None, alpha_fix=None):
    r"""
    Plots the state function and the event time error for different alpha's.

    Parameters
    ----------
    results_state_function : dict
        Contains absolute values of state function for different M, different step sizes
        and event detection or not.
    results_event_error : dict
        Contains all events for each considered number of coll. nodes, step size and
        event detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    dt_fix : float, optional
        Step size considered.
    alpha_fix : float, optional
        Factor alpha considered.
    """

    colors, markers, _ = plot_styling_stuff(prob_class)

    nnodes, dt_list, _, alphas = get_dict_keys(prob_class)

    dt_list = [dt_list] if isinstance(dt_list, float) else dt_list
    dt_list = [dt_fix] if dt_fix is not None else dt_list
    alphas = [alpha_fix] if alpha_fix is not None else alphas

    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 6))
        h_ax = ax.twinx()
        for M in nnodes:
            for use_SE in [True]:
                event_error_alpha = [results_event_error['fully_implicit_DAE'][M][dt][use_SE][alpha] for alpha in alphas]

                (line,) = ax.plot(
                    alphas,
                    event_error_alpha,
                    color=colors[M],
                    linestyle='solid',
                    # marker=markers[M],
                )

                line.set_label(r'$M={}$'.format(M))
                print(alphas)
                state_function_alpha = [results_state_function['fully_implicit_DAE'][M][dt][use_SE][alpha]['h_abs'] for alpha in alphas]
                print(state_function_alpha)
                h_ax.plot(
                    alphas,
                    state_function_alpha,
                    color=colors[M],
                    linestyle='dashdot',
                    marker=markers[M],
                    markeredgecolor='k',
                    markersize=15,
                    alpha=0.6,
                )

                if M == nnodes[-1]:  # dummy plot for more pretty legend
                    ax.plot(
                        1, event_error_alpha[0], color='black', linestyle='solid', label=r'$|t^*_{ex} - t^*_{SE}|$'
                    )
                    ax.plot(
                        1,
                        1e2,
                        color='black',
                        linestyle='dashdot',
                        marker=markers[M],
                        markeredgecolor='k',
                        markersize=15,
                        alpha=0.6,
                        label=r'$||h(t)||_\infty$',
                    )

        h_ax.tick_params(labelsize=16)
        h_ax.set_ylim(1e-15, 1e0)
        h_ax.set_yscale('log', base=10)
        h_ax.set_ylabel(r'Maximum value of h $||h(t)||_\infty$', fontsize=16)
        h_ax.minorticks_off()

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-15, 1e-1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'Factor $\alpha$', fontsize=16)
        ax.set_ylabel(r'Event time error $|t^*_{ex} - t^*_{SE}|$', fontsize=16)
        ax.grid(visible=True)
        ax.minorticks_off()
        ax.legend(frameon=True, fontsize=12, loc='upper right')
        ax.set_facecolor('#D3D3D3')

        fig.savefig('data/test_DAE_compare_alphas_dt{}.png'.format(dt), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_comparison_event_time_state_function(results_state_function, results_event_error, prob_class, alpha_fix=None):
    """
    Plots the event time error and the state function for different sweeper. Note
    that this function can be embedded into the other functions.

    Parameters
    ----------
    results_state_function : dict
        Contains absolute values of state function for different M, different step sizes
        and event detection or not.
    results_event_error : dict
        Contains all events for each considered number of coll. nodes, step size and
        event detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    """

    colors, markers, _ = plot_styling_stuff(prob_class, color_type='sweeper')

    nnodes, dt_list, _, alphas = get_dict_keys(prob_class)

    dt_list = [dt_list] if isinstance(dt_list, float) else dt_list
    alphas = [alpha_fix] if alpha_fix is not None else alphas

    for alpha in alphas:
        for M in nnodes:
            fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 6))
            h_ax = ax.twinx()
            for sweeper_cls_name in ['fully_implicit_DAE', 'EDIRK4DAE', 'TrapezoidalRuleDAE']:
                if sweeper_cls_name == 'fully_implicit_DAE':
                    sweeper_label = 'SDC'
                elif sweeper_cls_name == 'EDIRK4DAE':
                    sweeper_label = 'EDIRK4'
                elif sweeper_cls_name == 'TrapezoidalRuleDAE':
                    sweeper_label = 'Trapezoidal rule'
                else:
                    raise NotImplementedError
                for use_SE in [True]:
                    sweeper_event_error = [results_event_error[sweeper_cls_name][M][dt][use_SE][alpha] for dt in dt_list]

                    (line,) = ax.loglog(
                        dt_list,
                        sweeper_event_error,
                        color=colors[sweeper_cls_name],
                        linestyle='solid',
                        # marker=markers[M],
                    )

                    line.set_label(f'{sweeper_label}')
                    print(dt_list, sweeper_event_error)
                    for m in range(len(sweeper_event_error)):
                        if sweeper_event_error[m] == 1.0:
                            ax.annotate(
                                'X',
                                (dt_list[m], sweeper_event_error[m]),
                                xytext=(-13.0, 10),
                                textcoords="offset points",
                                color=colors[sweeper_cls_name],
                                fontsize=10,
                            )

                    sweeper_state_function = [results_state_function[sweeper_cls_name][M][dt][use_SE][alpha]['h_abs'] for dt in dt_list]
                    print(sweeper_state_function)
                    h_ax.loglog(
                        dt_list,
                        sweeper_state_function,
                        color=colors[sweeper_cls_name],
                        linestyle='dashdot',
                        marker=markers[sweeper_cls_name],
                        markeredgecolor='k',
                        markersize=15,
                        alpha=0.6,
                    )

                    if sweeper_cls_name == 'TrapezoidalRuleDAE':  # dummy plot for more pretty legend
                        ax.plot(
                            1, sweeper_event_error[0], color='black', linestyle='solid', label=r'$|t^*_{ex} - t^*_{SE}|$'
                        )
                        ax.plot(
                            1,
                            1e2,
                            color='black',
                            linestyle='dashdot',
                            marker=markers[sweeper_cls_name],
                            markeredgecolor='k',
                            markersize=15,
                            alpha=0.6,
                            label=r'$|h(y(T))|$',
                        )
            h_ax.tick_params(labelsize=16)
            h_ax.set_ylim(1e-15, 1e1)
            h_ax.set_yscale('log', base=10)
            h_ax.set_ylabel(r'Absolute value of h $|h(y(T))|$', fontsize=16)
            h_ax.minorticks_off()
    
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_ylim(1e-15, 1e1)
            ax.set_yscale('log', base=10)
            ax.set_xlabel(r'Step size $\Delta t$', fontsize=16)
            ax.set_ylabel(r'Event time error $|t^*_{ex} - t^*_{SE}|$', fontsize=16)
            ax.grid(visible=True)
            ax.minorticks_off()
            ax.legend(frameon=True, fontsize=12, loc='upper left')
            ax.set_facecolor('#D3D3D3')
    
            fig.savefig(f'data/test_DAE_comparison_event_time_state_function_M={M}_alpha={alpha}.png', dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


if __name__ == "__main__":
    # make_plots_for_test_DAE()
    make_plots_for_test_DAE_numerical_comparison()
    # make_plots_for_WSCC9_test_case()
