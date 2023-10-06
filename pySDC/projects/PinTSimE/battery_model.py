import numpy as np
from pathlib import Path

from pySDC.helpers.stats_helper import sort_stats, filter_stats, get_sorted
from pySDC.implementations.problem_classes.Battery import battery, battery_implicit, battery_n_capacitors
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

import pySDC.helpers.plot_helper as plt_helper

from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_step_size import LogStepSize
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

from pySDC.projects.PinTSimE.hardcoded_solutions import testSolution


class LogEventBattery(hooks):
    """
    Logs the problem dependent state function of the battery drain model.
    """

    def post_step(self, step, level_number):
        super(LogEventBattery, self).post_step(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='state_function',
            value=L.uend[1:] - P.V_ref[:],
        )


def generateDescription(
    dt,
    problem,
    sweeper,
    num_nodes,
    quad_type,
    QI,
    hook_class,
    use_adaptivity,
    use_switch_estimator,
    problem_params,
    restol,
    maxiter,
    max_restarts=None,
    tol_event=1e-10,
    alpha=1.0,
):
    r"""
    Generate a description for the battery models for a controller run.

    Parameters
    ----------
    dt : float
        Time step for computation.
    problem : pySDC.core.Problem
        Problem class that wants to be simulated.
    sweeper : pySDC.core.Sweeper
        Sweeper class for solving the problem class numerically.
    num_nodes : int
        Number of collocation nodes.
    quad_type : str
        Type of quadrature nodes, e.g. ``'LOBATTO'`` or ``'RADAU-RIGHT'``.
    QI : str
        Type of preconditioner used in SDC, e.g. ``'IE'`` or ``'LU'``.
    hook_class : List of pySDC.core.Hooks
        Logged data for a problem, e.g., hook_class=[LogSolution] logs the solution ``'u'``
        during the simulation.
    use_adaptivity : bool
        Flag if the adaptivity wants to be used or not.
    use_switch_estimator : bool
        Flag if the switch estimator wants to be used or not.
    problem_params : dict
        Dictionary containing the problem parameters.
    restol : float
        Residual tolerance to terminate.
    maxiter : int
        Maximum number of iterations to be done.
    max_restarts : int, optional
        Maximum number of restarts per step.
    tol_event : float, optional
        Tolerance for event detection to terminate.
    alpha : float, optional
        Factor that indicates how the new step size in the Switch Estimator is reduced.

    Returns
    -------
    description : dict
        Contains all information for a controller run.
    controller_params : dict
        Parameters needed for a controller run.
    """

    # initialize level parameters
    level_params = {
        'restol': -1 if use_adaptivity else restol,
        'dt': dt,
    }
    if use_adaptivity:
        assert restol == -1, "Please set restol to -1 or omit it"

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': quad_type,
        'num_nodes': num_nodes,
        'QI': QI,
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': maxiter,
    }
    assert 'errtol' not in step_params.keys(), 'No exact solution known to compute error'

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': hook_class,
        'mssdc_jac': False,
    }

    # convergence controllers
    convergence_controllers = {}
    if use_switch_estimator:
        switch_estimator_params = {
            'tol': tol_event,
            'alpha': alpha,
        }
        convergence_controllers.update({SwitchEstimator: switch_estimator_params})
    if use_adaptivity:
        adaptivity_params = {
            'e_tol': 1e-7,
        }
        convergence_controllers.update({Adaptivity: adaptivity_params})
    if max_restarts is not None:
        restarting_params = {
            'max_restarts': max_restarts,
            'crash_after_max_restarts': False,
        }
        convergence_controllers.update({BasicRestartingNonMPI: restarting_params})

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': problem,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }

    return description, controller_params


def controllerRun(description, controller_params, t0, Tend, exact_event_time_avail=None):
    """
    Executes a controller run for a problem defined in the description.

    Parameters
    ----------
    description : dict
        Contains all information for a controller run.
    controller_params : dict
        Parameters needed for a controller run.
    t0 : float
        Staring time of simulation.
    Tend : float
        End time of simulation.
    exact_event_time_avail : bool, optional
        Indicates if exact event time of a problem is available.

    Returns
    -------
    stats : dict
        Raw statistics from a controller run.
    """

    # ---- assume if it is set to False then no event time is available ----
    if exact_event_time_avail is not None:
        if not exact_event_time_avail:
            exact_event_time_avail = None
        else:
            pass

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    t_switch_exact = P.t_switch_exact if exact_event_time_avail is not None else None

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, t_switch_exact


def main():
    r"""
    Executes the simulation.

    Note
    ----
    Hardcoded solutions for battery models in `pySDC.projects.PinTSimE.hardcoded_solutions` are only computed for
    ``dt_list=[1e-2, 1e-3]`` and ``M_fix=4``. Hence changing ``dt_list`` and ``M_fix`` to different values could arise
    an ``AssertionError``.
    """

    # --- defines parameters for sweeper ----
    M_fix = 4
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': 'LOBATTO',
        'QI': 'IE',
    }

    # --- defines parameters for event detection ----
    handling_params = {
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    # ---- all parameters are stored in this dictionary ---- 
    all_params = {
        'sweeper_params': sweeper_params,
        'handling_params': handling_params,
    }

    hook_class = [LogSolution, LogEventBattery, LogEmbeddedErrorEstimate, LogStepSize]

    use_detection = [True]
    use_adaptivity = [True]

    for problem, sweeper in zip([battery, battery_implicit], [imex_1st_order, generic_implicit]):

        for defaults in [False, True]:
            # ---- for hardcoded solutions problem patameter defaults should match with parameters here ----
            if defaults:
                params_battery_1capacitor = {
                    'ncapacitors': 1,
                }
            else:
                params_battery_1capacitor = {
                    'ncapacitors': 1,
                    'C': np.array([1.0]),
                    'alpha': 1.2,
                    'V_ref': np.array([1.0]),
                }

            all_params.update({'problem_params': params_battery_1capacitor}) 

            _ = runSimulation(
                problem=problem,
                sweeper=sweeper,
                all_params=all_params,
                use_adaptivity=use_adaptivity,
                use_detection=use_detection,
                hook_class=hook_class,
                interval=(0.0, 0.3),
                dt_list=[1e-2, 1e-3],
                nnodes=[M_fix],
            )

    # --- defines parameters for the problem class ----
    params_battery_2capacitors = {
        'ncapacitors': 2,
        'C': np.array([1.0, 1.0]),
        'alpha': 1.2,
        'V_ref': np.array([1.0, 1.0]),
    }

    all_params.update({'problem_params': params_battery_2capacitors})

    _ = runSimulation(
        problem=battery_n_capacitors,
        sweeper=imex_1st_order,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(0.0, 0.5),
        dt_list=[1e-2, 1e-3],
        nnodes=[sweeper_params['num_nodes']],
    )


def runSimulation(problem, sweeper, all_params, use_adaptivity, use_detection, hook_class, interval, dt_list, nnodes):
    r"""
    Script that executes the simulation for a given problem class for given parameters defined by the user.

    Parameters
    ----------
    problem : pySDC.core.Problem
        Problem class to be simulated.
    sweeper : pySDC.core.Sweeper
        Sweeper that is used to simulate the problem class.
    all_params : dict
        Dictionary contains the problem parameters for ``problem``, the sweeper parameters for ``sweeper``,
        and handling parameters needed for event detection, i.e., ``max_restarts``, ``recomputed``, ``tol_event``,
        ``alpha``, and ``exact_event_time_available``.
    use_adaptivity : list of bool
       Indicates whether adaptivity is used in the simulation or not. Here a list is used to iterate over the
       different cases, i.e., ``use_adaptivity=[True, False]``.
    use_detection : list of bool
       Indicates whether event detection is used in the simulation or not. Here a list is used to iterate over the
       different cases, i.e., ``use_detection=[True, False]``.
    hook_class : list of pySDC.core.Hooks
       List containing the different hook classes to log data during the simulation, i.e., ``hook_class=[LogSolution]``
       logs the solution ``u``.
    interval : tuple
       Simulation interval.
    dt_list : list of float
       List containing different step sizes where the solution is computed.
    nnodes : list of int
       The solution can be computed for different number of collocation nodes.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    prob_cls_name = problem.__name__
    unknowns = {
        'battery': ['i_L', 'v_C'],
        'battery_implicit': ['i_L', 'v_C'],
        'battery_n_capacitors': ['i_L', 'v_C1', 'v_C2'],
        'DiscontinuousTestODE': ['u'],
    }

    maxiter = 8
    restol = -1

    u_num = {}

    for dt in dt_list:
        u_num[dt] = {}

        for M in nnodes:
            u_num[dt][M] = {}

            for use_SE in use_detection:
                u_num[dt][M][use_SE] = {}

                for use_A in use_adaptivity:
                    u_num[dt][M][use_SE][use_A] = {}

                    problem_params = all_params['problem_params']
                    sweeper_params = all_params['sweeper_params']
                    handling_params = all_params['handling_params']

                    M_fix = sweeper_params['num_nodes']
                    assert M_fix in nnodes, f"For fixed number of collocation nodes {M_fix} no solution will be computed!"

                    description, controller_params = generateDescription(
                        dt,
                        problem,
                        sweeper,
                        M,
                        sweeper_params['quad_type'],
                        sweeper_params['QI'],
                        hook_class,
                        use_adaptivity,
                        use_SE,
                        problem_params,
                        restol,
                        maxiter,
                        handling_params['max_restarts'],
                        handling_params['tol_event'],
                        handling_params['alpha'],
                    )

                    stats, t_switch_exact = controllerRun(
                        description=description,
                        controller_params=controller_params,
                        t0=interval[0],
                        Tend=interval[-1],
                        exact_event_time_avail=handling_params['exact_event_time_avail'],
                    )

                    u_num[dt][M][use_SE][use_A] = getDataDict(
                        stats, unknowns[prob_cls_name], use_A, use_SE, handling_params['recomputed'], t_switch_exact
                    )

                    plotSolution(u_num[dt][M][use_SE][use_A], prob_cls_name, use_A, use_SE)

                    testSolution(u_num[dt][M_fix][use_SE][use_A], prob_cls_name, dt, use_A, use_SE)

    return u_num


def plotStylingStuff():
    """
    Returns plot stuff such as colors, line styles for making plots more pretty.
    """

    colors = {
        False: {
            False: 'dodgerblue',
            True: 'navy',
        },
        True: {
            False: 'linegreen',
            True: 'darkgreen',
        },
    }

    return colors


def plotSolution(u_num, prob_cls_name, use_adaptivity, use_detection):  # pragma: no cover
    r"""
    Plots the numerical solution for one simulation run.

    Parameters
    ----------
    u_num : dict
        Contains numerical solution with corresponding times for different problem_classes, and
        labels for different unknowns of the problem.
    prob_cls_name : str
        Name of the problem class to be plotted.
    use_adaptivity : bool
        Indicates whether adaptivity is used in the simulation or not.
    """

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))

    unknowns = u_num['unknowns']
    for unknown in unknowns:
        ax.plot(u_num['t'], u_num[unknown], label=r"${}$".format(unknown))

    if use_detection:
        t_switches = u_num['t_switches']
        for i in range(len(t_switches)):
            ax.axvline(x=t_switches[i], linestyle='--', linewidth=0.8, color='r', label='Event {}'.format(i + 1))

    if use_adaptivity:
        dt_ax = ax.twinx()
        dt = u_num['dt']
        dt_ax.plot(dt[:, 0], dt[:, 1], linestyle='-', linewidth=0.8, color='k', label=r'$\Delta t$')
        dt_ax.set_ylabel(r'$\Delta t$', fontsize=16)
        dt_ax.legend(frameon=False, fontsize=12, loc='center right')

    ax.legend(frameon=False, fontsize=12, loc='upper right')
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(r'$u(t)$', fontsize=16)

    fig.savefig('data/{}_model_solution.png'.format(prob_cls_name), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def getDataDict(stats, unknowns, use_adaptivity, use_detection, recomputed, t_switch_exact):
    r"""
    Extracts statistics and store it in a dictionary. In this routine, from ``stats`` different data are extracted
    such as
    
    - each component of solution ``'u'`` and corresponding time domain ``'t'``,
    - the unknowns of the problem as labels ``'unknowns'``,
    - global error ``'e_global'`` after each step,
    - events found by event detection ``'t_switches''``,
    - exact event time ``'t_switch_exact'``,
    - event error ``'e_event'``,
    - state function ``'state_function'``,
    - embedded error estimate computing when using adaptivity ``'e_em'``,
    - (adjusted) step sizes ``'dt'``,
    - sum over restarts ``'sum_restarts'``,
    - and the sum over all iterations ``'sum_niters'``.

    Note
    ----
    In order to use these data, corresponding hook classes has to be defined before the simulation. Otherwise, no values can
    be obtained.
    
    The global error does only make sense when an exact solution for the problem is available. Since ``'e_global'`` is stored
    for each problem class, only for ``DiscontinuousTestODE`` the global error is taken into account when testing the solution.

    Also the event error ``'e_event'`` can only be computed if an exact event time is available. Since the function
    ``controllerRun`` returns ``t_switch_exact=None`` when no exact event time is available, in order to compute the event error,
    it has to be proven whether the list (in case of more than one event) contains ``None`` or not.

    Parameters
    ----------
    stats : dict
        Raw statistics of one simulation run.
    unknowns : list
        Unknowns of problem as string.
    use_adaptivity : bool
        Indicates whether adaptivity is used in the simulation or not.
    use_detection : bool
        Indicates whether event detection is used or not.
    recomputed : bool
        Indicates if values after successfully steps are used or not.
    t_switch_exact : float
        Exact event time of the problem.

    Returns
    -------
    res : dict
        Dictionary with extracted data separated with reasonable keys.
    """

    res =  {}

    # ---- numerical solution ----
    u_val = get_sorted(stats, type='u', sortby='time', recomputed=recomputed)
    res['t'] = np.array([item[0] for item in u_val])
    for i, label in enumerate(unknowns):
        res[label] = np.array([item[1][i] for item in u_val])

    res['unknowns'] = unknowns

    # ---- global error ----
    res['e_global'] = np.array(get_sorted(stats, type='e_global_post_step', sortby='time', recomputed=recomputed))

    # ---- event time(s) found by event detection ----
    if use_detection:
        switches = getRecomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'No events found!'
        t_switches = [t[1] for t in switches]
        res['t_switches'] = t_switches

        t_switch_exact = [t_switch_exact]
        res['t_switch_exact'] = t_switch_exact

        if not all(t is None for t in t_switch_exact):
            event_err = [abs(num_item - ex_item) for (num_item, ex_item) in zip(res['t_switches'], res['t_switch_exact'])]
            res['e_event'] = event_err

    h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
    h = np.array([np.abs(val[1]) for val in h_val])
    res['state_function'] = h

    # ---- embedded error and adapted step sizes----
    if use_adaptivity:
        res['e_em'] = np.array(
            get_sorted(stats, type='error_embedded_estimate', sortby='time', recomputed=recomputed)
        )
        res['dt'] = np.array(get_sorted(stats, type='dt', recomputed=recomputed))

    # ---- sum over restarts ----
    if use_adaptivity or use_detection:
        res['sum_restarts'] = np.sum(np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))[:, 1])

    # ---- sum over all iterations ----
    res['sum_niters'] = np.sum(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])
    return res


def getRecomputed(stats, type, sortby='time'):
    """
    Function that filters statistics after a recomputation. It stores all value of a type before restart. If there are
    multiple values with same time point, it only stores the elements with unique times.

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    type : str
        The type the be filtered.
    sortby : str, optional
        String to specify which key to use for sorting.

    Returns
    -------
    sorted_list : list
        List of filtered statistics.
    """

    sorted_nested_list = []
    times_unique = np.unique([me[0] for me in get_sorted(stats, type=type)])
    filtered_list = [
        filter_stats(
            stats,
            time=t_unique,
            num_restarts=max([me.num_restarts for me in filter_stats(stats, type=type, time=t_unique).keys()]),
            type=type,
        )
        for t_unique in times_unique
    ]
    for item in filtered_list:
        sorted_nested_list.append(sort_stats(item, sortby=sortby))
    sorted_list = [item for sub_item in sorted_nested_list for item in sub_item]
    return sorted_list


if __name__ == "__main__":
    main()
