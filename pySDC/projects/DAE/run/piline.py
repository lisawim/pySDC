from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
from pySDC.projects.DAE.problems.piline_dae import PilineDAE, PilinePowerOutageDAE
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.BDF_DAE import BDF_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.projects.PinTSimE.battery_model import get_recomputed
from pySDC.helpers.stats_helper import get_list_of_types, get_sorted
import pySDC.helpers.plot_helper as plt_helper


def get_description(dt, nvars, problem_class, hookclass, sweeper, use_detection, newton_tol=1e-12):
    """
    Returns the description for one simulation run.

    Parameters
    ----------
    dt : float
        Time step size for computation.
    nvars : int
        Number of variables of the problem.
    problem_class : pySDC.core.Problem.ptype_dae
        Problem class in DAE formulation that wants to be simulated.
    hookclass : pySDC.core.Hooks
        Hook class to log the data.
    sweeper : pySDC.core.Sweeper
        Sweeper class for solving the problem class.
    use_detection : bool
        Indicate whether switch detection should be used or not.
    newton_tol : float, optional
        Tolerance for solving the nonlinear system of DAE solver.

    Returns
    -------
    description : dict
        Contains all information for a controller run.
    controller_params : dict
        Specific parameters for the controller.
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-11
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['initial_guess'] = 'spread'


    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = newton_tol  # tollerance for implicit solver
    problem_params['nvars'] = nvars

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 25

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hookclass

    convergence_controllers = dict()
    if use_detection:
        switch_estimator_params = {}
        convergence_controllers.update({SwitchEstimator: switch_estimator_params})

    max_restarts = 1
    if max_restarts is not None:
        convergence_controllers[BasicRestartingNonMPI] = {
            'max_restarts': max_restarts,
            'crash_after_max_restarts': False,
        }

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    return description, controller_params


def controller_run(t0, Tend, controller_params, description):
    """
    Executes a controller run for time interval to be specified in the arguments.

    Parameters
    ----------
    t0 : float
        Initial time of simulation.
    Tend : float
        End time of simulation.
    controller_params : dict
        Parameters needed for the controller.
    description : dict
        Contains all information for a controller run.

    Returns
    -------
    stats : dict
        Raw statistics from a controller run.
    """

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    nfev = controller.MS[0].levels[0].sweep.nfev

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0
    f = open('data/PilinePowerOutageDAE_out.txt', 'w')
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %1i' % item
        f.write(out + '\n')
        print(out)
        min_iter = min(min_iter, item[1])
        max_iter = max(max_iter, item[1])

    mean_niter = np.mean([me[1] for me in iter_counts])
    out = 'Mean number of ierations: {}'.format(mean_niter)
    f.write(out + '\n')
    print(out)

    f.close()

    return stats, nfev

def plot_solution(stats, use_detection=False, cwd='./'):
    """
    Plots the solution of one simulation run (i.e., for one time step size).

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    use_detection : bool, optional
        Indicate whether switch detection should be used or not.
    cwd : str
        Current working directory.
    """

    vC1 = np.array([me[1][1] for me in get_sorted(stats, type='approx_solution')])
    vC2 = np.array([me[1][4] for me in get_sorted(stats, type='approx_solution')])
    iLp = np.array([me[1][9] for me in get_sorted(stats, type='approx_solution')])
    vRl = np.array([me[1][5] for me in get_sorted(stats, type='approx_solution')])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution')])

    if use_detection:
        switches = get_recomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'No switches found!'
        t_switch = [v[1] for v in switches]

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title(r'Solution /w switch detection', fontsize=8)
    ax.plot(t, vRl, label=r"Consumer's behavior")
    ax.axvline(x=0.05, linestyle='--', linewidth=0.8, color='r', label='Power outage')
    ax.axhline(y=0.2 * 150.0, linestyle='--', linewidth=0.8, color='k', label='Backup generator')
    if use_detection:
        for m in range(len(t_switch)):
            if m == 0:
                ax.axvline(x=t_switch[m], linestyle='--', linewidth=0.8, color='g', label='{} Event(s) found'.format(len(t_switch)))
            else:
                ax.axvline(x=t_switch[m], linestyle='--', linewidth=0.8, color='g')
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax.set_xlabel('Time[s]', fontsize=8)
    ax.set_ylabel('Voltage[V]', fontsize=8)
    fig.savefig('data/piline_outage_solution.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)

def pack_solution_data(t, u):
    """
    Packs the time together with solution values into a np.array.

    Parameters
    ----------
    t : np.1darray
        Time at which the solution is computed.
    u : np.1darray
        Solution values.

    Returns
    -------
    res_array : np.2darray
        Contains the time together with their solution values
    """

    res_array = np.zeros((2, len(t)))
    for m in range(len(t)):
        res_array[0, m], res_array[1, m] = t[m], u[m]
    return res_array


def diffs_around_event(c, Vs, dt_list, use_detection, results_simulations, results_events):
    """
    Plots the differences of an event around it (i.e., the differences at the time before, at, and after the event).

    Parameters
    ----------
    c : float
        Indicates when the backup generator must restart (used to compute difference at event).
    Vs : float
        Voltage at the voltage source (used to compute difference at event).
    dt_list : list
        Contains multiple time step sizes.
    use_detection : list
        Contains the iterable object for indicating whether a detection of events is used.
    results_simulations : dict
        Results of the solution for each time step size.
    results_events : dict
        Switching results for each time step size.
    """

    diffs_detection = []
    diffs = []
    for use_SE in use_detection:
        for dt_item in dt_list:
            u = results_simulations[use_SE][dt_item]

            t_switches = results_events[use_SE][dt_item]
            t_switch = t_switches[-1]

            if use_SE:
                diff_at_event = 0
                for m in range(u.shape[1]):
                    if np.isclose(u[0, m], t_switch, atol=1e-14):
                        diff_at_event = u[1, m] - c * Vs

                diffs_detection.append(diff_at_event)
            else:
                diff_before_event = 0
                diff_after_event = 0
                for m in range(1, u.shape[1] - 1):
                    if u[0, m - 1] <= t_switch <= u[0, m + 1]:
                        diff_before_event = u[1, m - 1] - c * Vs
                        diff_after_event = u[1, m + 1] - c * Vs

                diffs.append([diff_before_event, diff_after_event])

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title(r'Evaluating state function', fontsize=8)
    ax.plot(dt_list, [diff_item[0] for diff_item in diffs], 'ko--', linewidth=0.8, label='Before Event')
    ax.plot(dt_list, [diff_item for diff_item in diffs_detection], 'ro--', linewidth=0.8, label='At Event')
    ax.plot(dt_list, [diff_item[1] for diff_item in diffs], 'ko-', linewidth=0.8, label='After Event')

    ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax.set_xticks(dt_list)
    ax.set_xticklabels(dt_list)

    ax.set_xscale('log', base=10)
    ax.set_yscale('symlog', linthresh=1e-8)
    ax.set_ylim(-1, 1)

    ax.set_xlabel(r'Time step $\Delta t$', fontsize=8)
    ax.set_ylabel(r'$V_\mathrm{R_\ell}-cV_s$', fontsize=8)

    fig.savefig('data/piline_diffs_around_event.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def diffs_over_time(c, Vs, dt_list, use_detection, results_simulations, results_events):
    """
    Plots the state function over time. It can be investigated how large the error is.

    Parameters
    ----------
    c : float
        Indicates when the backup generator must restart (used to compute difference at event).
    Vs : float
        Voltage at the voltage source (used to compute difference at event).
    dt_list : list
        Contains multiple time step sizes.
    use_detection : list
        Contains the iterable object for indicating whether a detection of events is used.
    results_simulations : dict
        Results of the solution for each time step size.
    results_events : dict
        Switching results for each time step size.
    """

    for dt_item in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax.set_title(r'Evaluating state function over time for $\Delta t=%s$' % dt_item, fontsize=8)
        for use_SE in use_detection:
            u = results_simulations[use_SE][dt_item]

            t_switches = results_events[use_SE][dt_item]
            t_switch = t_switches[-1]

            if use_SE:
                ax.plot(u[0, :], u[1, :] - c * Vs, 'r--', linewidth=0.8, label=r'Detection - {}'.format(use_SE))
            else:
                ax.plot(u[0, :], u[1, :] - c * Vs, 'k-', linewidth=0.8, label=r'Detection - {}'.format(use_SE))

        for m in range(len(t_switches)):
            if m == 0:
                ax.axvline(x=t_switches[m], linestyle='--', linewidth=0.8, color='g', label='{} Event(s) found'.format(len(t_switches)))
            else:
                ax.axvline(x=t_switches[m], linestyle='--', linewidth=0.8, color='g')

        ax.legend(frameon=False, fontsize=8, loc='lower right')

        ax.set_xlim(t_switch - 0.0000001, t_switch + 0.0000001)
        ax.set_ylim(-1, 1)
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.set_xlabel(r'Time[s]', fontsize=8)
        ax.set_ylabel(r'$V_\mathrm{R_\ell}-cV_s$', fontsize=8)

        fig.savefig('data/piline_diffs_over_time_{}.png'.format(dt_item), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_nfev_against_residual(res_norm_against_nfev):
    """
    Plots different newtol's against the residual (max) norm, where the residuals after one step is considered.

    Parameters
    ----------
    res_norm_against_nfev : dict
        Contains the residual norm and corresponding number of function evaluations for one specific newton_tol.
    """

    lists = sorted(res_norm_against_nfev.items())
    newton_tols, res_norm_against_nfev_list = zip(*lists)
    res_norms, nfev = [me[0] for me in res_norm_against_nfev_list], [me[1] for me in res_norm_against_nfev_list]

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(5, 3))
    ax.loglog(newton_tols, res_norms, color='firebrick', marker='.', linewidth=0.8)
    for m in range(len(newton_tols)):
        ax.annotate(nfev[m], (newton_tols[m], res_norms[m]), xytext=(-8.5, 10), textcoords="offset points", fontsize=8)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_ylim(1e-14, 1e-10)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Inner tolerance', fontsize=8)
    ax.set_ylabel(r'$||r_\mathrm{DAE}||_\infty$', fontsize=8)
    ax.grid(visible=True)
    ax.minorticks_off()
    fig.savefig('data/residual_against_tolerances.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_residual_post_step(residual_post_step, dt, newton_tol):
    """
    Plots the residuals after each step for different time stp sizes and newton tolerances.
    """

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(5, 3))
    ax.plot([me[0] for me in residual_post_step],[me[1] for me in residual_post_step])
    fig.savefig('data/residual_post_step_dt{}_newtontol{}.png'.format(dt, newton_tol), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)

def main():
    """
    Main function that executes all the stuff containing:
        - plotting the solution for one single time step size,
        - plotting the differences around a discrete event (i.e., the differences at the time before, at, and after the event),
        - plotting the differences over time in both cases
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = approx_solution_hook

    nvars = 12
    problem_class = PilinePowerOutageDAE
    c = 0.2
    Vs = 150.0

    sweeper = fully_implicit_DAE
    newton_tolerances = [1e-7, 1e-1]

    use_detection = [False] #[False, True]

    t0 = 0.0
    Tend = 0.15

    dt_list = [5e-3, 1e-3, 5e-4, 1e-4]

    results_dt = dict()
    switching_dt = dict()
    results_simulations = dict()
    results_events = dict()

    res_norm_against_nfev = dict()
    for use_SE in use_detection:
        for dt_item in dt_list:
            for newton_tol in newton_tolerances:
                print(f'Controller run -- Simulation for step size: {dt_item}')

                description, controller_params = get_description(dt_item, nvars, problem_class, hookclass, sweeper, use_SE, newton_tol)

                description['problem_params']['c'] = c
                description['problem_params']['Vs'] = Vs

                stats, nfev = controller_run(t0, Tend, controller_params, description)

                plot_solution(stats, use_SE)

                residual_post_step = get_sorted(stats, type='residual_post_step', sortby='time')
                res_norm = max([me[1] for me in residual_post_step])
                res_norm_against_nfev[newton_tol] = [res_norm, nfev]
                print('Max. residual: ', res_norm, 'with newton_tol=', newton_tol)
                plot_residual_post_step(residual_post_step, dt_item, newton_tol)

    plot_nfev_against_residual(res_norm_against_nfev)

            #vRl = np.array([me[1][5] for me in get_sorted(stats, type='approx_solution')])
            #t = np.array([me[0] for me in get_sorted(stats, type='approx_solution')])

            #results_dt[dt_item] = pack_solution_data(t, vRl)

            #t_switches = (
            #    np.array([me[1] for me in get_recomputed(stats, type='switch', sortby='time')])
            #    if use_SE
            #    else np.zeros(1)
            #)
            #switching_dt[dt_item] = t_switches

        #results_simulations[use_SE] = results_dt
        #results_events[use_SE] = switching_dt

    #diffs_around_event(c, Vs, dt_list, use_detection, results_simulations, results_events)

    #diffs_over_time(c, Vs, dt_list, use_detection, results_simulations, results_events)


if __name__ == "__main__":
    main()
