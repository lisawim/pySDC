from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
from pySDC.projects.DAE.problems.ideal_gas_liquid_DAE import IdealGasLiquid
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook, sweeper_data
from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.projects.PinTSimE.battery_model import get_recomputed
from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Hooks import hooks
import pySDC.helpers.plot_helper as plt_helper


class event_data(hooks):
    """
    Hook class to add the state function of a discontinuous differential-algebraic equation after each step.
    Data of the state functions are problem dependent.
    """

    def __init__(self):
        """Initialization routine"""
        super(event_data, self).__init__()

    def post_step(self, step, level_number):
        """
        Default routine called after each step.

        Parameters
        ----------
        step : pySDC.core.Step
            Current step.
        level_number : pySDC.core.Level
            Current level number.
        """

        super(event_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='state_function',
            value=L.uend[0] / P.rho_L - P.V_d,
        )


def main():
    """
    Routine to run model problem
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = [approx_solution_hook, event_data, sweeper_data]

    nvars = 4
    problem_class = IdealGasLiquid

    sweeper = fully_implicit_DAE
    newton_tolerances = [1e-5] #[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] #1e-12

    use_detection = [False, True] #[False, True]

    # set time parameters
    t0 = 0.0
    Tend = 5.0 #1.3
    dt_list = [3e-2] #[1.035e-1]
    N_dt = int(Tend / dt_list[0])

    state_function_eval = dict()
    state_function_eval_dt = dict()
    switches_dt = dict()

    res_norm_against_newton_tol = dict()
    for newton_tol in newton_tolerances:
        for use_SE in use_detection:
            for dt in dt_list:
                print(f'Controller run -- Simulation for step size: {dt}')

                restol = -1 if use_SE else 1e-14

                description, controller_params = get_description(dt, nvars, problem_class, hookclass, sweeper, use_SE, restol, newton_tol)

                stats = controller_run(t0, Tend, controller_params, description)

                plot_solution(stats, use_SE)

                t = [me[0] for me in get_sorted(stats, type='state_function', recomputed=False)]
                h = [me[1] for me in get_sorted(stats, type='state_function', recomputed=False)]
                state_function_eval_dt[dt] = pack_solution_data(t, h)

                t_switches = (
                    np.array([me[1] for me in get_recomputed(stats, type='switch', sortby='time')])
                    if use_SE
                    else np.zeros(1)
                )
                switches_dt[dt] = t_switches

                # Could also be an indicator to reduce costs after SE or to indicate the costs in general?
                nfev = [me[1] for me in get_sorted(stats, type='nfev', sortby='time', recomputed=False)]
                nfev = round(sum(nfev) / N_dt)  # average of nfev across all time steps
                residual_post_step = get_sorted(stats, type='residual_post_step', sortby='time', recomputed=False)
                res_norm = max([me[1] for me in residual_post_step])
                res_norm_against_newton_tol[newton_tol] = [res_norm, nfev]

            state_function_eval[use_SE] = state_function_eval_dt

        diffs_over_time(dt_list, use_detection, state_function_eval, switches_dt)

    # only use for use_SE = False
    plot_nfev_against_residual(res_norm_against_newton_tol)


def get_description(dt, nvars, problem_class, hookclass, sweeper, use_detection, restol, newton_tol=1e-12):
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
    level_params['restol'] = restol
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'
    sweeper_params['initial_guess'] = 'spread'


    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = newton_tol  # tollerance for implicit solver
    problem_params['nvars'] = nvars

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 7

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 15
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

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')
    mean_niter = np.mean([me[1] for me in iter_counts])
    out = 'Mean number of iterations: %f4.2' % mean_niter
    print(out)

    return stats


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

    ML = np.array([me[1][0] for me in get_sorted(stats, type='approx_solution', recomputed=False)])
    P = np.array([me[1][2] for me in get_sorted(stats, type='approx_solution', recomputed=False)])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution', recomputed=False)])

    if use_detection:
        switches = get_recomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'No switches found!'
        t_switches = [v[1] for v in switches]

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(6.0, 3))
    ax.set_title(r'Solution of $M_L$', fontsize=10)
    ax.plot(t, ML, label=r'$M_L$')
    if use_detection:
        for m in range(len(t_switches)):
            if m == 0:
                ax.axvline(x=t_switches[m], linestyle='--', linewidth=0.8, color='r', label='{} Events found'.format(len(t_switches)))
            else:
                ax.axvline(x=t_switches[m], linestyle='--', linewidth=0.8, color='r')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.savefig('data/ideal_gas_liquid_solution_ML.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(6.0, 3))
    ax.set_title(r'Solution of $P$', fontsize=10)
    ax.plot(t, P, label=r'$P$')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    fig.savefig('data/ideal_gas_liquid_solution_P.png', dpi=300, bbox_inches='tight')
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


def diffs_over_time(dt_list, use_detection, state_function_eval_dt, switches_dt):
    """
    Plots the state function over time. It can be investigated how large the error is.

    Parameters
    ----------
    dt_list : list
        Contains multiple time step sizes.
    use_detection : list
        Contains the iterable object for indicating whether a detection of events is used.
    state_functions_eval : dict
        Results of the solution for each time step size.
    """

    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax.set_title(r'Evaluating state function over time for $\Delta t=%s$' % dt, fontsize=8)
        for use_SE in use_detection:
            h = state_function_eval_dt[use_SE][dt]
            t_switches = switches_dt[dt]
            t_switch = t_switches[-1]

            if use_SE:
                ax.plot(h[0, :], h[1, :], 'r--', linewidth=0.5, label=r'Detection - {}'.format(use_SE))

                for m in range(len(t_switches)):
                    if m == 0:
                        ax.axvline(
                            x=t_switches[m],
                            linestyle='--',
                            linewidth=0.5,
                            color='g',
                            label='{} Event(s) found'.format(len(t_switches)),
                        )
                    else:
                        ax.axvline(x=t_switches[m], linestyle='--', linewidth=0.5, color='g')
            else:
                ax.plot(h[0, :], h[1, :], 'k-', linewidth=0.5, label=r'Detection - {}'.format(use_SE))

        ax.legend(frameon=False, fontsize=8, loc='lower right')

        #ax.set_xlim(t_switch - 0.001, t_switch + 0.001)
        ax.set_ylim(-1, 1)
        ax.set_yscale('symlog', linthresh=1e-10)
        ax.set_xlabel(r'Time[s]', fontsize=8)
        ax.set_ylabel(r'State function $h(M_L)$', fontsize=8)

        fig.savefig('data/ideal_gas_liquid_diffs_over_time_{}.png'.format(dt), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_nfev_against_residual(res_norm_against_newton_tol):
    """
    Plots different newtol's against the residual (max) norm, where the residuals after one step is considered.

    Parameters
    ----------
    res_norm_against_newton_tol : dict
        Contains the residual norm and corresponding number of function evaluations for one specific newton_tol.
    """

    lists = sorted(res_norm_against_newton_tol.items())
    newton_tols, res_norm_against_newton_tol_list = zip(*lists)
    res_norms, nfev = [me[0] for me in res_norm_against_newton_tol_list], [me[1] for me in res_norm_against_newton_tol_list]

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(5, 3))
    ax.loglog(newton_tols, res_norms, color='firebrick', marker='.', linewidth=0.8)
    for m in range(len(newton_tols)):
        ax.annotate(nfev[m], (newton_tols[m], res_norms[m]), xytext=(-8.5, 10), textcoords="offset points", fontsize=8)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_ylim(1e-16, 1e+1)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Inner tolerance', fontsize=8)
    ax.set_ylabel(r'$||r_\mathrm{DAE}||_\infty$', fontsize=8)
    ax.grid(visible=True)
    ax.minorticks_off()
    fig.savefig('data/ideal_gas_liquid_residual_against_tolerances.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    main()

