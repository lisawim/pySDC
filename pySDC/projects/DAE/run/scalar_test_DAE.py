from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.problems.test_DAE import ScalarTestDAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook, error_hook
from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.projects.PinTSimE.battery_model import get_recomputed
from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def main():
    """
    Main function that executes all the stuff.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = [approx_solution_hook, error_hook]

    nvars = 2
    problem_class = ScalarTestDAE

    sweeper = fully_implicit_DAE
    num_nodes = 3
    newton_tol = 1e-12
    maxiter = 25

    use_detection = [True]

    t0 = 2.6
    Tend = 2.7

    dt_list = [1e-5]

    for use_SE in use_detection:
        for dt in dt_list:
            print(f'Controller run -- Simulation for step size: {dt}')

            restol = -1 if use_SE else 1e-12
            recomputed = False if use_SE else None

            description, controller_params = get_description(
                dt,
                nvars,
                problem_class,
                hookclass,
                sweeper,
                num_nodes,
                use_SE,
                restol,
                0.6 * dt,
                maxiter,
                newton_tol,
            )

            stats, t_switch_exact = controller_run(t0, Tend, controller_params, description)

            plot_solution(stats, recomputed, use_SE)

            err = get_sorted(stats, type='error_post_step', sortby='time', recomputed=recomputed)
            fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
            ax.set_title(r'Error', fontsize=8)
            ax.plot([me[0] for me in err], [me[1] for me in err])
            ax.set_yscale('log', base=10)
            fig.savefig('data/scalar_test_DAE_error.png', dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)

            if use_SE:
                print_event_time_error(stats, t_switch_exact)


def get_description(
    dt, nvars, problem_class, hookclass, sweeper, num_nodes, use_detection, restol, tol_SE, maxiter=25, newton_tol=1e-12
):
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
    num_nodes : int
        Number of collocation nodes used for integration.
    use_detection : bool
        Indicate whether switch detection should be used or not.
    restol : float
        Residual tolerance used as stopping criterion.
    maxiter : int
        Maximum number of iterations done per time step.
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
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['QI'] = 'LU'
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = newton_tol  # tollerance for implicit solver
    problem_params['nvars'] = nvars

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = hookclass

    convergence_controllers = dict()
    if use_detection:
        switch_estimator_params = {'tol': tol_SE}
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
    t_switch_exact = P.t_switch_exact

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, t_switch_exact


def plot_solution(stats, recomputed, use_detection=False, cwd='./'):
    """
    Plots the solution of one simulation run (i.e., for one time step size).

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    recomputed : bool
        Indicate that values after restart should be used.
    use_detection : bool, optional
        Indicate whether switch detection should be used or not.
    cwd : str
        Current working directory.
    """

    y = np.array([me[1][0] for me in get_sorted(stats, type='approx_solution', recomputed=recomputed)])
    z = np.array([me[1][1] for me in get_sorted(stats, type='approx_solution', recomputed=recomputed)])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution', recomputed=recomputed)])

    y_exact, z_exact = [], []
    t_switch_exact = 0.5 * np.arcsinh(100)
    for time in t:
        if time < t_switch_exact:
            y_exact.append(np.cosh(time))
            z_exact.append(np.sinh(time))
        else:
            y_exact.append(np.cosh(t_switch_exact))
            z_exact.append(np.sinh(t_switch_exact))

    if use_detection:
        switches = get_recomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'No switches found!'
        t_switch = [v[1] for v in switches]

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title(r'Solution /w switch detection', fontsize=8)
    ax.plot(t, y, 'r--', label=r"$y$")
    ax.plot(t, z, 'g--', label=r"$z$")
    ax.plot(t, y_exact, 'r-', label=r"$y_\mathrm{exact}$")
    ax.plot(t, z_exact, 'g-', label=r"$z_\mathrm{exact}$")
    if use_detection:
        for m in range(len(t_switch)):
            if m == 0:
                ax.axvline(
                    x=t_switch[m],
                    linestyle='--',
                    linewidth=0.8,
                    color='g',
                    label='{} Event(s) found'.format(len(t_switch)),
                )
            else:
                ax.axvline(x=t_switch[m], linestyle='--', linewidth=0.8, color='g')
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax.set_xlabel('Time[s]', fontsize=8)
    fig.savefig('data/scalar_test_DAE_solution.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def print_event_time_error(stats, t_switch_exact):
    """
    Prints the error between the exact event time and the event time founded by switch estimation.

    Parameter
    ---------
    stats : dict
        Raw statistics from a controller run.
    t_switch_exact : float
        Exact event time of the problem.
    """

    switches = get_recomputed(stats, type='switch', sortby='time')
    assert len(switches) >= 1, 'No switches found!'
    t_switches = [v[1] for v in switches]

    for m in range(len(t_switches)):
        err = abs(t_switch_exact - t_switches[m])
        print(f'Switch found: {t_switches[m]} -- Error to exact event time: {err}')


if __name__ == "__main__":
    main()
