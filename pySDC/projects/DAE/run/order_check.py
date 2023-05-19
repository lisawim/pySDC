from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1, problematic_f
from pySDC.projects.DAE.sweepers.BDF_DAE import BDF_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook, error_hook
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.PinTSimE.piline_model import setup_mpl
import pySDC.helpers.plot_helper as plt_helper


def get_description(dt, nvars, problem_class, newton_tol, hookclass, BDF_sweeper, k_step):
    """
    Returns the description for one simulation run.

    Args:
        dt (float): time step size for computation
        nvars (int): number of variables of the problem
        problem_class (pySDC.core.Problem.ptype_dae): problem class in DAE formulation that wants to be simulated
        newton_tol (float): Tolerance for solving the nonlinear system of DAE solver
        hookclass (pySDC.core.Hooks): hook class to log the data
        BDF_sweeper (pySDC.core.Sweeper): sweeper class for solving the problem class
        k_step (int): number of k-steps used for the multi-step (BDF) method

    Returns:
        description (dict): contains all information for a controller run
        controller_params (dict): specific parameters for the controller
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['starting_values'] = None
    sweeper_params['k_step'] = k_step

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = newton_tol  # tolerance for implicit solver
    problem_params['nvars'] = nvars

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 1

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    if hookclass is not None:
        controller_params['hook_class'] = hookclass

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = BDF_sweeper
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description, controller_params


def controller_run(t0, Tend, controller_params, description):
    """
    Executes a controller run for time interval to be specified in the arguments.

    Args:
        t0 (float): initial time of simulation
        Tend (float): end time of simulation
        controller_params (dict): parameters needed for the controller
        description (dict): contains all information for a controller run

    Returns:
        stats (dict): raw statistics from a controller run
    """

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


def plot_solution(nvars, problem_class, stats):
    """
    Plots the solution of the problem class integrated by the sweeper.

    Args:
        nvars (int): number of unknowns in the problem class
        problem_class (pySDC.core.Problem.ptype_dae): problem class in DAE formulation that wants to be simulated
        stats (dict): Raw statistics from a controller run
    """

    u1 = np.array([me[1][0] for me in get_sorted(stats, type='approx_solution', recomputed=False)])
    u2 = np.array([me[1][1] for me in get_sorted(stats, type='approx_solution', recomputed=False)])
    if nvars == 3:
        u3 = np.array([me[1][2] for me in get_sorted(stats, type='approx_solution', recomputed=False)])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution', recomputed=False)])

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title('Numerical solution')
    ax.plot(t, u1, label=r'$u_1$')
    ax.plot(t, u2, label=r'$u_2$')
    if nvars == 3:
        ax.plot(t, u3, label=r'$u_3$')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig('data/{}_solution.png'.format(problem_class.__name__), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_order(dt_list, global_errors, p, problem_class):
    """
    Plots the order of accuracy p for the sweeper used for the computation.

    Args:
        dt_list (list): list of step sizes
        global_errors (list): contains the global errors for each step size
        p (int): order of accuracy to be plotted and considered
        problem_class (pySDC.core.Problem.ptype_dae): problem class in DAE formulation that wants to be simulated
    """

    order_ref = [dt**p for dt in dt_list]
    global_err_dt = [err[1] for err in global_errors]

    setup_mpl()
    fig_order, ax_order = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax_order.set_title('Order of accuracy')
    ax_order.loglog(dt_list, order_ref, 'k--', label='Reference order $p={}$'.format(p))
    ax_order.loglog(dt_list, global_err_dt, 'o-', label='Order reached')
    ax_order.set_xlabel(r'$\Delta t$', fontsize=8)
    ax_order.set_ylabel(r'$||\bar{u}-\tilde{u}||_\infty$', fontsize=8)
    ax_order.legend(frameon=False, fontsize=8, loc='lower right')
    fig_order.savefig('data/{}_accuracy_order.png'.format(problem_class.__name__), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_order)


def main():
    """
    Main function, it executes the simulation of a problem class using a DAE sweeper for different step sizes.
    Also, it plots the solution for some random step size.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = [approx_solution_hook, error_hook]

    nvars = [3, 2]
    problem_classes = [simple_dae_1, problematic_f]
    newton_tol = 1e-12

    # for BDF methods, k_step defines the steps for multi-step as well as the order for BDF methods
    k_step = 1
    sweeper = BDF_DAE

    t0 = 0.0
    dt_list = [2 ** (-m) for m in range(2, 12)]

    random_dt = np.random.choice(dt_list)

    global_errors = list()
    for problem_class, n in zip(problem_classes, nvars):
        for dt_item in dt_list:
            print(f'Controller run -- Simulation for step size: {dt_item}')

            description, controller_params = get_description(
                dt_item, n, problem_class, newton_tol, hookclass, sweeper, k_step
            )

            if problem_class.__name__ == 'simple_dae_1':
                Tend = 1.0
            else:
                Tend = np.pi

            stats = controller_run(t0, Tend, controller_params, description)

            err = np.array([me[1] for me in get_sorted(stats, type='error_post_step', recomputed=False)])
            global_err_dt = max(err)
            global_errors.append([dt_item, global_err_dt])

            # plot solution of one random step size
            if dt_item == random_dt:
                plot_solution(n, problem_class, stats)

        plot_order(dt_list, global_errors, k_step, problem_class)

        global_errors = list()


if __name__ == "__main__":
    main()
