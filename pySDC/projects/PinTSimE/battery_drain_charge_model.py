import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery_drain_charge
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.piline_model import setup_mpl
from pySDC.projects.PinTSimE.battery_model import controller_run, log_data
import pySDC.helpers.plot_helper as plt_helper


def generate_description(
    dt,
    problem,
    sweeper,
    hook_class,
    use_adaptivity,
    use_switch_estimator,
):
    """
    Generate a description for the battery models for a controller run.
    Args:
        dt (float): time step for computation
        problem (pySDC.core.Problem.ptype): problem class that wants to be simulated
        sweeper (pySDC.core.Sweeper.sweeper): sweeper class for solving the problem class numerically
        hook_class (pySDC.core.Hooks): logged data for a problem
        use_adaptivity (bool): flag if adaptivity wants to be used or not
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not

    Returns:
        description (dict): contains all information for a controller run
        controller_params (dict): Parameters needed for a controller run
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['Ipv'] = 10
    problem_params['Rpv'] = 0.01
    problem_params['Cpv'] = 500 * 1e-6
    problem_params['R0'] = 0.1
    problem_params['C0'] = 1
    problem_params['Rline2'] = 0.5
    problem_params['Rl'] = 1
    problem_params['V_ref'] = 0.9
    problem_params['alpha'] = 6

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = problem  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = sweeper  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def run():
    """
    Executes the simulation for the battery model using two different sweepers and plot the results
    as <problem_class>_model_solution_<sweeper_class>.png
    """

    dt = 1e-2
    t0 = 0.0
    Tend = 10.0

    problem_classes = [battery_drain_charge]
    sweeper_classes = [imex_1st_order]

    recomputed = False
    use_switch_estimator = [False]
    use_adaptivity = [False]

    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for use_SE in use_switch_estimator:
            for use_A in use_adaptivity:
                description, controller_params = generate_description(
                    dt, problem, sweeper, log_data, use_A, use_SE,
                )

                stats = controller_run(description, controller_params, use_A, use_SE, t0, Tend)

            plot_voltages(description, problem.__name__, sweeper.__name__, recomputed, use_SE, use_A)


def plot_voltages(description, problem, sweeper, recomputed, use_switch_estimator, use_adaptivity, cwd='./'):
    """
    Routine to plot the numerical solution of the model

    Args:
        description(dict): contains all information for a controller run
        problem (pySDC.core.Problem.ptype): problem class that wants to be simulated
        sweeper (pySDC.core.Sweeper.sweeper): sweeper class for solving the problem class numerically
        recomputed (bool): flag if the values after a restart are used or before
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        use_adaptivity (bool): flag if adaptivity wants to be used or not
        cwd (str): current working directory
    """

    f = open(cwd + 'data/{}_{}_USE{}_USA{}.dat'.format(problem, sweeper, use_switch_estimator, use_adaptivity), 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    vCpv = np.array([me[1][0] for me in get_sorted(stats, type='u', recomputed=recomputed)])
    vC0 = np.array([me[1][1] for me in get_sorted(stats, type='u', recomputed=recomputed)])

    t = np.array([me[0] for me in get_sorted(stats, type='u', recomputed=recomputed)])

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title('Simulation of {} using {}'.format(problem, sweeper), fontsize=10)
    ax.plot(t, vCpv, label=r'$v_{C_{pv}}$')
    ax.plot(t, vC0, label=r'$v_{C_0}$')

    ax.axhline(y=0.9, linestyle='--', linewidth=0.8, color='g', label='$V_{ref}$')

    ax.legend(frameon=False, fontsize=8, loc='upper right')

    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel('Energy', fontsize=8)

    fig.savefig('data/{}_model_solution_{}.png'.format(problem, sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    run()
