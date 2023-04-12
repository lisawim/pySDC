from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.synchronous_generator import SynchronousGenerator, SynchronousGenerator_Piline
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def run():
    """
    Routine to run model problem
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-2

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nvars'] = 14
    problem_params['newton_tol'] = 1e-7

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = approx_solution_hook

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 30


    problem = SynchronousGenerator
    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = problem
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    Path("data").mkdir(parents=True, exist_ok=True)

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 10.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    id = np.array([me[1][6] for me in get_sorted(stats, type='approx_solution')])
    iq = np.array([me[1][7] for me in get_sorted(stats, type='approx_solution')])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution')])

    Tm = np.zeros(len(t))
    for m in range(len(t)):
        if round(t[m], 14) < 2:
            Tm[m] = 0.854
        else:
            Tm[m] = 0.354

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title('Simulation', fontsize=10)
    ax.plot(t, id, label=r'$i_d$')
    ax.plot(t, iq, label=r'$i_q$')
    #ax.plot(t, Tm, label=r'$T_m$')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig('data/{}.png'.format(problem.__name__), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    run()
