from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.soft_drink_manufacturing import SoftDrinkManufacturing_fully_implicit
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
    level_params['dt'] = 1e-3

    # initialize sweeper parameters
    sweeper_params = dict()
    #sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 1
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nvars'] = 9
    problem_params['newton_tol'] = 1e-7

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = approx_solution_hook

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 6

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = SoftDrinkManufacturing_fully_implicit
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
    Tend = 7.5

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    M1 = np.array([me[1][0] for me in get_sorted(stats, type='approx_solution')])
    M2 = np.array([me[1][1] for me in get_sorted(stats, type='approx_solution')])
    M3 = np.array([me[1][2] for me in get_sorted(stats, type='approx_solution')])
    Ml = np.array([me[1][3] for me in get_sorted(stats, type='approx_solution')])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution')])

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title(r'Solution of $M_l$', fontsize=10)
    #ax.plot(t, M1, label=r'$M_1$')
    #ax.plot(t, M2, label=r'$M_2$')
    #ax.plot(t, M3, label=r'$M_3$')
    ax.plot(t, Ml, label=r'$M_l$')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig('data/soft_drink_solution_Ml.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)

    fig2, ax2 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax2.set_title(r'Solution of $M_1$', fontsize=10)
    ax2.plot(t, M1, label=r'$M_1$')
    ax2.legend(frameon=False, fontsize=8, loc='upper right')
    fig2.savefig('data/soft_drink_solution_M1.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig2)

    fig3, ax3 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax3.set_title(r'Solution of $M_2$', fontsize=10)
    ax3.plot(t, M2, label=r'$M_2$')
    ax3.legend(frameon=False, fontsize=8, loc='upper right')
    fig3.savefig('data/soft_drink_solution_M2.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig3)

    fig4, ax4 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax4.set_title(r'Solution of $M_3$', fontsize=10)
    ax4.plot(t, M3, label=r'$M_3$')
    ax4.legend(frameon=False, fontsize=8, loc='upper right')
    fig4.savefig('data/soft_drink_solution_M3.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig4)


if __name__ == "__main__":
    run()
