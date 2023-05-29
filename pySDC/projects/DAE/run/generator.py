from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.synchronous_generator import SynchronousGenerator, SynchronousGenerator_Piline, SynchronousGenerator_5Ybus
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.implicit_Euler_DAE import implicit_Euler_DAE
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
    level_params['restol'] = 1e-9
    level_params['dt'] = 1e-3

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nvars'] = 28
    problem_params['newton_tol'] = 1e-12

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = approx_solution_hook

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10


    problem = SynchronousGenerator_5Ybus
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
    Tend = 0.05 #1e-3 #1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    id = np.array([me[1][6] for me in get_sorted(stats, type='approx_solution')])
    iq = np.array([me[1][7] for me in get_sorted(stats, type='approx_solution')])
    omega_m = np.array([me[1][12] for me in get_sorted(stats, type='approx_solution')])
    delta_r = np.array([me[1][13] for me in get_sorted(stats, type='approx_solution')])
    v_d = np.array([me[1][14] for me in get_sorted(stats, type='approx_solution')])
    v_q = np.array([me[1][15] for me in get_sorted(stats, type='approx_solution')])
    #V_re = np.array([me[1][16] for me in get_sorted(stats, type='approx_solution')])
    #V_im = np.array([me[1][17] for me in get_sorted(stats, type='approx_solution')])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution')])

    Tm = np.zeros(len(t))
    for m in range(len(t)):
        if round(t[m], 14) < 0.2:
            Tm[m] = 0.854
        else:
            Tm[m] = 0.354

    wb = 100 * np.pi
    freq = omega_m * wb / (2*np.pi)

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title('Simulation', fontsize=10)
    ax.plot(t, id, label=r'$i_d$')
    ax.plot(t, iq, label=r'$i_q$')
    #ax.plot(t, Tm, label=r'$T_m$')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig('data/{}.png'.format(problem.__name__), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)

    fig2, ax2 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax2.set_title('Frequency', fontsize=10)
    ax2.plot(t, freq, label='Frequency')
    ax2.plot(t, omega_m, label=r'$\omega_m$')
    ax2.legend(frameon=False, fontsize=8, loc='upper right')
    fig2.savefig('data/{}_frequency.png'.format(problem.__name__), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig2)


if __name__ == "__main__":
    run()
