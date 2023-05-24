from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
from pySDC.projects.DAE.problems.soft_drink_manufacturing import IdealGasLiquid
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.implicit_Euler_DAE import implicit_Euler_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.projects.PinTSimE.battery_model import get_recomputed
from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def run():
    """
    Routine to run model problem
    """

    use_SE = True

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-6
    level_params['dt'] = 1e-4

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nvars'] = 4
    problem_params['newton_tol'] = 1e-7

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = approx_solution_hook

    convergence_controllers = dict()
    if use_SE:
        switch_estimator_params = {}
        convergence_controllers.update({SwitchEstimator: switch_estimator_params})

    max_restarts = 1
    if max_restarts is not None:
        convergence_controllers[BasicRestartingNonMPI] = {
            'max_restarts': max_restarts,
            'crash_after_max_restarts': False,
        }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 6

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = IdealGasLiquid
    description['problem_params'] = problem_params
    description['sweeper_class'] = fully_implicit_DAE
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    Path("data").mkdir(parents=True, exist_ok=True)

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = 2.758

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    ML = np.array([me[1][0] for me in get_sorted(stats, type='approx_solution')])
    MG = np.array([me[1][1] for me in get_sorted(stats, type='approx_solution')])
    P = np.array([me[1][2] for me in get_sorted(stats, type='approx_solution')])
    G = np.array([me[1][3] for me in get_sorted(stats, type='approx_solution')])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution')])

    if use_SE:
        switches = get_recomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'No switches found!'
        t_switch = [v[1] for v in switches]

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(6.0, 3))
    ax.set_title(r'Solution of $M_L$', fontsize=10)
    ax.plot(t, ML, label=r'$M_L$')
    if use_SE:
        for m in range(len(t_switch)):
            ax.axvline(x=t_switch[m], linestyle='--', linewidth=0.8, color='r', label='Switch {}'.format(m + 1))
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig('data/ideal_gas_liquid_solution_ML.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)

    fig2, ax2 = plt_helper.plt.subplots(1, 1, figsize=(6.0, 3))
    ax2.set_title(r'Solution of $M_G$', fontsize=10)
    ax2.plot(t, MG, label=r'$M_G$')
    ax2.legend(frameon=False, fontsize=8, loc='upper right')
    fig2.savefig('data/ideal_gas_liquid_solution_MG.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig2)

    fig3, ax3 = plt_helper.plt.subplots(1, 1, figsize=(6.0, 3))
    ax3.set_title(r'Solution of $P$', fontsize=10)
    ax3.plot(t, P, label=r'$P$')
    ax3.legend(frameon=False, fontsize=8, loc='upper right')
    fig3.savefig('data/ideal_gas_liquid_solution_P.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig3)

    fig4, ax4 = plt_helper.plt.subplots(1, 1, figsize=(6.0, 3))
    ax4.set_title(r'Solution of $G$', fontsize=10)
    ax4.plot(t, G, label=r'$G$')
    ax4.legend(frameon=False, fontsize=8, loc='upper right')
    fig4.savefig('data/ideal_gas_liquid_solution_G.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig4)


if __name__ == "__main__":
    run()
