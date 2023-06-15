from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
from pySDC.projects.DAE.run.piline import get_description
from pySDC.projects.DAE.problems.buck_dae import BuckConverterDAE
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.projects.PinTSimE.battery_model import get_recomputed
from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper


def main():
    """
    A simple test program to see the fully implicit SDC solver in action
    """

    use_SE = True

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = 5e-3

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 5
    sweeper_params['initial_guess'] = 'spread'
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1e-12  # tollerance for implicit solver
    problem_params['nvars'] = 13

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 15
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

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = BuckConverterDAE
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
    Tend = 10.0  #3.47 #10.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    vC1 = np.array([me[1][1] for me in get_sorted(stats, type='approx_solution')])
    vC2 = np.array([me[1][4] for me in get_sorted(stats, type='approx_solution')])
    iLp = np.array([me[1][9] for me in get_sorted(stats, type='approx_solution')])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution')])

    if use_SE:
        switches = get_recomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'No switches found!'
        t_switch = [v[1] for v in switches]
        print(t_switch)

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title(r'Solution of Buck Converter', fontsize=8)
    ax.plot(t, vC1, 'k-', label=r'$v_{C1}$', linewidth=1)
    ax.plot(t, vC2, 'b-', label=r'$v_{C2}$', linewidth=1)
    ax.plot(t, iLp, 'g-', label=r'$i_L$', linewidth=1)
    if use_SE:
        for m in range(len(t_switch)):
            if m == 0:
                ax.axvline(x=t_switch[m], linestyle='--', linewidth=0.8, color='r', label='{} Events'.format(len(t_switch)))
            else:
                ax.axvline(x=t_switch[m], linestyle='--', linewidth=0.8, color='r')
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.savefig('data/buck_solution.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)

def main2():
    """
    Main function that executes all the stuff containing:
        - plotting the solution for one single time step size,
        - plotting the differences around a discrete event (i.e., the differences at the time before, at, and after the event)
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = approx_solution_hook

    nvars = 13
    problem_class = BuckConverter_DAE


if __name__ == "__main__":
    main()
