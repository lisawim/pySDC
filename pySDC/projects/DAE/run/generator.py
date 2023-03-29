from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.synchronous_generator import SynchronousGenerator
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
    level_params['dt'] = 1e-1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3

    # initialize problem parameters
    problem_params = dict()
    problem_params['Ld'] = 1.81
    problem_params['Lq'] = 1.76
    problem_params['Lffd'] = 1.82
    problem_params['Lf1d'] = 1.66
    problem_params['L11d'] = 1.83
    problem_params['L11q'] = 2.34
    problem_params['L22q'] = 1.74
    problem_params['Lad'] = 1.66
    problem_params['Laq'] = 1.61
    problem_params['Ra'] = 3 * 1e-1
    problem_params['Rfd'] = 6 * 1e-4
    problem_params['R1d'] = 2.84 * 1e-2
    problem_params['R1q'] = 6.2 * 1e-3
    problem_params['R2q'] = 2.37 * 1e-2
    problem_params['H'] = 3.53
    problem_params['Kd'] = 0.0
    problem_params['w0'] = 2 * np.pi * 60
    problem_params['wb'] = 2 * np.pi * 60
    problem_params['vfd'] = 8.736 * 1e-4
    problem_params['Zline'] = -0.269 - 0.15*j
    problem_params['vbus'] = 0.7

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [approx_solution_hook, error_hook]

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = SynchronousGenerator
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

    # check error
    err = get_sorted(stats, type='error_post_step', sortby='time')
    err = np.linalg.norm([err[i][1] for i in range(len(err))], np.inf)
    print(f"Error is {err}")
    assert np.isclose(err, 0.0, atol=1e-4), "Error too large."

   id = np.array([me[1][10] for me in get_sorted(stats, type='approx_solution_hook', recomputed=False)])
   iq = np.array([me[1][11] for me in get_sorted(stats, type='approx_solution_hook', recomputed=False)])
   Tm = np.array([me[1][19] for me in get_sorted(stats, type='approx_solution_hook', recomputed=False)])
   t = np.array([me[0] for me in get_sorted(stats, type='approx_solution_hook', recomputed=False)])

   fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
   ax.set_title('Simulation', fontsize=10)
   ax.plot(t, id, label=r'$i_d$')
   ax.plot(t, iq, label=r'$i_q$')
   ax.plot(t, Tm, label=r'$T_m$')
   ax.legend(frameon=False, fontsize=8, loc='upper right')
   fig.savefig('data/generator.png'.format(problem, sweeper), dpi=300, bbox_inches='tight')
   plt_helper.plt.close(fig)


if __name__ == "__main__":
    run()
