from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.IEEE9BusSystem import IEEE9BusSystem
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
    level_params['restol'] = 1e-12
    level_params['dt'] = 1e-3

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 2
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nvars'] = 28
    problem_params['newton_tol'] = 1e-12

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = approx_solution_hook

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20


    problem = IEEE9BusSystem
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
    Tend = 2.0 #1e-3 #1.0

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    m = 3
    n = 9
    x0 = np.array([1.0591, 0.79193, 0.77098, 1.077, 0.76899, 0.71139, 0, 0.62385, 0.62505, 0.015514,
             -0.71253, -0.73358, 0.061423, 1.0645, 0.94343, 376.99, 376.99, 376.99, 1.0849,
             1.7917, 1.4051, 0.19528, 0.3225, 0.25292, 1.1077, 1.905, 1.4538, 0.71863, 1.6366,
             0.85245, 0.71863, 1.6366, 0.85245, 1.0591, 0.79193, 0.77098, 1.077, 0.76899, 0.71139, 0, 0.62385, 0.62505, 0.015514, -0.71253, -0.73358, 0.061423, 1.0645, 0.94343, 376.99, 376.99, 376.99, 1.0849, 1.7917, 1.4051, 0.19528, 0.3225, 0.25292, 1.1077, 1.905, 1.4538, 0.71863, 1.6366, 0.85245, 0.71863, 1.6366, 0.85245, 0.30185, 1.2884, 0.56058, 0.67159, 0.93446, 0.62021, 1.04, 1.025, 1.025, 1.0258, 0.99563, 1.0127, 1.0258, 1.0159, 1.0324, 0, 0.16197, 0.081415, -0.03869, -0.069618, -0.064357, 0.064921, 0.012698, 0.034326])
    # V = np.array([me[1][11*m + 2*m:11*m + 2*m + n] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    V = np.array([me[1][11*m + 2*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    # print([me[1][11*m + 2*m:11*m + 2*m + n] for me in get_sorted(stats, type='approx_solution', sortby='time', recomputed=False)])
    Eqp = np.array([me[1][0:m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    Si1d = np.array([me[1][m:2*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    Edp = np.array([me[1][2*m:3*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    ESi2q = np.array([me[1][3*m:4*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    Delta = np.array([me[1][4*m:5*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    w = np.array([me[1][5*m:6*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    Efd = np.array([me[1][6*m:7*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    RF = np.array([me[1][7*m:8*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    VR = np.array([me[1][8*m:9*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    TM = np.array([me[1][9*m:10*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    PSV = np.array([me[1][10*m:11*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    Id = np.array([me[1][11*m:11*m + m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    Iq = np.array([me[1][11*m + m:11*m + 2*m] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    V = np.array([me[1][11*m + 2*m:11*m + 2*m + n] for me in get_sorted(stats, type='approx_solution', sortby='time')])
    TH = np.array([me[1][11*m + 2*m + n:11*m + 2*m + 2 * n] for me in get_sorted(stats, type='approx_solution', sortby='time')])

    # print('Eqp=', Eqp - x0[0:m])
    # print('Si1d=', Si1d - x0[m:2*m])
    # print('Edp=', Edp - x0[2*m:3*m])
    # print('ESi2q=', ESi2q - x0[3*m:4*m])
    # print('Delta=', Delta - x0[4*m:5*m])
    # print('w=', w - x0[5*m:6*m])
    # print('Efd=', Efd - x0[6*m:7*m])
    # print('RF=', RF - x0[7*m:8*m])
    # print('VR=', VR - x0[8*m:9*m])
    # print('TM=', TM - x0[9*m:10*m])
    # print('PSV=', PSV - x0[10*m:11*m])
    # print('Id=', Id - x0[11*m:12*m])
    # print('Iq=', Iq - x0[12*m:13*m])
    # print('V=', V - x0[11*m + 2*m:11*m + 2*m + n])
    # print('TH=', TH - x0[11*m + 2*m + n:11*m + 2*m + 2 * n])


    fig3, ax3 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax3.plot(t, V[:, 0], label='V0')
    ax3.plot(t, V[:, 1], label='V1')
    ax3.plot(t, V[:, 2], label='V2')
    ax3.plot(t, V[:, 3], label='V3')
    ax3.plot(t, V[:, 4], label='V4')
    ax3.plot(t, V[:, 5], label='V5')
    ax3.plot(t, V[:, 6], label='V6')
    ax3.plot(t, V[:, 7], label='V7')
    ax3.plot(t, V[:, 8], label='V8')
    ax3.legend(loc='upper right', fontsize=10)
    fig3.savefig('data/V0.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig3)

    


if __name__ == "__main__":
    run()
