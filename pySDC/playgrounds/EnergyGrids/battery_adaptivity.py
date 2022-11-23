import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator

# from pySDC.implementations.sweeper_classes.generic_LU import generic_LU
from pySDC.playgrounds.EnergyGrids.log_data_battery import log_data_battery
from pySDC.projects.PinTSimE.piline_model import setup_mpl
import pySDC.helpers.plot_helper as plt_helper


def main(use_switch_estimator=True):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize adaptivity parameters
    adaptivity_params = dict()
    adaptivity_params['e_tol'] = 1e-12

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1e-10
    level_params['dt'] = 1e-3

    # initialize sweeper parameters
    sweeper_params = dict()

    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = 1
    problem_params['R'] = 1
    problem_params['L'] = 1
    problem_params['alpha'] = 1.2
    problem_params['V_ref'] = 1.0
    problem_params['set_switch'] = np.array([False], dtype=bool)
    problem_params['t_switch'] = np.zeros(1)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data_battery
    controller_params['mssdc_jac'] = False

    # initialize convergence controller parameters
    convergence_controllers = dict()
    convergence_controllers[Adaptivity] = adaptivity_params
    switch_estimator_params = {}
    convergence_controllers.update({SwitchEstimator: switch_estimator_params})
    print(convergence_controllers)

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['convergence_controllers'] = convergence_controllers  # pass convergence controller parameters

    # set time parameters
    t0 = 0.0
    Tend = 0.5

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # fname = 'data/piline.dat'
    fname = 'battery.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    restarts_sorted = np.array(get_sorted(stats, type='restart', recomputed=None))[:, 1]
    print("Restarts: -- ", np.sum(restarts_sorted))

    plot_voltages(use_switch_estimator)


def plot_voltages(use_switch_estimator, cwd='./'):
    """
    Routine to plot the numerical solution of the model
    """

    f = open(cwd + 'battery.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    cL = get_sorted(stats, type='current L', sortby='time', recomputed=False)
    vC = get_sorted(stats, type='voltage C', sortby='time', recomputed=False)

    times = [v[0] for v in cL]

    dt = np.array(get_sorted(stats, type='dt', recomputed=False))
    list_gs = get_sorted(stats, type='restart')

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1)
    ax.plot(times, [v[1] for v in cL], label='$i_L$')
    ax.plot(times, [v[1] for v in vC], label='$v_C$')

    if use_switch_estimator:
        val_switch = get_sorted(stats, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        ax.axvline(x=t_switch[0], linestyle='--', color='r', label='Switch')
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Energy', fontsize=20)
    #for element in list_gs:
    #    if element[1] > 0:
    #        ax.axvline(element[0], linestyle='--', color='r', label='Restart')
    dt_ax = ax.twinx()
    dt_ax.plot(dt[:, 0], dt[:, 1], 'ko--', label='dt')
    dt_ax.set_ylabel('dt', fontsize=20)
    # dt_ax.set_yscale('log', base=10)

    ax.legend(frameon=False, loc='upper right')
    dt_ax.legend(frameon=False, loc='center right')

    fig.savefig('battery_adaptivity.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
