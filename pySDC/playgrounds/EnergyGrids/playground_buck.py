import numpy as np
import dill
from scipy.integrate import solve_ivp

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.collocation_classes.gauss_lobatto import CollGaussLobatto
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.BuckConverter import buck_converter
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.playgrounds.EnergyGrids.log_data import log_data
import pySDC.helpers.plot_helper as plt_helper


def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-9
    level_params['dt'] = 0.0001     # 0.25

    # initialize sweeper parameters
    sweeper_params = dict()
    # sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['collocation_class'] = CollGaussLobatto
    sweeper_params['num_nodes'] = 5     # 3
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    
    # initialize problem parameters
    problem_params = dict()
    problem_params['duty'] = 1
    problem_params['fsw'] = 1e3  
    problem_params['Vs'] = 10.0
    problem_params['Rs'] = 0.5
    problem_params['C1'] = 1e-3
    problem_params['L1'] = 1e-3
    problem_params['C2'] = 1e-3
    problem_params['Rl'] = 5 
    
    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data
    
    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = buck_converter                  # pass problem class
    description['problem_params'] = problem_params                # pass problem parameters
    description['sweeper_class'] = imex_1st_order              # pass sweeper
    description['sweeper_params'] = sweeper_params                # pass sweeper parameters
    description['level_params'] = level_params                    # pass level parameters
    description['step_params'] = step_params   
    
    # set time parameters
    t0 = 0.0
    Tend = 0.02
    
    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    fname = 'data/buck.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()


def plot_voltages(cwd='./'):
    f = open(cwd + 'data/buck.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    v1 = sort_stats(filter_stats(stats, type='v1'), sortby='time')
    v2 = sort_stats(filter_stats(stats, type='v2'), sortby='time')
    p3 = sort_stats(filter_stats(stats, type='p3'), sortby='time')

    times = [v[0] for v in v1]

    # plt_helper.setup_mpl()
    plt_helper.plt.plot(times, [v[1] for v in v1], label='v1')
    plt_helper.plt.plot(times, [v[1] for v in v2], label='v2')
    plt_helper.plt.plot(times, [v[1] for v in p3], label='p3')
    plt_helper.plt.legend()

    plt_helper.plt.show()


if __name__ == "__main__":
    main()
    plot_voltages()
