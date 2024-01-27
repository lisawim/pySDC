from mpi4py import MPI
import numpy as np
from pathlib import Path

from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.SemiExplicitDAE import SemiExplicitDAE
from pySDC.projects.DAE.sweepers.fully_implicit_DAE_MPI import fully_implicit_DAE_MPI
from pySDC.projects.DAE.sweepers.SemiExplicitDAEMPI import SemiExplicitDAEMPI

from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAE, LinearTestDAEMinion
from pySDC.projects.DAE.problems.WSCC9BusSystem import WSCC9BusSystem

import pySDC.helpers.plot_helper as plt_helper
import matplotlib.pyplot as plt

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.DAE.misc.HookClass_DAE import (
    error_hook,
    LogGlobalErrorPostStepAlgebraicVariable,
    LogGlobalErrorPostIter,
    LogGlobalErrorPostIterAlg,
)
from pySDC.implementations.hooks.default_hook import DefaultHooks
from pySDC.implementations.hooks.log_work import LogWork

from pySDC.projects.DAE.run.DAE_study import plotStylingStuff
from pySDC.projects.PinTSimE.battery_model import getDataDict, plotSolution

def run(QI, M, pre_problem_params, maxiter, dt, Tend, problem, sweeper, comm, useMPI):
    """
    Routine to initialise iteration test parameters
    """

    sweeper_cls_name = problem.__name__

    Path("data").mkdir(parents=True, exist_ok=True)
    Path(f"data/{sweeper_cls_name}").mkdir(parents=True, exist_ok=True)

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5e-13
    level_params['dt'] = dt

    # This comes as read-in for the sweeper class
    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': QI,
        'initial_guess': 'spread',
        'comm': comm,
    }
    if useMPI:
        sweeper_params.update({'comm': comm})

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['newton_tol'] = 1e-10
    problem_params['method'] = 'hybr'
    problem_params.update(pre_problem_params)

    # This comes as read-in for the step class
    step_params = dict()
    step_params['maxiter'] = maxiter

    hook_class = [
        DefaultHooks,
        LogSolution,
        # error_hook,
        # LogWork,
        # LogGlobalErrorPostStepAlgebraicVariable,
        # LogGlobalErrorPostIter,
        # LogGlobalErrorPostIterAlg,
    ]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = problem
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    t0 = 0.0

    # instantiate controller
    # if useMPI:
        # controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
        # P = controller.S.levels[0].prob
    # else:
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob    
    
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


def main():
    lamb_diff = 2.0  # [10 ** m for m in range(-3, 2)]
    lamb_alg = 1.0
    t_line_outage = 0.0
    detect_disturbance = False
    dt_list = [0.01]  # np.logspace(-1.0, -2.0, num=6)  # [1e-6]
    Tend = 2 * dt_list[0]
    nnodes = [3]  # [2, 3, 4, 5]
    maxiter = 100
    QIs = ['IE']

    useMPI = True
    sweeper_classes = [fully_implicit_DAE]  # [fully_implicit_DAE_MPI] if useMPI else [fully_implicit_DAE]

    problem_classes = [LinearTestDAEMinion] * len(sweeper_classes)
    prob_cls_name = problem_classes[0].__name__

    pre_problem_params = {}
    if prob_cls_name == 'LinearTestDAE':
        pre_problem_params = {'lamb_diff': lamb_diff, 'lamb_alg': lamb_alg}
        plot_labels = None
    elif prob_cls_name == 'WSCC9BusSystem':
        pre_problem_params = {'t_line_outage': t_line_outage, 'detect_disturbance': detect_disturbance}
        plot_labels = [r'$P_{SV,0}$', r'$P_{SV,1}$', r'$P_{SV,2}$']
    elif prob_cls_name == 'LinearTestDAEMinion':
        plot_labels = None

    if useMPI:
        comm = MPI.COMM_WORLD
    else:
        comm = None

    u_num = {}
    for prob_cls, sweeper_cls in zip(problem_classes, sweeper_classes):
        sweeper_cls_name = sweeper_cls.__name__
        u_num[sweeper_cls_name] = {}

        for dt in dt_list:
            u_num[sweeper_cls_name][dt] = {}

            for M in nnodes:
                u_num[sweeper_cls_name][dt][M] = {}

                for QI in QIs:
                    u_num[sweeper_cls_name][dt][M][QI] = {}

                    stats = run(QI, M, pre_problem_params, maxiter, dt, Tend, prob_cls, sweeper_cls, comm, useMPI)

                    u_num[sweeper_cls_name][dt][M][QI] = getDataDict(
                        stats, prob_cls.__name__, maxiter, False, False, False, None
                    )

                    plotSolution(
                        u_num[sweeper_cls_name][dt][M][QI],
                        prob_cls.__name__,
                        sweeper_cls_name,
                        False,
                        False,
                        plot_labels,
                    )


if __name__ == "__main__":
    main()
