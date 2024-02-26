from mpi4py import MPI
import numpy as np
from pathlib import Path

from pySDC.core.Errors import ParameterError

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.fully_implicit_DAE_MPI import fully_implicit_DAE_MPI
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE

import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.battery_model import getDataDict, plotSolution

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepAlgebraicVariable
from pySDC.implementations.hooks.default_hook import DefaultHooks
from pySDC.implementations.hooks.log_work import LogWork


def run(QI, M, pre_problem_params, maxiter, t0, dt, Tend, problem, sweeper, comm):
    """
    Routine to initialise iteration test parameters
    """

    Path("data").mkdir(parents=True, exist_ok=True)
    Path(f"data/{problem.__name__}").mkdir(parents=True, exist_ok=True)

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
    # for parallel SDC the number of collocation nodes M needs to be equal to number of processes
    # since we parallelize over these nodes
    assert M == comm.Get_size(), f"Number of nodes does not match with number of processes!"
    sweeper_params.update({'comm': comm})

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['newton_tol'] = 1e-10
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

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob    
    
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


def main():
    t0 = 1.0
    dt_list = [0.01] 
    Tend = 2.0
    nnodes = [3]
    maxiter = 100
    QIs = ['IEpar']  # since we perform parallel SDC 'IE' and 'LU' cannot be used here. A diagonal matrix only can be used for that
    # thus try instead ['IEpar', 'MIN', 'MIN-SR-S', 'Qpar']

    sweeper_classes = [fully_implicit_DAE_MPI]

    problem_classes = [DiscontinuousTestDAE] * len(sweeper_classes)

    pre_problem_params = {}
    plot_labels = None  # None indicates that all unknowns shall be plotted (a bit confusing I know..)

    comm = MPI.COMM_WORLD

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

                    stats = run(QI, M, pre_problem_params, maxiter, t0, dt, Tend, prob_cls, sweeper_cls, comm)

                    # only first process should plot
                    if comm.Get_rank() == 0:
                        u_num[sweeper_cls_name][dt][M][QI] = getDataDict(
                            stats, prob_cls.__name__, False, False, False, None
                        )

                        plotSolution(
                            u_num[sweeper_cls_name][dt][M][QI],
                            prob_cls.__name__,
                            sweeper_cls_name,
                            False,
                            False,
                        )


if __name__ == "__main__":
    main()
