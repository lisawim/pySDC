import numpy as np
from pathlib import Path

from pySDC.core.Errors import ParameterError

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.Runge_Kutta_DAE import BackwardEulerDAE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1, problematic_f

import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.battery_model import runSimulation

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepAlgebraicVariable
from pySDC.implementations.hooks.default_hook import DefaultHooks
from pySDC.implementations.hooks.log_work import LogWork


def main():
    """
    Executes the simulation.
    """

    # no sweeper params necessary here, M_fix, 'quad_type', 'QI' is only for SDC and the function is build in the way
    # that these params are necessary, but it will be ignored in RK sweepers
    M_fix = 3
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': 'RADAU-RIGHT',
        'QI': 'IE',
    }

    # defines parameters for event detection, restol, and max. number of iterations
    # only maxiter = 1 is important. RK methods are not iterative so maxiter > 1 leads to unnecessary computations
    # restol = -1 since we "do only one iteration"
    # all other parameters can be ignored since we do no switch estimation here
    handling_params = {
        'restol': -1,
        'maxiter': 1,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    # newton_tol has to be set very small for RK since we only do one iteration
    # here, nvars has to be set for each considered problem
    problem_params = {
        'nvars': 3,
        'newton_tol': 1e-14,
    }

    all_params = {
        'sweeper_params': sweeper_params,
        'handling_params': handling_params,
        'problem_params': problem_params,
    }

    hook_class = [DefaultHooks, LogSolution, LogGlobalErrorPostStep, LogGlobalErrorPostStepAlgebraicVariable, LogWork]

    # both are False, because we don't do switch estimation and adaptivity
    use_detection = [False]
    use_adaptivity = [False]

    problem = DiscontinuousTestDAE
    prob_cls_name = problem.__name__
    sweeper = BackwardEulerDAE
    sweeper_cls_name = sweeper.__name__

    dt_list = [0.01]

    # I made my life a bit more easy, so the complete simulation is done in another file
    # namely pySDC/projects/PinTSimE/battery_model.py
    # you will only a get a dictionary also created in battery_model.py
    M = [M_fix]
    u_num = runSimulation(
        problem=problem,
        sweeper=sweeper,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(1.0, 2.5),
        dt_list=dt_list,
        nnodes=M,
    )

    plotSolution(u_num[dt_list[0]][M[0]][use_detection[0]][use_adaptivity[0]], prob_cls_name, sweeper_cls_name, use_adaptivity[0], use_detection[0])


def plotSolution(u_num, prob_cls_name, sweeper_cls_name, use_adaptivity, use_detection):
    r"""
    Plots the numerical solution for one simulation run.

    Parameters
    ----------
    u_num : dict
        Contains numerical solution with corresponding times for different problem_classes, and
        labels for different unknowns of the problem.
    prob_cls_name : str
        Name of the problem class to be plotted.
    sweeper_cls_name : str
        Name of the sweeper.
    use_adaptivity : bool
        Indicates whether switch estimation is used in the simulation or not.
    use_adaptivity : bool
        Indicates whether adaptivity is used in the simulation or not.
    """

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))

    unknowns = u_num['unknowns']
    unknowns_labels = u_num['unknowns_labels']
    for unknown, unknown_label in zip(unknowns, unknowns_labels):
        ax.plot(u_num['t'], u_num[unknown], label=unknown_label)

    if use_detection:
        t_switches = u_num['t_switches']
        for i in range(len(t_switches)):
            ax.axvline(x=t_switches[i], linestyle='--', linewidth=0.8, color='r', label='Event {}'.format(i + 1))

    if use_adaptivity:
        dt_ax = ax.twinx()
        dt = u_num['dt']
        dt_ax.plot(dt[:, 0], dt[:, 1], linestyle='-', linewidth=0.8, color='k', label=r'$\Delta t$')
        dt_ax.set_ylabel(r'$\Delta t$', fontsize=16)
        dt_ax.legend(frameon=False, fontsize=12, loc='center right')

    ax.legend(frameon=False, fontsize=12, loc='upper right')
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(r'$u(t)$', fontsize=16)

    fig.savefig(f'data/{prob_cls_name}_solution_{sweeper_cls_name}.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    main()