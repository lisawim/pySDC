import numpy as np
from pathlib import Path

from pySDC.core.Errors import ParameterError

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE

import pySDC.helpers.plot_helper as plt_helper
import matplotlib.pyplot as plt

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun, getDataDict, plotSolution

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.DAE.misc.HookClass_DAE import (
    error_hook,
    LogGlobalErrorPostStepAlgebraicVariable,
)
from pySDC.implementations.hooks.default_hook import DefaultHooks
from pySDC.implementations.hooks.log_work import LogWork


def main():
    r"""
    Executes the simulation.
    """

    # defines parameters for sweeper
    M_fix = 3  # this M_fix will be ignored!! You can set number of collocation nodes in line 88
    # you can also set more than one like nnodes = [3, 4, 5]
    quad_type = 'RADAU-RIGHT'
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': quad_type,
        'QI': ['IE', 'LU', 'MIN3', 'MIN-SR-S'],
    }

    # defines parameters for event detection, restol, and max. number of iterations
    handling_params = {
        'restol': 1e-13,
        'maxiter': 50,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    problem_params = {
        'newton_tol': [1e-3],  # [10 ** (-m) for m in range(2, 10)],
    }

    all_params = {
        'sweeper_params': sweeper_params,
        'handling_params': handling_params,
        'problem_params': problem_params,
    }

    hook_class = [
        DefaultHooks,
        LogSolution,
        error_hook,
        LogWork,
        LogGlobalErrorPostStepAlgebraicVariable,
    ]

    use_detection = [False]
    use_adaptivity = [False]

    problem = DiscontinuousTestDAE
    prob_cls_name = problem.__name__
    sweeper = fully_implicit_DAE
    sweeper_cls_name = sweeper.__name__

    # dt_fix = 0.01
    dt_list = np.logspace(-1.0, -2.0, num=6)

    u_num = runSimulationStudy(
        problem=problem,
        sweeper=sweeper,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(1.0, 2.0),
        dt_list=dt_list,
        nnodes=[3],
    )

    plotDtAgainstMeanNiter(u_num, prob_cls_name, sweeper_cls_name)

    printItersAndNumberFunctionCalls(u_num, prob_cls_name, sweeper_cls_name)


def runSimulationStudy(problem, sweeper, all_params, use_adaptivity, use_detection, hook_class, interval, dt_list, nnodes):
    r"""
    Script that executes the simulation for a given problem class for given parameters defined by the user. This function
    is intented to use for studies, hence it includes several for-loops to loop over more different parameters.

    Parameters
    ----------
    problem : pySDC.core.Problem
        Problem class to be simulated.
    sweeper : pySDC.core.Sweeper
        Sweeper that is used to simulate the problem class.
    all_params : dict
        Dictionary contains the problem parameters for ``problem``, the sweeper parameters for ``sweeper``,
        and handling parameters needed for event detection, i.e., ``max_restarts``, ``recomputed``, ``tol_event``,
        ``alpha``, and ``exact_event_time_available``.
    use_adaptivity : list of bool
       Indicates whether adaptivity is used in the simulation or not. Here a list is used to iterate over the
       different cases, i.e., ``use_adaptivity=[True, False]``.
    use_detection : list of bool
       Indicates whether event detection is used in the simulation or not. Here a list is used to iterate over the
       different cases, i.e., ``use_detection=[True, False]``.
    hook_class : list of pySDC.core.Hooks
       List containing the different hook classes to log data during the simulation, i.e., ``hook_class=[LogSolution]``
       logs the solution ``u``.
    interval : tuple
       Simulation interval.
    dt_list : list of float
       List containing different step sizes where the solution is computed.
    nnodes : list of int
       The solution can be computed for different number of collocation nodes.
    """

    prob_cls_name = problem.__name__

    Path("data").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}").mkdir(parents=True, exist_ok=True)

    u_num = {}

    t0 = interval[0]
    Tend = interval[-1]

    QI = all_params['sweeper_params']['QI']
    preconditioners = [QI] if isinstance(QI, str) else QI

    maxiter = all_params['handling_params']['maxiter']

    newton_tol = all_params['problem_params']['newton_tol']
    newton_tolerances = [newton_tol] if isinstance(newton_tol, float) else newton_tol

    tol_event = all_params['handling_params']['tol_event']
    tolerances_event = [tol_event] if isinstance(tol_event, float) else tol_event

    alpha = all_params['handling_params']['alpha']
    alphas = [alpha] if isinstance(alpha, float) else alpha

    for QI in preconditioners:
        u_num[QI] = {}

        for dt in dt_list:
            u_num[QI][dt] = {}

            for M in nnodes:
                u_num[QI][dt][M] = {}

                for use_SE in use_detection:
                    u_num[QI][dt][M][use_SE] = {}

                    for use_A in use_adaptivity:
                        u_num[QI][dt][M][use_SE][use_A] = {}

                        for newton_tol in newton_tolerances:
                            u_num[QI][dt][M][use_SE][use_A][newton_tol] = {}

                            for tol_event in tolerances_event:
                                u_num[QI][dt][M][use_SE][use_A][newton_tol][tol_event] = {}

                                for alpha in alphas:
                                    u_num[QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha] = {}

                                    # print(f'Controller run -- Simulation for step size: {dt} -- M={M} -- newton_tol={newton_tol}')

                                    problem_params = all_params['problem_params']
                                    sweeper_params = all_params['sweeper_params']
                                    handling_params = all_params['handling_params']

                                    restol = -1 if use_A else handling_params['restol']

                                    problem_params.update({'newton_tol': newton_tol})

                                    description, controller_params = generateDescription(
                                        dt=dt,
                                        problem=problem,
                                        sweeper=sweeper,
                                        num_nodes=M,
                                        quad_type=sweeper_params['quad_type'],
                                        QI=QI,
                                        hook_class=hook_class,
                                        use_adaptivity=use_A,
                                        use_switch_estimator=use_SE,
                                        problem_params=problem_params,
                                        restol=restol,
                                        maxiter=maxiter,
                                        max_restarts=handling_params['max_restarts'],
                                        tol_event=tol_event,
                                        alpha=alpha,
                                    )

                                    stats, t_switch_exact = controllerRun(
                                        description=description,
                                        controller_params=controller_params,
                                        t0=t0,
                                        Tend=Tend,
                                        exact_event_time_avail=handling_params['exact_event_time_avail'],
                                    )

                                    u_num[QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha] = getDataDict(
                                        stats, prob_cls_name, use_A, use_SE, handling_params['recomputed'], t_switch_exact
                                    )

                                    plotSolution(
                                        u_num[QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha],
                                        prob_cls_name,
                                        use_A,
                                        use_SE,
                                    )

    return u_num


def plotStylingStuff(color_type, linestyle_type='QI'):
    """
    Returns plot stuff such as colors, line styles for making plots more pretty.

    Parameters
    ----------
    color_type : str
        Can be 'M' or 'QI'.
    linestyle_type : str, optional
        Can be 'M', 'QI' or 'newton_tol'.
    """

    if color_type == 'M':
        keys = [2, 3, 4, 5]
        colors = {
            2: 'limegreen',
            3: 'firebrick',
            4: 'deepskyblue',
            5: 'purple',
        }
    elif color_type == 'QI':
        keys = ['IE', 'LU', 'MIN', 'MIN3', 'MIN-SR-S', 'MIN-SR-NS']
        colors = {
            'IE': 'lightcoral',
            'LU': 'olivedrab',
            'MIN': 'turquoise',
            'MIN3': 'mediumorchid',
            'MIN-SR-S': 'darkorange',
            'MIN-SR-NS': 'dodgerblue',
        }
    else:
        raise ParameterError(f"No color type for {color_type} implemented!")

    if linestyle_type == 'M' or linestyle_type == 'QI':
        linestyles = {key:val for key, val in zip(keys, ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed'])}
    elif linestyle_type == 'newton_tol':
        linestyles = {
            0: 'solid',
            1: 'dotted',
            2: 'dashed',
            3: 'dashdot',
            4: (0, (1, 10)),
            5: (0, (1, 1)),
            6: (5, (10, 3)),
            7: (0, (5, 10)),
            8: (0, (5, 1)),
            9: (0, (3, 10, 1, 10)),
        }
    else:
        raise ParameterError(f"No linestyle type for {linestyle_type} implemented!")

    markers = {key:val for key, val in zip(keys, ['s', 'o', '*', 'd', 'h', '8'])}

    xytext = {key:val for key, val in zip(keys, [(-13, -13), (-13.0, 15), (-13.0, -10), (-13.0, -22), (-13.0, 42), (-13.0, -40)])}

    return colors, linestyles, markers, xytext


def plotDtAgainstMeanNiter(u_num, prob_cls_name, sweeper_cls_name):
    """
    Plots for different step sizes and different preconditioners the number of function calls in Newton-like solver.

    Considered keys: Only QI, dt have to be lists! All other keys are not considered!

    Parameters
    ----------
    u_num : dict
        Contains the numerical solution as well as some other statistics for different parameters.
    prob_cls_name : str
        Name of the problem class.
    sweeper_cls_name : str
        Name of the sweeper.
    """

    colors, linestyles, markers, xytext = plotStylingStuff(color_type='QI')

    QI_keys = list(u_num.keys())
    dt_keys = list(u_num[QI_keys[0]].keys())
    M_keys = list(u_num[QI_keys[0]][dt_keys[0]].keys())
    use_SE_keys = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]].keys())
    use_A_keys = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]].keys())
    newton_tolerances = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]].keys())
    tol_event = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]].keys())
    alpha = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]].keys())

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for M in M_keys:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
        for QI in QI_keys:
            mean_niters = [
                u_num[QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]]['mean_niters']
                for dt in dt_keys
            ]

            ax.semilogx(
                dt_keys,
                mean_niters,
                color=colors[QI],
                linestyle=linestyles[QI],
                marker=markers[QI],
                linewidth=1.1,
                alpha=0.6,
                label=f'$Q_\Delta={QI}$',
            )

        # ax.set_title(f"M={M_keys[0]} for {quad_type}")
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(0, 20)
        ax.set_xscale('log', base=10)
        ax.set_xlabel(r'Step size $\Delta t$', fontsize=16)
        ax.set_ylabel(r'Mean number of iterations', fontsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=False, fontsize=12, loc='upper right')
        ax.minorticks_off()

        fig.savefig(f"data/{prob_cls_name}/plot_dt_against_mean_niter_{sweeper_cls_name}_M={M}.png", dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def printItersAndNumberFunctionCalls(u_num, prob_cls_name, sweeper_cls_name):
    """
    Prints number of iterations needed to converge. Moreover, number of function calls required over all
    iterations is printed.

    Considered keys: Only newton_tol, M have to be lists! All other keys are not considered!

    Parameters
    ----------
    u_num : dict
        Contains the numerical solution as well as some other statistics for different parameters.
    prob_cls_name : str
        Name of the problem class.
    sweeper_cls_name : str
        Name of the sweeper.
    """

    QI_keys = list(u_num.keys())
    dt_keys = list(u_num[QI_keys[0]].keys())
    M_keys = list(u_num[QI_keys[0]][dt_keys[0]].keys())
    use_SE_keys = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]].keys())
    use_A_keys = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]].keys())
    newton_tolerances = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]].keys())
    tol_event = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]].keys())
    alpha = list(u_num[QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]].keys())

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]
    newton_tol_keys = newton_tolerances if len(newton_tolerances) > 1 else [newton_tolerances[0]]

    print()
    print(f'----------------------- Solve {prob_cls_name} using {sweeper_cls_name} -----------------------')
    for M in M_keys:
        for QI in QI_keys:
            for ind, newton_tol in enumerate(newton_tolerances):
                mean_niters = u_num[QI][dt_keys[0]][M][use_SE_keys[0]][use_A_keys[0]][newton_tol][tol_event[0]][alpha[0]]['mean_niters']
                nfev_root_solver = u_num[QI][dt_keys[0]][M][use_SE_keys[0]][use_A_keys[0]][newton_tol][tol_event[0]][alpha[0]]['newton'][:, 1]
                mean_nfev_root_solver = np.mean(nfev_root_solver)

                print()
                print(f'The costs for solving one time step of size {dt_keys[0]} with tolerance {newton_tol}')
                print(f'with M={M} using {QI} preconditioner are:')
                print()
                print(f'Mean number of iterations: {mean_niters}')
                print(f'Mean number of function calls along all iterations: {mean_nfev_root_solver}')
                print()
                print(f'----------------------------------------------------------------------------------------------')
                print()



if __name__ == "__main__":
    main()
