import numpy as np
from pathlib import Path

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import CosineProblem, VanDerPol, EmbeddedLinearTestDAE
from pySDC.implementations.problem_classes.odeScalar import ProtheroRobinson

from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE
from pySDC.projects.DAE.sweepers.genericImplicitEmbedded import genericImplicitEmbedded
from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAE, LinearTestDAEIntegralFormulation

import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun, getDataDict, plotSolution

from pySDC.implementations.hooks.log_solution import LogSolution, LogSolutionAfterIteration
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostIter, LogGlobalErrorPostStep
from pySDC.implementations.hooks.default_hook import DefaultHooks
from pySDC.implementations.hooks.log_work import LogWork
from pySDC.projects.DAE.misc.hooksEpsEmbedding import (
    LogGlobalErrorPostStep,
    LogGlobalErrorPostStepPerturbation,
    LogGlobalErrorPostIter,
    LogGlobalErrorPostIterPerturbation,
)

from pySDC.projects.DAE.misc.HookClass_DAE import (
    LogGlobalErrorPostStepDifferentialVariable,
    LogGlobalErrorPostStepAlgebraicVariable,
    LogGlobalErrorPostIterDiff,
    LogGlobalErrorPostIterAlg,
)


def main():
    r"""
    Executes the simulation.
    """

    # defines parameters for sweeper
    M_fix = 3
    quad_type = 'RADAU-RIGHT'
    initial_guess = 'IE'
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': quad_type,
        'QI': ['IE'], #['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S'],
    }

    # defines parameters for event detection, restol, and max. number of iterations
    maxiter = 100#250
    handling_params = {
        'restol': 1e-12,#1e-10,
        'maxiter': maxiter,
        'compute_one_step': True,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    epsilons = [1.0]#[10 ** (-k) for k in range(1, 11, 1)]#np.logspace(-12.0, 0.0, 100)
    eps_fix = epsilons#[eps for eps in epsilons if eps <= 1e-10]#epsilons

    problems = [LinearTestDAEIntegralFormulation]
    prob_cls_name = problems[0].__name__
    sweeper = [genericImplicitEmbedded]

    if prob_cls_name == 'EmbeddedLinearTestDAEMinion':
        handling_params.update({'compute_ref': True})
    elif prob_cls_name == 'EmbeddedLinearTestDAE':
        handling_params.update({'compute_ref': True})
    else:
        handling_params.update({'compute_ref': False})

    handling_params.update({'compute_ref': False})

    problem_params = {
        'newton_tol': 1e-12,
    }

    hook_class = [
        DefaultHooks,
        LogSolution,
        # LogSolutionAfterIteration,
        LogWork,
        # LogGlobalErrorPostStep,
        # LogGlobalErrorPostIter,
        # LogGlobalErrorPostStep,
        # LogGlobalErrorPostStepPerturbation,
        LogGlobalErrorPostStepDifferentialVariable,
        LogGlobalErrorPostStepAlgebraicVariable,
        # LogGlobalErrorPostIterDiff,
        # LogGlobalErrorPostIterAlg,
        # LogGlobalErrorPostIter,
        # LogGlobalErrorPostIterPerturbation,
    ]
    all_params = {
        'sweeper_params': sweeper_params,
        'handling_params': handling_params,
        'problem_params': problem_params,
    }

    use_detection = [False]
    use_adaptivity = [False]

    t0 = 0.0
    dt_list = np.logspace(-4.0, -0.0, num=30)#[10 ** (-k) for k in range(0, 4, 1)]
    t1 = 0.5

    u_num = runSimulationStudy(
        problems=problems,
        sweeper=sweeper,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(t0, t1),  # note that the interval will be ignored if compute_one_step=True!
        dt_list=dt_list,
        nnodes=[4],
        epsilons=epsilons,
    )

    # plotErrIterODEForDifferentEps(u_num, prob_cls_name)

    # plotIterations(u_num, prob_cls_name, quad_type)

    # plotErrors(u_num, prob_cls_name, quad_type)

    # plotErrorsM(u_num, prob_cls_name, quad_type, initial_guess)

    # plotErrorInEachIterationEps(u_num, prob_cls_name)

    # plotResidual(u_num, prob_cls_name, quad_type)

    # plotResidualM(u_num, prob_cls_name, quad_type, initial_guess)

    # plotResidualDt(u_num, prob_cls_name, quad_type, initial_guess)

    # plotResidualEps(u_num, prob_cls_name, quad_type, initial_guess)

    # plotQuantitiesAlongIterations(u_num, prob_cls_name, eps_fix, maxiter)

    plot_accuracy_order(u_num, prob_cls_name, quad_type, eps_fix, initial_guess)

    # plotOrderAfterEachIteration(u_num, prob_cls_name)

    # plotCosts(u_num, prob_cls_name, initial_guess)

    # plotCostsDt(u_num, prob_cls_name, initial_guess)

    # plotCostsEps(u_num, prob_cls_name, initial_guess)


def runSimulationStudy(problems, sweeper, all_params, use_adaptivity, use_detection, hook_class, interval, dt_list, nnodes, epsilons, cwd='./'):
    r"""
    Script that executes the simulation for a given problem class for given parameters defined by the user. This function
    is intented to use for studies, hence it includes several for-loops to loop over more different parameters.

    Parameters
    ----------
    problems : pySDC.core.Problem
        List of Problem classes to be simulated.
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

    prob_cls_name = problems[0].__name__

    Path("data").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}/eps_embedding").mkdir(parents=True, exist_ok=True)

    u_num = {}

    t0 = interval[0]
    Tend = interval[-1]

    sweeper_classes = sweeper if isinstance(sweeper, list) else [sweeper]

    QI = all_params['sweeper_params']['QI']
    preconditioners = [QI] if isinstance(QI, str) else QI

    maxiter = all_params['handling_params']['maxiter']

    compute_one_step = all_params['handling_params']['compute_one_step']

    newton_tol = all_params['problem_params']['newton_tol']
    newton_tolerances = [newton_tol] if isinstance(newton_tol, float) else newton_tol

    tol_event = all_params['handling_params']['tol_event']
    tolerances_event = [tol_event] if isinstance(tol_event, float) else tol_event

    alpha = all_params['handling_params']['alpha']
    alphas = [alpha] if isinstance(alpha, float) else alpha

    epsilons = [epsilons] if isinstance(epsilons, float) else epsilons

    for problem, sweeper in zip(problems, sweeper_classes):
        sweeper_cls_name = sweeper.__name__
        u_num[sweeper_cls_name] = {}
        print(f'Controller run for {sweeper_cls_name}')

        for QI in preconditioners:
            u_num[sweeper_cls_name][QI] = {}
            print(f"Preconditioner: {QI}")
            for dt in dt_list:
                print(f"Current time step size: {dt}")
                u_num[sweeper_cls_name][QI][dt] = {}

                Tend = t0 + dt if compute_one_step else interval[-1]

                for M in nnodes:
                    u_num[sweeper_cls_name][QI][dt][M] = {}

                    for use_SE in use_detection:
                        u_num[sweeper_cls_name][QI][dt][M][use_SE] = {}

                        for use_A in use_adaptivity:
                            u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A] = {}

                            for newton_tol in newton_tolerances:
                                u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol] = {}

                                for tol_event in tolerances_event:
                                    u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol][tol_event] = {}

                                    for alpha in alphas:
                                        u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha] = {}

                                        for eps in epsilons:
                                            u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha][eps] = {}
                                            # print(f'Controller run -- Simulation for step size: {dt} -- M={M} -- newton_tol={newton_tol}')
                                            print(f"Epsilon: {eps}")
                                            problem_params = all_params['problem_params']
                                            sweeper_params = all_params['sweeper_params']
                                            handling_params = all_params['handling_params']

                                            # problem_params.update({'eps': eps})

                                            restol = -1 if use_A else handling_params['restol']

                                            problem_params.update({'newton_tol': newton_tol})

                                            description, controller_params, controller = generateDescription(
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
                                                controller=controller,
                                                t0=t0,
                                                Tend=Tend,
                                                exact_event_time_avail=handling_params['exact_event_time_avail'],
                                            )

                                            u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha][eps] = getDataDict(
                                                stats, prob_cls_name, use_A, use_SE, handling_params['recomputed'], t_switch_exact, maxiter
                                            )

                                            plot_labels = None
                                            plotSolution(
                                                u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha][eps],
                                                prob_cls_name,
                                                sweeper.__name__,
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
        keys = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        colors = {
            2: 'limegreen',
            3: 'firebrick',
            4: 'deepskyblue',
            5: 'purple',
            6: 'olivedrab',
            7: 'darkorange',
            8: 'dodgerblue',
            9: 'deeppink',
            10: 'turquoise',
            11: 'darkblue',
            12: 'mediumorchid',
            13: 'lightcoral',
        }
    elif color_type == 'QI':
        keys = ['IE', 'LU', 'MIN', 'MIN-A', 'MIN3', 'MIN-SR-S', 'MIN-SR-NS', 'IEpar', 'Qpar', 'PIC']
        colors = {
            'IE': 'limegreen',  #'lightcoral',
            'LU': 'firebrick',  # 'olivedrab',
            'MIN': 'deepskyblue',  # 'turquoise',
            'MIN-A': 'limegreen',
            'MIN3': 'mediumorchid',
            'MIN-SR-S': 'olivedrab',  # 'darkorange',
            'MIN-SR-NS': 'dodgerblue',
            'IEpar': 'purple',  # 'firebrick',
            'Qpar': 'darkblue',
            'PIC': 'deeppink',
        }
    elif color_type == 'sweeper':
        keys = ['fully_implicit_DAE', 'SemiExplicitDAE', 'SemiExplicitDAEReduced', 'generic_implicit']
        colors = {
            'fully_implicit_DAE': 'firebrick',
            'SemiExplicitDAE': 'deepskyblue',
            'SemiExplicitDAEReduced': 'olivedrab',
            'generic_implicit': 'purple',
        }
    else:
        # raise ParameterError(f"No color type for {color_type} implemented!")
        keys = np.arange(0, 13, 1)
        colors = {
            0: 'turquoise',
            1: 'deepskyblue',
            2: 'purple',
            3: 'firebrick',
            4: 'limegreen',
            5: 'orange',
            6: 'plum',
            7: 'salmon',
            8: 'forestgreen',
            9: 'midnightblue',
            10: 'gold',
            11: 'silver',
            12: 'black',
        }

    if linestyle_type == 'M' or linestyle_type == 'QI':
        linestyles = {key:val for key, val in zip(keys, ['solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed'])}
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
            10: 'solid',
            11: 'dashed',
            12: 'dotted',
            13: 'dashdot',
        }
    else:
        raise ParameterError(f"No linestyle type for {linestyle_type} implemented!")

    markers = {key:val for key, val in zip(keys, ['s', 'o', '*', 'd', 'h', '8', '^', 'v', 'p', 'D', '>', '<'])}

    xytext = {key:val for key, val in zip(keys, [(-13, -13), (-13.0, 15), (-13.0, -10), (-13.0, -22), (-13.0, 42), (-13.0, -40), (-13.0, 50), (-13.0, -52), (-13, 20)])}

    return colors, linestyles, markers, xytext

def getDataDictKeys(u_num):
    r"""
    Returns the keys of all sub dictionaries of the ``u_num`` dict.

    Parameters
    ----------
    u_num : dict
        Contains the numerical solution as well as some other statistics for different parameters.

    Returns
    -------
    sweeper_keys : list
        Keys with used sweepers.
    QI_keys : list
        Keys with used preconditioners.
    dt_keys : list
        Keys with all step sizes.
    M_keys : list
        Keys of used number of collocation nodes.
    use_SE_keys : list
        Keys with bool's if SE is used or not.
    use_A_keys : list
        Keys with bool's if A is used or not.
    newton_tolerances : list
        Keys with used inner tolerances.
    tol_event : :list
        Keys with used tolerances to find the event using SE.
    alpha : list
       Keys of used factors :math:`\alpha` for the SE.
    """

    sweeper_keys = list(u_num.keys())
    QI_keys = list(u_num[sweeper_keys[0]].keys())
    dt_keys = list(u_num[sweeper_keys[0]][QI_keys[0]].keys())
    M_keys = list(u_num[sweeper_keys[0]][QI_keys[0]][dt_keys[0]].keys())
    use_SE_keys = list(u_num[sweeper_keys[0]][QI_keys[0]][dt_keys[0]][M_keys[0]].keys())
    use_A_keys = list(u_num[sweeper_keys[0]][QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]].keys())
    newton_tolerances = list(u_num[sweeper_keys[0]][QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]].keys())
    tol_event = list(u_num[sweeper_keys[0]][QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]].keys())
    alpha = list(u_num[sweeper_keys[0]][QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]].keys())
    lamb_diff_keys = list(u_num[sweeper_keys[0]][QI_keys[0]][dt_keys[0]][M_keys[0]][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]].keys())

    return sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, lamb_diff_keys


def NestedListIntoSingleList(res):
    tmp_list = [item for item in res]
    err_iter = []
    for item in tmp_list:
        err_dt = [me[1] for me in item]
        if len(err_dt) > 0:
            err_iter.append([me[1] for me in item])
        # else:
        #     err_iter.append(err_iter[-1])
    err_iter = [item[0] for item in err_iter]
    return err_iter


def plotErrors(u_num, prob_cls_name, quad_type):
    colors, linestyles, markers, _ = plotStylingStuff(color_type='QI')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for i, dt in enumerate(dt_keys):

        for M in M_keys:
            fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 7.5))

            for QI in QI_keys:
                # res_diff = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['e_global'][:, 1] for eps in eps_keys]
                res_diff = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['e_global'] for eps in eps_keys]
                print(res_diff)
                err_diff = res_diff
                # err_diff = [max(abs(val)) for val in res_diff]

                ax[0].loglog(
                    eps_keys,
                    err_diff,
                    color=colors[QI],
                    # linestyle=linestyles[QI],
                    marker=markers[QI],
                    markeredgecolor='k',
                    linewidth=1.1,
                    label=f'{QI}',
                )

                # res_alg = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['e_global_algebraic'][:, 1] for eps in eps_keys]
                res_alg = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['e_global_algebraic'] for eps in eps_keys]
                err_alg = res_alg
                # err_alg = [max(abs(val)) for val in res_alg]

                ax[1].loglog(
                    eps_keys,
                    err_alg,
                    color=colors[QI],
                    # linestyle=linestyles[QI],
                    marker=markers[QI],
                    markeredgecolor='k',
                    linewidth=1.1,
                    label=f'{QI}',
                )

            ax[0].set_ylabel(r'Error in $y$', fontsize=16)
            ax[1].set_ylabel(r'Error in $z$', fontsize=16)
            for ax in [ax[0], ax[1]]:
                ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_ylim(1e-15, 1e3)
                ax.set_xscale('log', base=10)
                ax.grid(visible=True)
                ax.legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
                ax.minorticks_off()
            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotErrorsDiffEps_{quad_type}_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


def plotErrorsM(u_num, prob_cls_name, quad_type, initial_guess):
    colors, linestyles, markers, _ = plotStylingStuff(color_type='M')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for i, dt in enumerate(dt_keys):

        for QI in QI_keys:
            fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 7.5))

            for M in M_keys:
                res_diff = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['e_global'][:, 1] for eps in eps_keys]
                # res_diff = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['e_global'] for eps in eps_keys]
                print(res_diff)
                err_diff = res_diff
                err_diff = [max(abs(val)) for val in res_diff]

                ax[0].loglog(
                    eps_keys,
                    err_diff,
                    color=colors[M],
                    # linestyle=linestyles[QI],
                    marker=markers[M],
                    markeredgecolor='k',
                    linewidth=1.1,
                    label=rf'$M=${M}',
                )

                res_alg = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['e_global_algebraic'][:, 1] for eps in eps_keys]
                # res_alg = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['e_global_algebraic'] for eps in eps_keys]
                err_alg = res_alg
                err_alg = [max(abs(val)) for val in res_alg]

                ax[1].loglog(
                    eps_keys,
                    err_alg,
                    color=colors[M],
                    # linestyle=linestyles[QI],
                    marker=markers[M],
                    markeredgecolor='k',
                    linewidth=1.1,
                    label=rf'$M=${M}',
                )

            ax[0].set_ylabel(r'Error in $y$', fontsize=16)
            ax[1].set_ylabel(r'Error in $z$', fontsize=16)
            for ax in [ax[0], ax[1]]:
                ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_ylim(1e-15, 1e3)
                ax.set_xscale('log', base=10)
                ax.grid(visible=True)
                ax.legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
                ax.minorticks_off()
            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotErrorsDiffEps_{quad_type}_QI={QI}_dt={dt}_initial_guess={initial_guess}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


def plotErrorInEachIterationEps(u_num, prob_cls_name):
    """
    Plots the error in each iteration by taking the maximum value of the error in each iteration.
    For a semi-explicit problem the error in the differential and in the algebraic variable is plotted.

    Considered keys: Only sweeper, QI, M have to be lists! All other keys are not considered!

    Parameters
    ----------
    u_num : dict
        Contains the numerical solution as well as some other statistics for different parameters.
    prob_cls_name : str
        Name of the problem class.
    """

    colors, linestyles, markers, _ = plotStylingStuff(color_type='eps')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    type_err = ['e_global_post_iter', 'e_global_algebraic_post_iter']
    for dt in dt_keys:
        for QI in QI_keys:
            for M in M_keys:
                fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 5))

                for e, eps in enumerate(eps_keys):
                    for ind, label in enumerate(type_err):
                        if ind == 0:
                            ylabel = r'Error norm $||e_{diff}||_\infty$'
                        else:
                            ylabel = r'Error norm $||e_{alg}||_\infty$'

                        res = u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps][label]
                        tmp_list = [item for item in res]
                        err_iter = []
                        for item in tmp_list:
                            err_dt = [me[1] for me in item]
                            if len(err_dt) > 0:
                                err_iter.append(max([me[1] for me in item]))
                            else:
                                err_iter.append(err_iter[-1])
                        # print(err_iter)
                        ax[ind].semilogy(
                            np.arange(1, len(err_iter) + 1),
                            err_iter,
                            color=colors[e],
                            linestyle=linestyles[e],
                            linewidth=1.5,
                            label=rf'$\varepsilon$={eps}',
                        )

                        ax[ind].tick_params(axis='both', which='major', labelsize=16)
                        ax[ind].set_ylim(1e-16, 1e+4)
                        ax[ind].set_yscale('log', base=10)
                        ax[ind].set_xlabel(r'Iteration', fontsize=16)
                        ax[ind].set_ylabel(ylabel, fontsize=16)
                        ax[ind].grid(visible=True)
                        ax[ind].legend(frameon=False, fontsize=10, loc='upper right', ncol=3)  # , ncol=int(0.5*len(QI_keys)))
                        ax[ind].minorticks_off()

                fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotErrorOverIterations_M={M}_QI={QI}_dt_{dt}.png", dpi=300, bbox_inches='tight')
                plt_helper.plt.close(fig)


def plotResidual(u_num, prob_cls_name, quad_type):
    colors, linestyles, markers, _ = plotStylingStuff(color_type='QI')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for i, dt in enumerate(dt_keys):

        for M in M_keys:
            fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 7.5))

            for QI in QI_keys:
                residual = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['residual'][:, 1] for eps in eps_keys]

                ax.loglog(
                    eps_keys,
                    residual,
                    color=colors[QI],
                    # linestyle=linestyles[QI],
                    marker=markers[QI],
                    markeredgecolor='k',
                    linewidth=1.1,
                    label=f'{QI}',
                )

            ax.set_ylabel(r'Residual norm', fontsize=16)
            ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_ylim(1e-10, 1e-5)
            ax.set_xscale('log', base=10)
            ax.set_yscale('log', base=10)
            ax.grid(visible=True)
            ax.legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
            ax.minorticks_off()
            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotResidualDiffEps_{quad_type}_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


def plotResidualM(u_num, prob_cls_name, quad_type, initial_guess):
    colors, linestyles, markers, _ = plotStylingStuff(color_type='M')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for i, dt in enumerate(dt_keys):

        for QI in QI_keys:
            fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 7.5))

            for M in M_keys:
                residual = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['residual'][:, 1] for eps in eps_keys]

                ax.loglog(
                    eps_keys,
                    residual,
                    color=colors[M],
                    # linestyle=linestyles[QI],
                    marker=markers[M],
                    markeredgecolor='k',
                    linewidth=1.1,
                    label=rf'$M={M}$',
                )

            ax.set_ylabel(r'Residual norm', fontsize=16)
            ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_ylim(1e-10, 1e-5)
            ax.set_xscale('log', base=10)
            ax.set_yscale('log', base=10)
            ax.grid(visible=True)
            ax.legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
            ax.minorticks_off()
            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotResidualDiffEps_{quad_type}_QI={QI}_dt={dt}_initial_guess={initial_guess}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


def plotResidualDt(u_num, prob_cls_name, quad_type, initial_guess):
    colors, _, markers, _ = plotStylingStuff(color_type='dt')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for M in M_keys:
        for QI in QI_keys:
            fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 7.5))

            for i, dt in enumerate(dt_keys):
                residual = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['residual'][:, 1] for eps in eps_keys]
                print(residual)
                ax.loglog(
                    eps_keys,
                    residual,
                    color=colors[i],
                    # linestyle=linestyles[QI],
                    marker=markers[i],
                    markeredgecolor='k',
                    linewidth=1.1,
                    label=rf'$\Delta t=${dt}',
                )

            ax.set_ylabel(r'Residual norm', fontsize=16)
            ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_ylim(1e-15, 1e-5)
            ax.set_xscale('log', base=10)
            ax.set_yscale('log', base=10)
            ax.grid(visible=True)
            ax.legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
            ax.minorticks_off()
            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotResidualDiffEps_{quad_type}_QI={QI}_M={M}_initial_guess={initial_guess}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


def plotResidualEps(u_num, prob_cls_name, quad_type, initial_guess):
    colors, _, markers, _ = plotStylingStuff(color_type='eps')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for M in M_keys:
        for QI in QI_keys:
            fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 8.5))

            for e, eps in enumerate(eps_keys):
                residual = [u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['residual'][:, 1] for dt in dt_keys]

                ax.loglog(
                    dt_keys,
                    residual,
                    color=colors[e],
                    # linestyle=linestyles[QI],
                    marker=markers[e],
                    markeredgecolor='k',
                    linewidth=1.1,
                    label=rf'$\varepsilon=${eps}',
                )

            ax.set_ylabel(r'Residual norm', fontsize=16)
            ax.set_xlabel(r'Time step size $\Delta t$', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_ylim(1e-14, 1e-6)
            ax.set_xscale('log', base=10)
            ax.set_yscale('log', base=10)
            ax.grid(visible=True)
            ax.legend(frameon=False, fontsize=12, loc='upper left' , ncol=3)
            ax.minorticks_off()
            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotResidualDiffEpsDt_{quad_type}_QI={QI}_M={M}_initial_guess={initial_guess}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


def plotQuantitiesAlongIterations(u_num, prob_cls_name, eps_fix, maxiter):
    colors, linestyles, markers, _ = plotStylingStuff(color_type='QI')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    eps_list = eps_fix if eps_fix is not None else eps_keys
    print(eps_list, type(eps_list))
    for i, dt in enumerate(dt_keys):

        for M in M_keys:

            for eps_item in eps_list:
                fig, ax = plt_helper.plt.subplots(1, 1, figsize=(9.5, 7.5))
                fig2, ax2 = plt_helper.plt.subplots(1, 2, figsize=(17.5, 7.5))

                for QI in QI_keys:
                    res_post_iter = u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item]['residual_post_iter']
                    res_post_iter = NestedListIntoSingleList(res_post_iter)

                    ax.semilogy(
                        np.arange(1, len(res_post_iter) + 1),
                        res_post_iter,
                        color=colors[QI],
                        # linestyle=linestyles[QI],
                        marker=markers[QI],
                        markeredgecolor='k',
                        linewidth=1.1,
                        label=f'{QI}',
                    )

                    err_diff_iter = u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item]['e_global_post_iter']
                    err_diff_iter = NestedListIntoSingleList(err_diff_iter)

                    ax2[0].semilogy(
                        np.arange(1, len(err_diff_iter) + 1),
                        err_diff_iter,
                        color=colors[QI],
                        # linestyle=linestyles[QI],
                        marker=markers[QI],
                        markeredgecolor='k',
                        linewidth=1.1,
                        label=f'{QI}',
                    )

                    err_alg_iter = u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item]['e_global_algebraic_post_iter']
                    err_alg_iter = NestedListIntoSingleList(err_alg_iter)

                    ax2[1].semilogy(
                        np.arange(1, len(err_alg_iter) + 1),
                        err_alg_iter,
                        color=colors[QI],
                        # linestyle=linestyles[QI],
                        marker=markers[QI],
                        markeredgecolor='k',
                        linewidth=1.1,
                        label=f'{QI}',
                    )
                ax.set_ylabel(r'Residual norm', fontsize=16)
                ax.set_xlabel(r'Iteration $k$', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_xlim(0, maxiter + 1)
                ax.set_ylim(1e-15, 1e3)
                ax.set_yscale('log', base=10)
                ax.grid(visible=True)
                ax.legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
                ax.minorticks_off()

                ax2[0].set_ylabel(r'Error in $y$', fontsize=16)
                ax2[1].set_ylabel(r'Error in $z$', fontsize=16)
                for ax in [ax2[0], ax2[1]]:
                    ax.set_xlabel(r'Iteration $k$', fontsize=16)
                    ax.tick_params(axis='both', which='major', labelsize=16)
                    ax.set_xlim(0, maxiter + 1)
                    ax.set_ylim(1e-15, 1e3)
                    ax.set_yscale('log', base=10)
                    ax.grid(visible=True)
                    ax.legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
                    ax.minorticks_off()
                
                fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotResidualAlongIterations_eps={eps_item}_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
                plt_helper.plt.close(fig)
                fig2.savefig(f"data/{prob_cls_name}/eps_embedding/plotErrorAlongIterations_eps={eps_item}_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
                plt_helper.plt.close(fig2)


def plot_accuracy_order(u_num, prob_cls_name, quad_type, eps_fix=None, initial_guess=None):
    """
    Plots the order of accuracy for both differential and algebraic variable.

    Note that only for non-stiff systems the reached order of accuracy can be expected! In particular,
    the order could not be reached in the stiff component of the system but probably in the other
    non-stiff components. Has to be studied intensively!

    Considered keys: Only sweeper, QI, dt, M have to be lists! All other keys are not considered!

    Parameters
    ----------
    u_num : dict
        Contains the numerical solution as well as some other statistics for different parameters.
    prob_cls_name : str
        Name of the problem class.
    quad_type : str
        Type of quadrature used for integration in SDC.
    index : int
        Index of DAE problem.
    """

    colors, linestyles, markers, xytext = plotStylingStuff(color_type='eps')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    eps_list = eps_fix if eps_fix is not None else eps_keys

    type_err = ['e_global'] if prob_cls_name == 'CosineProblem' or prob_cls_name == 'ProtheroRobinson' else ['e_global', 'e_global_algebraic']
    for sweeper_cls_name in sweeper_keys:
        for QI in QI_keys:

            for M in M_keys:
                if prob_cls_name == 'CosineProblem' or prob_cls_name == 'ProtheroRobinson':
                    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 7.5))
                else:
                    fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 7.5))

                for eps_i, eps_item in enumerate(eps_list):

                    for ind, label in enumerate(type_err):          
                        res = [
                            u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item][label][:, 1]
                            for dt in dt_keys
                        ]

                        norms = [max(val) for val in res]
                        p_nonstiff = M - 1 #2 * M - 1
                        p_stiff = M - 1

                        # order_ref_nonstiff = [norms[-1] * (dt_keys[m] / dt_keys[-1]) ** p_nonstiff for m in range(len(dt_keys))]
                        order_ref_nonstiff = [(dt) ** p_nonstiff for dt in dt_keys]
                        # order_ref_stiff = [norms[-1] * (dt_keys[m] / dt_keys[-1]) ** p_stiff for m in range(len(dt_keys))]
                        ax_wrapper = ax if prob_cls_name == 'CosineProblem' or prob_cls_name == 'ProtheroRobinson' else ax[ind]

                        if eps_i == 0:
                            ax_wrapper.loglog(
                                dt_keys,
                                order_ref_nonstiff,
                                color='gray',
                                linestyle='--',
                                linewidth=0.9,
                                # label=f'Non-stiff ref. order $p={p_nonstiff}$',
                                label=f'Ref. order $p={p_nonstiff}$',
                            )
                        # elif eps_i == 2:
                        #     ax_wrapper.loglog(
                        #         dt_keys,
                        #         order_ref_stiff,
                        #         color='grey',
                        #         linestyle='dashdot',
                        #         linewidth=0.9,
                        #         label=f'Stiff ref. order $p={p_stiff}$',
                        #     )

                        ax_wrapper.loglog(
                            dt_keys,
                            norms,
                            color=colors[eps_i],
                            # linestyle=linestyles[M],
                            marker=markers[eps_i],
                            linewidth=0.8,
                            label=rf'$\varepsilon=${eps_item}',
                        )

                        mean_niters = [
                            u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item]['mean_niters']
                            for dt in dt_keys
                        ]
                        xytext_order = {
                            2: (0, 10),
                            3: (0, 12),
                            4: (0, -14),
                            5: (0, -12),
                        }

                        # for m in range(len(dt_keys)):
                        #     ax[ind].annotate(
                        #         int(mean_niters[m]),
                        #         (dt_keys[m], norms[m]),
                        #         # xytext=xytext_order[M],
                        #         textcoords="offset points",
                        #         color='black',
                        #         fontsize=12,
                        #     )

                        if ind == 0:
                            ylabel = r'Error norm $||e_{diff}||_\infty$'
                        else:
                            ylabel = r'Error norm $||e_{alg}||_\infty$'

                        ax_wrapper.tick_params(axis='both', which='major', labelsize=16)
                        ax_wrapper.set_ylim(1e-15, 1e1)#(1e-16, 1e-1)
                        ax_wrapper.set_xscale('log', base=10)
                        ax_wrapper.set_yscale('log', base=10)
                        ax_wrapper.set_xlabel(r'$\Delta t$', fontsize=16)
                        ax_wrapper.set_ylabel(ylabel, fontsize=16)
                        # ax_wrapper.grid(visible=True)
                        if prob_cls_name == 'CosineProblem' or prob_cls_name == 'ProtheroRobinson':
                            ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=3)
                        else:
                            ax[0].legend(frameon=False, fontsize=10, loc='upper left', ncol=3)
                        ax_wrapper.minorticks_off()

                fig.savefig(f"data/{prob_cls_name}/eps_embedding/plot_order_accuracy_{sweeper_cls_name}_{QI}_M={M}_initial_guess={initial_guess}.png", dpi=300, bbox_inches='tight')
                plt_helper.plt.close(fig)


def plotOrderAfterEachIteration(u_num, prob_cls_name, eps_fix=None):
    """
    Plots the order after each iteration for the differential variable and the algebraic variable.
    It is recommended to set restol to -1, s.t. maxiter iterations will be performed. Moreover,
    we know that SDC can reach a maximum order of 2*M - 1 for 'RADAU-RIGHT' nodes, maxiter can be
    set to 2 * M - 1.

    Note that only for non-stiff systems the reached order of accuracy can be expected! For stiff
    systems it also cannot be expected that the order is increased by one after one iteration!

    Considered keys: Only sweeper, dt, M have to be lists! All other keys are not considered!

    Parameters
    ----------
    u_num : dict
        Contains the numerical solution as well as some other statistics for different parameters.
    prob_cls_name : str
        Name of the problem class.
    """

    colors, linestyles, _, _ = plotStylingStuff(color_type='QI')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    eps_list = eps_fix if eps_fix is not None else eps_keys

    type_err = ['e_global_post_iter', 'e_global_algebraic_post_iter']
    for QI in QI_keys:

        for M in M_keys:

            for eps_i, eps_item in enumerate(eps_list):
                fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 7.5))

                for ind, label in enumerate(type_err):
                    if label == 'e_global_post_iter':
                        ylabel = r'Error norm $||\mathbf{u}_d - \mathbf{u}_d^k||_\infty$'
                    elif label == 'e_global_algebraic_post_iter':
                        ylabel = r'Error norm $||\mathbf{u}_a - \mathbf{u}_a^k||_\infty$'

                    res = [
                        u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item][label]
                        for dt in dt_keys
                    ]

                    k = 0
                    matrixIterAgainstDt = np.zeros((len(res[0]), len(dt_keys)))
                    for item in res:
                        err_iter = np.zeros(len(item))
                        # matrix that stores max. err in columns over all iterations for one dt, i.e., each row contains error for iteration k over all dt
                        for i, subitem in enumerate(item):
                            t, err = zip(*subitem)
                            err_iter[i] = max(err) # err_iter.append(max(err))

                        matrixIterAgainstDt[:, k] = err_iter
                        k += 1

                    for p in range(1, matrixIterAgainstDt.shape[0] + 1):
                        print(f"p={p}")
                        # order_ref = [matrixIterAgainstDt[p - 1, 0] * (dt / dt_keys[0]) ** p for dt in dt_keys]
                        order_ref = [(0.1 * dt) ** p for dt in dt_keys]
                        ax[ind].loglog(dt_keys, matrixIterAgainstDt[p - 1, :], '-*', label=f'Iteration {p}')
                        # ax[ind].loglog(dt_keys, order_ref, color='gray', '--')  # , label=f'Order {p} ref.')
                        ax[ind].loglog(dt_keys, order_ref, linestyle='--', color='gray')
                        if p == 2 * M - 1:
                            break
                        
                    ax[ind].tick_params(axis='both', which='major', labelsize=16)
                    ax[ind].set_ylim(1e-14, 1e1)
                    ax[ind].set_xscale('log', base=10)
                    ax[ind].set_yscale('log', base=10)
                    ax[ind].set_xlabel(r'$\Delta t$', fontsize=16)
                    ax[ind].set_ylabel(ylabel, fontsize=16)
                    # ax[ind].grid(visible=True)
                    if ind == 0:
                        ax[ind].legend(frameon=False, fontsize=10, loc='upper left')
                    ax[ind].minorticks_off()

                fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotOrderIterations_{QI}_M={M}_eps={eps_item}.png", dpi=300, bbox_inches='tight')
                plt_helper.plt.close(fig)


def plotCosts(u_num, prob_cls_name, initial_guess):
    r"""
    Plots the number of Newton iterations to solve implicit system at each node and number of SDC iterations.


    """
    colors, linestyles, markers, _ = plotStylingStuff(color_type='M')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for i, dt in enumerate(dt_keys):

        for QI in QI_keys:
            fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 7.5))

            for M in M_keys:
                nfev_root_solver = [
                    u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item]['newton'][:, 1]
                    for eps_item in eps_keys
                ]
                nfev_root_sum = [sum(item) for item in nfev_root_solver]
                # nfev_root_sum = np.sum(nfev_root_solver)
                ax[0].semilogx(
                    eps_keys,
                    nfev_root_sum,
                    color=colors[M],
                    linestyle=linestyles[M],
                    marker=markers[M],
                    linewidth=0.8,
                    label=f'$M={M}$',
                )

                sum_niters = [
                    u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item]['sum_niters']
                    for eps_item in eps_keys
                ]
                ax[1].semilogx(
                    eps_keys,
                    sum_niters,
                    color=colors[M],
                    linestyle=linestyles[M],
                    marker=markers[M],
                    linewidth=0.8,
                    label=f'$M={M}$',
                )

            ax[0].set_ylabel(f"Number of Newton iterations", fontsize=16)
            ax[1].set_ylabel(f"Number of SDC iterations", fontsize=16)
            ax[0].legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
            for ind in [0, 1]:
                ax[ind].set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
                ax[ind].tick_params(axis='both', which='major', labelsize=16)
                # ax[ind].set_ylim(1e-15, 1e3)
                ax[ind].set_xscale('log', base=10)
                ax[ind].grid(visible=True)
                ax[ind].minorticks_off()
            
            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotCosts_QI={QI}_dt={dt}_initial_guess={initial_guess}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


def plotCostsDt(u_num, prob_cls_name, initial_guess):
    r"""
    Plots the number of Newton iterations to solve implicit system at each node and number of SDC iterations.


    """
    colors, _, markers, _ = plotStylingStuff(color_type='dt')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for M in M_keys:    

        for QI in QI_keys:
            fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 7.5))

            for i, dt in enumerate(dt_keys):
                nfev_root_solver = [
                    u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item]['newton'][:, 1]
                    for eps_item in eps_keys
                ]
                nfev_root_sum = [sum(item) for item in nfev_root_solver]
                # nfev_root_sum = np.sum(nfev_root_solver)
                ax[0].semilogx(
                    eps_keys,
                    nfev_root_sum,
                    color=colors[i],
                    # linestyle=linestyles[M],
                    marker=markers[i],
                    linewidth=0.8,
                    label=rf'$\Delta t=${dt}',
                )

                sum_niters = [
                    u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps_item]['sum_niters']
                    for eps_item in eps_keys
                ]
                ax[1].semilogx(
                    eps_keys,
                    sum_niters,
                    color=colors[i],
                    # linestyle=linestyles[M],
                    marker=markers[i],
                    linewidth=0.8,
                    label=rf'$\Delta t=${dt}',
                )

            ax[0].set_ylabel(f"Number of Newton iterations", fontsize=16)
            ax[1].set_ylabel(f"Number of SDC iterations", fontsize=16)
            ax[0].legend(frameon=False, fontsize=12, loc='upper right' , ncol=2)
            for ind in [0, 1]:
                ax[ind].set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
                ax[ind].tick_params(axis='both', which='major', labelsize=16)
                # ax[ind].set_ylim(1e-15, 1e3)
                ax[ind].set_xscale('log', base=10)
                ax[ind].grid(visible=True)
                ax[ind].minorticks_off()
            
            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotCosts_QI={QI}_M={M}_initial_guess={initial_guess}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


def plotCostsEps(u_num, prob_cls_name, initial_guess):
    r"""
    Plots the number of Newton iterations to solve implicit system at each node and number of SDC iterations.
    """
    colors, _, markers, _ = plotStylingStuff(color_type='eps')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha, eps_keys = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for M in M_keys:    

        for QI in QI_keys:
            # fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 7.5))
            fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 8.5))

            for e, eps in enumerate(eps_keys):
                nfev_root_solver = [
                    u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['newton'][:, 1]
                    for dt in dt_keys
                ]
                nfev_root_sum = [sum(item) for item in nfev_root_solver]
                # nfev_root_sum = np.sum(nfev_root_solver)
                # ax[0].semilogx(
                #     dt_keys,
                #     nfev_root_sum,
                #     color=colors[e],
                #     # linestyle=linestyles[M],
                #     marker=markers[e],
                #     linewidth=0.8,
                #     label=rf'$\varepsilon=${eps}',
                # )

                sum_niters = [
                    u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][eps]['sum_niters']
                    for dt in dt_keys
                ]
                # ax[1].semilogx(
                ax.semilogx(
                    dt_keys,
                    sum_niters,
                    color=colors[e],
                    # linestyle=linestyles[M],
                    marker=markers[e],
                    linewidth=0.8,
                    label=rf'$\varepsilon=${eps}',
                )

            # ax[0].set_ylabel(f"Number of Newton iterations", fontsize=16)
            ax.set_ylabel(f"Number of SDC iterations", fontsize=16)
            ax.legend(frameon=False, fontsize=10, loc='upper right' , ncol=3)
            # for ind in [0, 1]:
                # ax[ind].set_xlabel(r'$\Delta t$', fontsize=16)
                # ax[ind].tick_params(axis='both', which='major', labelsize=16)
                # # ax[ind].set_ylim(1e-15, 1e3)
                # ax[ind].set_xscale('log', base=10)
                # ax[ind].grid(visible=True)
                # ax[ind].minorticks_off()
            ax.set_xlabel(r'$\Delta t$', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.set_ylim(1, 130)
            ax.set_xscale('log', base=10)
            ax.grid(visible=True)
            ax.minorticks_off()

            fig.savefig(f"data/{prob_cls_name}/eps_embedding/plotCostsEpsDt_QI={QI}_M={M}_initial_guess={initial_guess}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)



if __name__ == "__main__":
    main()