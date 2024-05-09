import numpy as np
from pathlib import Path
import cProfile

from pySDC.core.Errors import ParameterError

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.RungeKuttaDAE import (
    BackwardEulerDAE,
    TrapezoidalRuleDAE,
    KurdiEDIRK45_2DAE,
    EDIRK4DAE,
    DIRK43_2DAE,
    # semi-implicit variants
    SemiImplicitBackwardEulerDAE,
    SemiImplicitTrapezoidalRuleDAE,
    SemiImplicitKurdiEDIRK45_2DAE,
    SemiImplicitEDIRK4DAE,
    SemiImplicitDIRK43_2DAE,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1, problematic_f
from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAE

import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun, getDataDict, plotSolution

from pySDC.implementations.hooks.log_solution import LogSolution, LogSolutionAfterIteration
from pySDC.projects.DAE.misc.HookClass_DAE import (
    LogGlobalErrorPostStepDifferentialVariable,
    LogGlobalErrorPostStepAlgebraicVariable,
)
from pySDC.implementations.hooks.default_hook import DefaultHooks
from pySDC.implementations.hooks.log_work import LogWork
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep


def main():
    r"""
    Executes the simulation.
    """

    # defines parameters for sweeper
    M_fix = 3
    quad_type = 'RADAU-RIGHT'
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': quad_type,
        'QI': ['IE'],
        'initial_guess': 'spread',
    }

    # defines parameters for event detection, restol, and max. number of iterations
    handling_params = {
        'restol': -1,
        'maxiter': 1,
        'compute_one_step': False,#True,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    problem = LinearTestDAE
    prob_cls_name = problem.__name__
    sweeper = [SemiImplicitBackwardEulerDAE]

    problem_params = {
        'newton_tol': 1e-15,
    }

    prob = problem(**problem_params)
    isSemiExplicit = True if prob.dtype_u.__name__ == 'DAEMesh' else False

    hook_class = [
        DefaultHooks,
        LogSolution,
        # LogGlobalErrorPostStep,
        # LogSolutionAfterIteration,
        # LogWork,
        LogGlobalErrorPostStepDifferentialVariable,
        LogGlobalErrorPostStepAlgebraicVariable,
    ]

    all_params = {
        'sweeper_params': sweeper_params,
        'handling_params': handling_params,
        'problem_params': problem_params,
    }

    use_detection = [False]
    use_adaptivity = [False]

    index = {
        'LinearTestDAE': 1,
        'LinearTestDAEMinion': 1,
        'DiscontinuousTestDAE': 1,
        'simple_dae_1': 2,
        'problematic_f': 2,
    }

    t0 = 0.0
    dt_list = np.logspace(-2.0, -0.5, num=20)
    t1 = 3.0  # note that the interval will be ignored if compute_one_step=True!

    u_num = runSimulationStudy(
        problem=problem,
        sweeper=sweeper,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(t0, t1),
        dt_list=dt_list,
        nnodes=[3],
    )

    plot_accuracy_order(u_num, prob_cls_name, quad_type, index, isSemiExplicit)

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
    Path(f"data/{prob_cls_name}/Talk").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}/Talk/Costs").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}/Talk/Errors").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}/Talk/Eigenvalues").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}/Talk/Manifold").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}/Costs").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}/Errors").mkdir(parents=True, exist_ok=True)
    Path(f"data/{prob_cls_name}/Eigenvalues").mkdir(parents=True, exist_ok=True)

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

    for sweeper in sweeper_classes:
        sweeper_cls_name = sweeper.__name__
        u_num[sweeper_cls_name] = {}
        print(f'Controller run for {sweeper_cls_name}')

        for QI in preconditioners:
            u_num[sweeper_cls_name][QI] = {}

            for dt in dt_list:
                u_num[sweeper_cls_name][QI][dt] = {}
                print(dt)
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

                                        problem_params = all_params['problem_params']
                                        sweeper_params = all_params['sweeper_params']
                                        handling_params = all_params['handling_params']

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

                                        u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha] = getDataDict(
                                            stats, prob_cls_name, use_A, use_SE, handling_params['recomputed'], t_switch_exact
                                        )

                                        if prob_cls_name == 'LinearTestDAE':
                                            plot_labels = None
                                        elif prob_cls_name == 'LinearTestDAEMinion':
                                            plot_labels = None
                                        elif prob_cls_name == 'simple_dae_1':
                                            plot_labels = None
                                        elif prob_cls_name == 'WSCC9BusSystem':
                                            plot_labels = [r'$P_{SV,0}$', r'$P_{SV,1}$', r'$P_{SV,2}$', r'$I_{d,0}$', r'$I_{d,1}$', r'$I_{d,2}$',]
                                        elif prob_cls_name == 'PilineDAE':
                                            plot_labels = [r'$v_{C_1}$', r'$v_{C_2}$', r'$i_{L_\pi}$']
                                        else:
                                            plot_labels = None

                                        plotSolution(
                                            u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha],
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

    return sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha

def plot_accuracy_order(u_num, prob_cls_name, quad_type, index, isSemiExplicit):
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

    colors, linestyles, markers, xytext = plotStylingStuff(color_type='M')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha = getDataDictKeys(u_num)

    type_err = ['e_global', 'e_global_algebraic'] if isSemiExplicit else ['e_global']
    for sweeper_cls_name in sweeper_keys:
        for QI in QI_keys:
            if isSemiExplicit:
                fig, ax = plt_helper.plt.subplots(1, 2, figsize=(15.5, 5))
            else:
                fig, ax = plt_helper.plt.subplots(1, 1, figsize=(8.5, 8.5))

            for ind, label in enumerate(type_err):
                ax_wrapper = ax[ind] if isSemiExplicit else ax         
                for M in M_keys:
                    res = [
                        u_num[sweeper_cls_name][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][label][:, 1]
                        for dt in dt_keys
                    ]

                    norms = [max(val) for val in res]
                    # print(f'For M={M} the errors {label} are: {norms}')

                    # if index[prob_cls_name] == 1:
                    #     if label == 'e_global':
                    #         p = 2 * M - 1 if quad_type == 'RADAU-RIGHT' else (2 * M - 2 if quad_type == 'LOBATTO' else 2 * M - 1)
                    #     else:
                    #         p = 2 * M - 1 if quad_type == 'RADAU-RIGHT' else (2 * M - 2 if quad_type == 'LOBATTO' else M)
                    # elif index[prob_cls_name] == 2:
                    #     if label == 'e_global':
                    #         p = 2 * M - 1 if quad_type == 'RADAU-RIGHT' else (2 * M - 2 if quad_type == 'LOBATTO' else M)
                    #     else:
                    #         if quad_type == 'RADAU-RIGHT':
                    #             p = M
                    #         elif quad_type == 'LOBATTO':
                    #             if M % 2 == 0:
                    #                 p = M
                    #             else:
                    #                 p = M - 1
                    #         elif quad_type == 'RADAU-LEFT':
                    #             p = M - 1
                    # else:
                    #     raise NotImplementedError(f"Reference order for index {index[prob_cls_name]} and quadrature type {quad_type} not implemented!")
                    # p = 2 * M - 1
                    p = 1
                    order_ref = [norms[-1] * (dt_keys[m] / dt_keys[-1]) ** p for m in range(len(dt_keys))]

                    ax_wrapper.loglog(
                        dt_keys,
                        norms,
                        color=colors[M],
                        linestyle=linestyles[M],
                        marker=markers[M],
                        linewidth=0.8,
                        label=f'$M={M}$',
                    )

                    ax_wrapper.loglog(dt_keys, order_ref, color='k', linestyle=linestyles[M], label=f'Ref. order $p={p}$')

                    mean_niters = [
                        u_num[sweeper_cls_name][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]]['mean_niters']
                        for dt in dt_keys
                    ]
                    xytext_order = {
                        2: (0, 10),
                        3: (0, 12),
                        4: (0, -14),
                        5: (0, -12),
                    }

                    for m in range(len(dt_keys)):
                        ax_wrapper.annotate(
                            int(mean_niters[m]),
                            (dt_keys[m], norms[m]),
                            xytext=xytext_order[M],
                            textcoords="offset points",
                            color='black',
                            fontsize=12,
                        )


                if isSemiExplicit:
                    if ind == 0:
                        ylabel = r'Error norm $||e_{diff}||_\infty$'
                    else:
                        ylabel = r'Error norm $||e_{alg}||_\infty$'
                else:
                    ylabel = r'Error norm $||e||_\infty$'

                ax_wrapper.tick_params(axis='both', which='major', labelsize=16)
                ax_wrapper.set_ylim(1e-5, 1e5)
                ax_wrapper.set_xscale('log', base=10)
                ax_wrapper.set_yscale('log', base=10)
                ax_wrapper.set_xlabel(r'$\Delta t$', fontsize=16)
                ax_wrapper.set_ylabel(ylabel, fontsize=16)
                ax_wrapper.grid(visible=True)
                ax_wrapper.legend(frameon=False, fontsize=12, loc='upper left')
                ax_wrapper.minorticks_off()

            fig.savefig(f"data/{prob_cls_name}/plot_order_accuracy_{sweeper_cls_name}_{QI}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


if __name__ == "__main__":
    main()
