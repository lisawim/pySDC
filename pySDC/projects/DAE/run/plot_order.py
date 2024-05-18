import numpy as np
from pathlib import Path

from pySDC.core.Errors import ParameterError

from pySDC.projects.DAE.sweepers.genericImplicitEmbedded import genericImplicitEmbedded, genericImplicitEmbedded2
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEIntegralFormulation,
    LinearTestDAEIntegralFormulation2,
    LinearTestDAEMinionIntegralFormulation,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEIntegralFormulation
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1IntegralFormulation

import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun, getDataDict, plotSolution
from pySDC.projects.DAE.run.study_SPP import getOrder, plotStylingStuff

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
    M_fix = 2
    quad_type = 'RADAU-RIGHT'
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': quad_type,
        'QI': ['IE'],
        'initial_guess': 'spread',
    }

    # defines parameters for event detection, restol, and max. number of iterations
    handling_params = {
        'restol': 1e-15,
        'maxiter': 150,#600,
        'compute_one_step': True,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    problem = LinearTestDAEIntegralFormulation
    prob_cls_name = problem.__name__
    sweeper = [genericImplicitEmbedded]

    problem_params = {
        'newton_tol': 1e-15,
    }

    prob = problem(**problem_params)
    isSemiExplicit = True if prob.dtype_u.__name__ == 'DAEMesh' else False

    # be careful with hook classes - for error classes choose only the necessary ones!
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

    t0 = 1.0
    dt_list = np.logspace(-3.0, -0.5, num=30)
    t1 = 1.5  # note that the interval will be ignored if compute_one_step=True!

    u_num = runSimulationStudy(
        problem=problem,
        sweeper=sweeper,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(t0, t1),
        dt_list=dt_list,
        nnodes=[5],
    )

    plot_accuracy_order(u_num, prob_cls_name, quad_type, index, isSemiExplicit, (t0, t1))

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

    if not isinstance(problem, list) and isinstance(problem.__name__, str):
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

    elif isinstance(problem, list):
        for prob in problem:
            Path("data").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}/Talk").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}/Talk/Costs").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}/Talk/Errors").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}/Talk/Eigenvalues").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}/Talk/Manifold").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}/Costs").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}/Errors").mkdir(parents=True, exist_ok=True)
            Path(f"data/{prob.__name__}/Eigenvalues").mkdir(parents=True, exist_ok=True)

    u_num = {}

    t0 = interval[0]
    Tend = interval[-1]

    problem_classes = problem if isinstance(problem, list) else [problem]
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

    for problem, sweeper in zip(problem_classes, sweeper_classes):
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

                                        prob_cls_name = problem.__name__

                                        restol = -1 if use_A else handling_params['restol']

                                        problem_params.update({'newton_tol': newton_tol})

                                        eps = 1e-10
                                        problem_params.update({'eps': eps})

                                        if not sweeper_cls_name == 'generic_implicit':
                                            if problem_params.get('eps') is not None:
                                                del problem_params['eps']

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
                                            stats, prob_cls_name, use_A, use_SE, handling_params['recomputed'], t_switch_exact, maxiter
                                        )

                                        plotSolution(
                                            u_num[sweeper_cls_name][QI][dt][M][use_SE][use_A][newton_tol][tol_event][alpha],
                                            prob_cls_name,
                                            sweeper.__name__,
                                            use_A,
                                            use_SE,
                                        )
    return u_num


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

def plot_accuracy_order(u_num, prob_cls_name, quad_type, index, isSemiExplicit, interval):
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
                fig, ax = plt_helper.plt.subplots(1, 2, figsize=(16.5, 9.5))
                ax[0].set_title(rf"Order for $t_0=${interval[0]} to $T=${interval[-1]}")
                ax[1].set_title(rf"Perturbed order for $t_0=${interval[0]} to $T=${interval[-1]}")
            else:
                fig, ax = plt_helper.plt.subplots(1, 1, figsize=(9.5, 9.5))
                ax.set_title(rf"Order for $t_0=${interval[0]} to $T=${interval[-1]}")

            for ind, label in enumerate(type_err):
                ax_wrapper = ax[ind] if isSemiExplicit else ax         
                for M in M_keys:
                    res = [
                        u_num[sweeper_cls_name][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]][label][:, 1]
                        for dt in dt_keys
                    ]

                    norms = [max(val) for val in res]

                    p = getOrder(prob_cls_name, label, M)
                    order_ref = [(dt) ** p for dt in dt_keys]

                    ax_wrapper.loglog(
                        dt_keys,
                        norms,
                        color=colors[M],
                        linestyle=linestyles[M],
                        marker=markers[M],
                        linewidth=0.8,
                        label=f'$M={M}$',
                    )

                    ax_wrapper.loglog(
                        dt_keys,
                        order_ref,
                        color='gray',
                        linestyle='--',
                        linewidth=0.9,
                        label=f'Ref. order $p={p}$',
                    )

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
                ax_wrapper.set_ylim(1e-16, 1e1)
                ax_wrapper.set_xscale('log', base=10)
                ax_wrapper.set_yscale('log', base=10)
                ax_wrapper.set_xlabel(r'$\Delta t$', fontsize=16)
                ax_wrapper.set_ylabel(ylabel, fontsize=16)
                ax_wrapper.grid(visible=True)
                ax_wrapper.legend(frameon=False, fontsize=12, loc='upper left')
                ax_wrapper.minorticks_off()

            fig.savefig(f"data/{prob_cls_name}/plot_order_accuracy_{sweeper_cls_name}_{QI}_M={M}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)


if __name__ == "__main__":
    main()
