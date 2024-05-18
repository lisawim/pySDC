import numpy as np

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.DAE.sweepers.genericImplicitEmbedded import genericImplicitEmbedded, genericImplicitEmbedded2

from pySDC.implementations.problem_classes.singularPerturbed import (
    EmbeddedLinearTestDAE,
    EmbeddedLinearTestDAEMinion,
)
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEIntegralFormulation,
    LinearTestDAEIntegralFormulation2,
    LinearTestDAEMinionIntegralFormulation,
    LinearTestDAEMinionIntegralFormulation2,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import (
    DiscontinuousTestDAEIntegralFormulation,
    DiscontinuousTestDAEIntegralFormulation2,
)
from pySDC.projects.DAE.problems.simple_DAE import (
    simple_dae_1IntegralFormulation,
    simple_dae_1IntegralFormulation2,
    pendulum_2dIntegralFormulation,
    pendulum_2dIntegralFormulation2,
    LinearIndexTwoDAEIntegralFormulation,
    LinearIndexTwoDAEIntegralFormulation2,
)

import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.DAE.run.study_SPP import plotStylingStuff, NestedListIntoSingleList
from pySDC.projects.DAE.run.plot_order import runSimulationStudy, getDataDictKeys

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.DAE.misc.HookClass_DAE import (
    LogGlobalErrorPostIterDiff,
    LogGlobalErrorPostIterAlg,
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
        'restol': 1e-12,
        'maxiter': 180,#600,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    problem = [
        EmbeddedLinearTestDAE,
        # LinearTestDAEIntegralFormulation,
        LinearTestDAEIntegralFormulation2,
        # LinearTestDAEMinionIntegralFormulation,
        # LinearTestDAEMinionIntegralFormulation2,
        # simple_dae_1IntegralFormulation,
        # simple_dae_1IntegralFormulation2,
        # DiscontinuousTestDAEIntegralFormulation,
        # DiscontinuousTestDAEIntegralFormulation2,
        # pendulum_2dIntegralFormulation,
        # pendulum_2dIntegralFormulation2,
        # LinearIndexTwoDAEIntegralFormulation,
        # LinearIndexTwoDAEIntegralFormulation2,
    ]
    prob_cls_name = problem[0].__name__
    sweeper = [
        generic_implicit,
        # genericImplicitEmbedded,
        genericImplicitEmbedded2,
    ]

    problem_params = {
        'newton_tol': 1e-15,
    }

    # be careful with hook classes - for error classes choose only the necessary ones!
    hook_class = [
        DefaultHooks,
        LogSolution,
        # LogGlobalErrorPostStep,
        # LogSolutionAfterIteration,
        # LogWork,
        # LogGlobalErrorPostIterDiff,
        # LogGlobalErrorPostIterAlg,
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
    dt_list = np.logspace(-4.0, -0.5, num=10)
    t1 = 1.5  # note that the interval will be ignored if compute_one_step=True!

    compute_one_step = True
    handling_params.update({'compute_one_step': compute_one_step})

    u_num = runSimulationStudy(
        problem=problem,
        sweeper=sweeper,
        all_params=all_params,
        use_adaptivity=use_adaptivity,
        use_detection=use_detection,
        hook_class=hook_class,
        interval=(t0, t1),
        dt_list=dt_list,
        nnodes=[2, 3, 4, 5],
    )

    plotCompareIterationsEmbeddedMethods(u_num, prob_cls_name, handling_params['maxiter'])


def plotCompareIterationsEmbeddedMethods(u_num, prob_cls_name, maxiter):
    colors, _, _, _ = plotStylingStuff(color_type='M')

    linestyle = {
        'generic_implicit': 'solid',
        'genericImplicitEmbedded': 'solid',
        'genericImplicitEmbedded2': 'dashdot',
    }

    marker = {
        'generic_implicit': '*',
        'genericImplicitEmbedded': 'o',
        'genericImplicitEmbedded2' : 's',
    }

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for QI in QI_keys:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(9.5, 9.5))

        for sweeper_cls_name in sweeper_keys:
            for M in M_keys:    
                niters = [
                    u_num[sweeper_cls_name][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]]['mean_niters']
                    for dt in dt_keys
                ]

                if sweeper_cls_name == 'genericImplicitEmbedded':
                    label = rf'Embedded-SDC1 - $M={M}$'
                elif sweeper_cls_name == 'genericImplicitEmbedded2':
                    label = rf'Embedded-SDC2 - $M={M}$'
                elif sweeper_cls_name == 'generic_implicit':
                    label = rf'SDC - $M={M}$'
                else:
                    raise NotImplementedError()

                ax.semilogx(
                    dt_keys,
                    niters,
                    color=colors[M],
                    linestyle=linestyle[sweeper_cls_name],
                    marker=marker[sweeper_cls_name],
                    linewidth=0.8,
                    label=label,
                )
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1, maxiter + 50)
        ax.set_xscale('log', base=10)
        # ax.set_yscale('log', base=10)
        ax.set_xlabel(r'$\Delta t$', fontsize=16)
        ax.set_ylabel('Mean number of iterations', fontsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=False, fontsize=12, loc='upper left', ncol=2)
        ax.minorticks_off()

        fig.savefig(f"data/{prob_cls_name}/plotIterationsEmbeddedMethods_{QI}.png", dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


if __name__ == "__main__":
    main()
