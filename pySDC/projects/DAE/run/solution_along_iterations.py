import numpy as np

from pySDC.projects.DAE.sweepers.genericImplicitEmbedded import genericImplicitEmbedded, genericImplicitEmbedded2
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEIntegralFormulation,
    LinearTestDAEIntegralFormulation2,
    LinearTestDAEMinionIntegralFormulation,
    LinearTestDAEMinionIntegralFormulation2,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEIntegralFormulation
from pySDC.projects.DAE.problems.simple_DAE import (
    simple_dae_1IntegralFormulation,
    simple_dae_1IntegralFormulation2,
    LinearIndexTwoDAEIntegralFormulation,
    LinearIndexTwoDAEIntegralFormulation2,
    pendulum_2dIntegralFormulation,
    pendulum_2dIntegralFormulation2,
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
        'maxiter': 100,#600,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    problem = LinearIndexTwoDAEIntegralFormulation2
    prob_cls_name = problem.__name__
    sweeper = [genericImplicitEmbedded2]

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
        LogGlobalErrorPostIterDiff,
        LogGlobalErrorPostIterAlg,
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
    dt_list = np.logspace(-2.0, -0.5, num=30)
    t1 = 0.5  # note that the interval will be ignored if compute_one_step=True!

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
        nnodes=[5],
    )

    plotQuantitiesAlongIterations(u_num, prob_cls_name, handling_params['maxiter'])


def plotQuantitiesAlongIterations(u_num, prob_cls_name, maxiter):
    colors, linestyles, markers, _ = plotStylingStuff(color_type='QI')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha = getDataDictKeys(u_num)

    QI_keys = QI_keys if len(QI_keys) > 1 else [QI_keys[0]]
    M_keys = M_keys if len(M_keys) > 1 else [M_keys[0]]

    for i, dt in enumerate(dt_keys):

        for M in M_keys:

            fig, ax = plt_helper.plt.subplots(1, 1, figsize=(9.5, 7.5))
            fig2, ax2 = plt_helper.plt.subplots(1, 2, figsize=(17.5, 7.5))

            for QI in QI_keys:
                res_post_iter = u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]]['residual_post_iter']
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

                err_diff_iter = u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]]['e_global_post_iter']
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

                err_alg_iter = u_num[sweeper_keys[0]][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]]['e_global_algebraic_post_iter']
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
            ax.set_ylim(1e-17, 1e3)
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
            
            fig.savefig(f"data/{prob_cls_name}/plotResidualAlongIterations_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig)
            fig2.savefig(f"data/{prob_cls_name}/plotErrorAlongIterations_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
            plt_helper.plt.close(fig2)


if __name__ == "__main__":
    main()
