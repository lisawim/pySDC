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

from pySDC.projects.DAE.run.study_SPP import plotStylingStuff
from pySDC.projects.DAE.run.plot_order import runSimulationStudy, getDataDictKeys

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.DAE.misc.HookClass_DAE import (
    LogGlobalErrorPostStepDifferentialVariable,
    LogGlobalErrorPostStepAlgebraicVariable,
)
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
        'maxiter': 130,#600,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    problem = LinearTestDAEMinionIntegralFormulation
    prob_cls_name = problem.__name__
    sweeper = [genericImplicitEmbedded]

    problem_params = {
        'newton_tol': 1e-15,
    }

    prob = problem(**problem_params)
    isSemiExplicit = True if prob.dtype_u.__name__ == 'DAEMesh' else False

    # be careful with hook classes - for error classes choose only the necessary ones!
    hook_class = [
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
    dt_list = [1e-2]#np.logspace(-2.0, -1.1, num=10)
    t1 = 5.0  # note that the interval will be ignored if compute_one_step=True!
    # nSteps = range(32, 64, 4)
    # dt_list = [(t1 - t0) / n for n in nSteps]
    compute_one_step = True#False
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
        nnodes=[2],
    )

    # plotSolutionOnManifold(u_num, prob_cls_name, problem, problem_params, compute_one_step)


def plotSolutionOnManifold(u_num, prob_cls_name, problem, problem_params, compute_one_step):
    r"""
    Plots the value of the algebraic constraints after computation.
    """

    colors, linestyles, markers, xytext = plotStylingStuff(color_type='M')

    sweeper_keys, QI_keys, dt_keys, M_keys, use_SE_keys, use_A_keys, newton_tolerances, tol_event, alpha = getDataDictKeys(u_num)

    for sweeper_cls_name in sweeper_keys:
        for QI in QI_keys:
            for M in M_keys:
                fig, ax = plt_helper.plt.subplots(1, 1, figsize=(9.5, 9.5))

                prob = problem(**problem_params)
                uEnd = [
                    u_num[sweeper_cls_name][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]]['u'][-1]
                    for dt in dt_keys
                ]

                tEnd = [u_num[sweeper_cls_name][QI][dt][M][use_SE_keys[0]][use_A_keys[0]][newton_tolerances[0]][tol_event[0]][alpha[0]]['t'][-1]
                    for dt in dt_keys
                ]

                if compute_one_step:
                    g = [max(prob.eval_f(u, dt).alg[:]) for u, dt in zip(uEnd, dt_keys)]
                else:
                    g = [max(prob.eval_f(u, t1).alg[:]) for u, t1 in zip(uEnd, tEnd)]

                ax.loglog(
                    dt_keys,
                    g,
                    color=colors[M],
                    linestyle=linestyles[M],
                    marker=markers[M],
                    linewidth=0.8,
                    label='Value of algebraic constraints',
                )

                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_xscale('log', base=10)
                # ax.set_yscale('log', base=10)
                ax.set_yscale('symlog', linthresh=1e-13)
                ax.set_ylim(-1e-4, 1e-4)
                ax.set_xlabel(r'$\Delta t$', fontsize=16)
                ax.set_ylabel(r'$|g(y, z)|$', fontsize=16)
                ax.grid(visible=True)
                ax.legend(frameon=False, fontsize=12, loc='upper left')
                ax.minorticks_off()

                fig.savefig(f"data/{prob_cls_name}/plot_solution_manifold_{sweeper_cls_name}_{QI}_M={M}.png", dpi=300, bbox_inches='tight')
                plt_helper.plt.close(fig)       


if __name__ == "__main__":
    main()
