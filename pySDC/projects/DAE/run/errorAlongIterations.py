import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import (
    EmbeddedLinearTestDAE,
    EmbeddedLinearTestDAEMinion,
    EmbeddedDiscontinuousTestDAE,
)

from pySDC.projects.DAE.sweepers.genericImplicitEmbedded import genericImplicitEmbedded2
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEIntegralFormulation2,
    LinearTestDAEMinionIntegralFormulation2,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEIntegralFormulation2

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostIter, LogGlobalErrorPostIterPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg

from pySDC.helpers.stats_helper import get_sorted


def nestedListIntoSingleList(res):
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


def main():
    problems = [
        EmbeddedLinearTestDAE,
        LinearTestDAEIntegralFormulation2,
    ]
    sweepers = [generic_implicit, genericImplicitEmbedded2]

    # sweeper params
    M = 4
    quad_type = 'RADAU-RIGHT'
    QI = 'IE'

    # parameters for convergence
    nSweeps = 60

    # hook class to be used
    hook_class = {
        'generic_implicit': [LogGlobalErrorPostIter, LogGlobalErrorPostIterPerturbation],
        'genericImplicitEmbedded2': [
            LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg
        ],
    }

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-14

    epsList = [1e-3, 1e-4, 1e-5]
    epsValues = {
        'generic_implicit': epsList,
        'genericImplicitEmbedded2': [0.0],
    }

    t0 = 0.0
    dtValues = [10 ** (-m) for m in range(1, 10)]

    colors = [
        'lightblue',
        'deepskyblue',
        'dodgerblue',
        'royalblue',
        'mediumblue',
        'lightgray',
        'darkgray',
        'gray',
        'dimgray',
    ]

    for problem, sweeper in zip(problems, sweepers):
        epsLoop = epsValues[sweeper.__name__]

        for eps in epsLoop:
            fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
            figRes, axRes = plt.subplots(1, 1, figsize=(9.5, 9.5))

            for i, dt in enumerate(dtValues):
                print(eps, dt)
                problem_params = {
                    'newton_tol': newton_tol,
                }
                if not eps == 0.0:
                    problem_params = {'eps': eps}

                restol = 1e-12 if not eps == 0.0 else 1e-13

                description, controller_params, controller = generateDescription(
                    dt=dt,
                    problem=problem,
                    sweeper=sweeper,
                    num_nodes=M,
                    quad_type=quad_type,
                    QI=QI,
                    hook_class=hook_class[sweeper.__name__],
                    use_adaptivity=use_A,
                    use_switch_estimator=use_SE,
                    problem_params=problem_params,
                    restol=restol,
                    maxiter=nSweeps,
                    max_restarts=max_restarts,
                    tol_event=tol_event,
                    alpha=alpha,
                )

                stats, _ = controllerRun(
                    description=description,
                    controller_params=controller_params,
                    controller=controller,
                    t0=t0,
                    Tend=dt,
                    exact_event_time_avail=None,
                )

                errDiffIter = [get_sorted(stats, iter=k, type='e_global_post_iter', sortby='time') for k in range(1, nSweeps + 1)]
                errAlgIter = [get_sorted(stats, iter=k, type='e_global_algebraic_post_iter', sortby='time') for k in range(1, nSweeps + 1)]
                resIter = [get_sorted(stats, iter=k, type='residual_post_iteration', sortby='time') for k in range(1, nSweeps + 1)]

                errDiffIter = nestedListIntoSingleList(errDiffIter)
                errAlgIter = nestedListIntoSingleList(errAlgIter)
                resIter = nestedListIntoSingleList(resIter)

                ax[0].set_title(rf"Differential error for $\varepsilon=${eps}")
                ax[0].semilogy(
                    np.arange(1, len(errDiffIter) + 1),
                    errDiffIter,
                    color=colors[i],
                    marker='*',
                    markersize=10.0,
                    linewidth=4.0,
                    solid_capstyle='round',
                    label=rf'$\Delta t=${dt}',
                )

                ax[1].set_title(rf"Algebraic error for $\varepsilon=${eps}")
                ax[1].semilogy(
                    np.arange(1, len(errAlgIter) + 1),
                    errAlgIter,
                    color=colors[i],
                    marker='*',
                    markersize=10.0,
                    linewidth=4.0,
                    solid_capstyle='round',
                )

                axRes.set_title(rf"Residual for $\varepsilon=${eps}")
                axRes.semilogy(
                    np.arange(1, len(resIter) + 1),
                    resIter,
                    color=colors[i],
                    marker='*',
                    markersize=10.0,
                    linewidth=4.0,
                    solid_capstyle='round',
                    label=rf'$\Delta t=${dt}',
                )

            for ax_wrapper in [ax[0], ax[1], axRes]:
                ax_wrapper.tick_params(axis='both', which='major', labelsize=14)
                ax_wrapper.set_xlabel(r'Iteration $k$', fontsize=20)
                ax_wrapper.set_xlim(0, nSweeps + 1)
                if not ax_wrapper == axRes:
                    ax_wrapper.set_ylim(1e-16, 1e1)
                else:
                    ax_wrapper.set_ylim(1e-16, 1e-2)
                ax_wrapper.set_yscale('log', base=10)
                ax_wrapper.minorticks_off()

            ax[0].set_ylabel(r"$||e_{diff}^k||_\infty$", fontsize=20)
            ax[1].set_ylabel(r"$||e_{alg}^k||_\infty$", fontsize=20)
            axRes.set_ylabel('Residual norm', fontsize=20)
            ax[0].legend(frameon=False, fontsize=12, loc='upper left', ncols=2)
            axRes.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

            fig.savefig(f"data/{problems[0].__name__}/{sweeper.__name__}/plotErrorAlongIterations_QI={QI}_M={M}_eps={eps}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            figRes.savefig(f"data/{problems[0].__name__}/{sweeper.__name__}/plotResidualAlongIterations_QI={QI}_M={M}_eps={eps}.png", dpi=300, bbox_inches='tight')
            plt.close(figRes)


if __name__ == "__main__":
    main()
