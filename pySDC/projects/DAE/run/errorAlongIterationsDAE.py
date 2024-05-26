import numpy as np
import matplotlib.pyplot as plt

from projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained, genericImplicitEmbedded
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEConstrained,
    LinearTestDAEEmbedded,
    LinearTestDAEMinionConstrained,
    LinearTestDAEMinionEmbedded,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEEmbedded

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
        LinearTestDAEConstrained,
        LinearTestDAEEmbedded,
    ]
    sweepers = [genericImplicitConstrained, genericImplicitEmbedded]

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'IE'

    # parameters for convergence
    restol = 1e-14
    nSweeps = 60

    # hook class to be used
    hook_class = [
            LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg
        ]

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-14

    t0 = 0.0
    dtValues = [10 ** (-m) for m in range(1, 10)]

    for dt in dtValues:
        fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
        figRes, axRes = plt.subplots(1, 1, figsize=(9.5, 9.5))

        for problem, sweeper in zip(problems, sweepers):
            print(sweeper, dt)
            problem_params = {
                'newton_tol': newton_tol,
            }

            description, controller_params, controller = generateDescription(
                dt=dt,
                problem=problem,
                sweeper=sweeper,
                num_nodes=M,
                quad_type=quad_type,
                QI=QI,
                hook_class=hook_class,
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

            linestyle = 'solid' if sweeper.__name__ == 'genericImplicitConstrained' else 'dashed'
            solid_capstyle = 'round' if linestyle == 'solid' else None
            dash_capstyle = 'round' if linestyle == 'dashed' else None
            label = f'{QI}-constrained' if sweeper.__name__ == 'genericImplicitConstrained' else f'{QI}-embedded'
            ax[0].semilogy(
                np.arange(1, len(errDiffIter) + 1),
                errDiffIter,
                color='blue',
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
            )

            ax[1].semilogy(
                np.arange(1, len(errAlgIter) + 1),
                errAlgIter,
                color='blue',
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
            )

            axRes.semilogy(
                np.arange(1, len(resIter) + 1),
                resIter,
                color='blue',
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
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

        fig.savefig(f"data/LinearTestSPP/plotErrorAlongIterations_QI={QI}_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        figRes.savefig(f"data/LinearTestSPP/plotResidualAlongIterations_QI={QI}_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
        plt.close(figRes)


if __name__ == "__main__":
    main()
