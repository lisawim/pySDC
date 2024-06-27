import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import SPPchatGPT, LinearTestSPP, LinearTestSPPMinion, DiscontinuousTestSPP
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded, genericImplicitConstrained
from pySDC.projects.DAE.problems.TestDAEs import (
    chatGPTDAEEmbedded,
    chatGPTDAEConstrained,
    LinearTestDAEEmbedded,
    LinearTestDAEConstrained,
    LinearTestDAEMinionEmbedded,
    LinearTestDAEMinionConstrained,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEEmbedded, DiscontinuousTestDAEConstrained

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostIter, LogGlobalErrorPostIterPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimatePostIter

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
        # testequation0d,
        # SPPchatGPT,
        # chatGPTDAEEmbedded,
        # chatGPTDAEConstrained,
        # LinearTestSPP,
        # LinearTestDAEEmbedded,
        # LinearTestDAEConstrained,
        LinearTestSPPMinion,
        LinearTestDAEMinionEmbedded,
        LinearTestDAEMinionConstrained,
        # DiscontinuousTestSPP,
        # DiscontinuousTestDAEEmbedded,
        # DiscontinuousTestDAEConstrained,
    ]
    sweepers = [generic_implicit, genericImplicitEmbedded, genericImplicitConstrained]

    # sweeper params
    M = 6
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'

    # parameters for convergence
    nSweeps = 25#7
    conv_type = 'initial_rel'#'increment'

    # hook class to be used
    hook_class = {
        'generic_implicit': [LogGlobalErrorPostIter, LogGlobalErrorPostIterPerturbation, LogEmbeddedErrorEstimatePostIter],
        'genericImplicitEmbedded': [
            LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg, LogEmbeddedErrorEstimatePostIter
        ],
        'genericImplicitConstrained': [
            LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg, LogEmbeddedErrorEstimatePostIter
        ],
    }

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-12

    epsList = [10 ** (-m) for m in range(5, 12)]
    epsValues = {
        'generic_implicit': epsList,
        'genericImplicitEmbedded': [0.0],
        'genericImplicitConstrained': [0.0],
    }

    t0 = 0.0
    Tend = 1.0
    nSteps = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    dtValues = (Tend - t0) / nSteps
    # dtValues = np.logspace(-2.5, 0.0, num=7)

    colors = [
        # 'lightsalmon',
        # 'lightcoral',
        # 'indianred',
        'firebrick',
        'brown',
        'maroon',
        'lightgray',
        'darkgray',
        'gray',
        'dimgray',
    ]
    # colors = list(reversed(colors))

    for i, dt in enumerate(dtValues):
        fig, ax = plt.subplots(1, 3, figsize=(24.0, 9.0))

        for problem, sweeper in zip(problems, sweepers):
            epsLoop = epsValues[sweeper.__name__]

            for e, eps in enumerate(epsLoop):
                print(f"{dt=}, {eps=}")
                if not eps == 0.0:
                    problem_params = {
                        'eps': eps,
                        'newton_tol': newton_tol,
                    }
                else:
                    problem_params = {
                        'newton_tol': newton_tol,
                    }

                if not conv_type == 'increment':
                    residual_type = conv_type
                    restol = 1e-13
                    e_tol = -1
                else:
                    residual_type = None
                    restol = -1
                    e_tol = -1

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
                    residual_type=residual_type,
                    e_tol=e_tol,
                )

                stats, _ = controllerRun(
                    description=description,
                    controller_params=controller_params,
                    controller=controller,
                    t0=t0,
                    Tend=t0+dt,
                    exact_event_time_avail=None,
                )

                errDiffIter = [get_sorted(stats, iter=k, type='e_global_post_iter', sortby='time') for k in range(1, nSweeps + 1)]
                errAlgIter = [get_sorted(stats, iter=k, type='e_global_algebraic_post_iter', sortby='time') for k in range(1, nSweeps + 1)]

                if not conv_type == 'increment':
                    quantityIter = [get_sorted(stats, iter=k, type='residual_post_iteration', sortby='time') for k in range(1, nSweeps + 1)]
                else:
                    type = 'error_embedded_estimate_post_iteration'
                    quantityIter = [get_sorted(stats, iter=k, type=type, sortby='time') for k in range(1, nSweeps + 1)]

                errDiffIter = nestedListIntoSingleList(errDiffIter)
                errAlgIter = nestedListIntoSingleList(errAlgIter)
                quantityIter = nestedListIntoSingleList(quantityIter)

                color = colors[e] if not eps == 0.0 else 'k'
                if eps != 0.0:
                    linestyle = 'solid'
                    label=rf'$\varepsilon=${eps}'
                    marker=None
                    markersize=None
                elif eps == 0.0 and sweeper.__name__ == 'genericImplicitEmbedded':
                    linestyle = 'solid'
                    label=rf'$\varepsilon=${eps} -' + r'$\mathtt{SDC-E}$'
                    marker=None
                    markersize=None
                else:
                    linestyle = 'dashed'
                    label=rf'$\varepsilon=${eps} - ' + r'$\mathtt{SDC-C}$'
                    marker='*'
                    markersize=10.0
                solid_capstyle = 'round' if linestyle == 'solid' else None
                dash_capstyle = 'round' if linestyle == 'dashed' else None

                ax[0].set_title(rf"Differential error for $\Delta t=${dt}")
                ax[0].semilogy(
                    np.arange(1, len(errDiffIter) + 1),
                    errDiffIter,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    linewidth=4.0,
                    linestyle=linestyle,
                    solid_capstyle=solid_capstyle,
                    dash_capstyle=dash_capstyle,
                    label=label,
                )

                ax[1].set_title(rf"Algebraic error for $\Delta t=${dt}")
                ax[1].semilogy(
                    np.arange(1, len(errAlgIter) + 1),
                    errAlgIter,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    linewidth=4.0,
                    linestyle=linestyle,
                    solid_capstyle=solid_capstyle,
                    dash_capstyle=dash_capstyle,
                    label=label,
                )
                print(eps, quantityIter)
                if conv_type == 'increment':
                    ax[2].set_title(rf"Increment for $\Delta t=${dt}")
                else:
                    ax[2].set_title(rf"Residual for $\Delta t=${dt}")
                ax[2].semilogy(
                    np.arange(1, len(quantityIter) + 1),
                    quantityIter,
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    linewidth=4.0,
                    linestyle=linestyle,
                    solid_capstyle=solid_capstyle,
                    dash_capstyle=dash_capstyle,
                    label=label,
                )

        for ax_wrapper in [ax[0], ax[1], ax[2]]:
            ax_wrapper.tick_params(axis='both', which='major', labelsize=14)
            ax_wrapper.set_xlabel(r'Iteration $k$', fontsize=20)
            ax_wrapper.set_xlim(0, nSweeps + 1)
            # ax_wrapper.set_xticks(np.arange(1, nSweeps + 1))
            # ax_wrapper.set_xticklabels([i for i in np.arange(1, nSweeps + 1)])
            if not ax_wrapper == ax[2]:
                ax_wrapper.set_ylim(1e-15, 1e0)
                # ax_wrapper.set_ylim(1e-5, 1e0)
            else:
                ax_wrapper.set_ylim(1e-15, 1e5)
            ax_wrapper.set_yscale('log', base=10)
            ax_wrapper.minorticks_off()

        ax[0].set_ylabel(r"$||e_{y}^k||_\infty$", fontsize=20)
        ax[1].set_ylabel(r"$||e_{z}^k||_\infty$", fontsize=20)
        if conv_type == 'initial_rel':
            ax[2].set_ylabel(r"$||r^k||_\infty/||r^0||_\infty$", fontsize=20)
        elif conv_type == 'increment':
            ax[2].set_ylabel(r"$||u^{k+1} - u^k||_\infty$", fontsize=20)
        else:
            raise NotImplementedError()
        ax[0].legend(frameon=False, fontsize=12, loc='upper right', ncols=2)

        fig.savefig(f"data/{problems[0].__name__}/{i}_plotQuantitiesAlongIterations_QI={QI}_M={M}_dt={dt}_{conv_type}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
