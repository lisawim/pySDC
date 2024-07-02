import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.errors import ParameterError

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP, LinearTestSPPMinion, DiscontinuousTestSPP

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained, genericImplicitEmbedded
from pySDC.projects.DAE.problems.LinearTestDAEMinion import (
    LinearTestDAEMinionEmbedded,
    LinearTestDAEMinionConstrained,
)
from pySDC.projects.DAE.problems.LinearTestDAE import (
    LinearTestDAEEmbedded,
    LinearTestDAEConstrained,
)
from pySDC.projects.DAE.problems.chatGPTDAE import (
    chatGPTDAEEmbedded,
    chatGPTDAEConstrained,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEEmbedded, DiscontinuousTestDAEConstrained

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.implementations.hooks.log_work import LogWork
from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimatePostIter

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        # LinearTestSPPMinion,
        # LinearTestDAEMinionEmbedded,
        # LinearTestDAEMinionConstrained,
        LinearTestSPP,
        LinearTestDAEEmbedded,
        LinearTestDAEConstrained,
        # DiscontinuousTestSPP,
        # DiscontinuousTestDAEEmbedded,
        # DiscontinuousTestDAEConstrained,
    ]
    sweepers = [generic_implicit, genericImplicitEmbedded, genericImplicitConstrained]

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'IE'#'LU'
    conv_type = 'increment'#'increment'

    # parameters for convergence
    nSweeps = 24#14

    # hook class to be used
    hook_class = {
        'generic_implicit': [
            LogWork,
            LogGlobalErrorPostStep,
            LogGlobalErrorPostStepPerturbation,
        ],
        'genericImplicitEmbedded': [
            LogWork,
            LogGlobalErrorPostStepDifferentialVariable,
            LogGlobalErrorPostStepAlgebraicVariable,
        ],
        'genericImplicitConstrained': [
            LogWork,
            LogGlobalErrorPostStepDifferentialVariable,
            LogGlobalErrorPostStepAlgebraicVariable,
        ],
    }

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    lintol = 1e-12

    eps_list = [10 ** (-m) for m in range(2, 12)]#[10 ** (-m) for m in range(1, 11)]
    epsValues = {
        'generic_implicit': eps_list,
        'genericImplicitConstrained': [0.0],
        'genericImplicitEmbedded': [0.0],
    }

    t0 = 0.0
    Tend = 1.0
    nSteps = np.array([2, 5, 10, 20, 50, 100, 200])#, 500, 1000])
    dtValues = (Tend - t0) / nSteps
    # dtValues = np.logspace(-2.5, 0.0, num=10)#np.logspace(-3.0, -1.1, num=16)

    colors = [
        'lightsalmon',
        'lightcoral',
        'indianred',
        'firebrick',
        'brown',
        'maroon',
        'lightgray',
        'darkgray',
        'gray',
        'dimgray',
    ]

    fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
    for problem, sweeper in zip(problems, sweepers):
        epsilons = epsValues[sweeper.__name__]

        for i, eps in enumerate(epsilons):
            costs = []
            errorsDiff, errorsAlg = [], []

            for dt in dtValues:
                print(eps, dt)
                if not eps == 0.0:
                    problem_params = {
                        'lintol': lintol,
                        'eps': eps,
                    }
                else:
                    problem_params = {
                        'lintol': lintol,
                    }

                if not conv_type == 'increment':
                    residual_type = conv_type
                    restol = 1e-11
                    e_tol = -1
                else:
                    residual_type = None
                    restol = -1
                    e_tol = 1e-11

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
                    Tend=Tend,
                    exact_event_time_avail=None,
                )

                gmres = np.array(get_sorted(stats, type='work_gmres', sortby='time'))
                rhs = np.array(get_sorted(stats, type='work_rhs', sortby='time'))

                cost = sum(gmres[:, 1]) + sum(rhs[:, 1])
                costs.append(cost)

                errDiffValues = np.array(get_sorted(stats, type='e_global_post_step', sortby='time'))
                errAlgValues = np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time'))

                errorsDiff.append(max(errDiffValues[:, 1]))
                errorsAlg.append(max(errAlgValues[:, 1]))

                nIter = np.mean(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])
                print(f"Mean number of iterations: {nIter}\n")

            color = colors[i] if not eps == 0.0 else 'k'
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
            ax[0].loglog(
                costs,
                errorsDiff,
                color=color,
                marker=marker,
                markersize=markersize,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
            )

            ax[1].loglog(
                costs,
                errorsAlg,
                color=color,
                marker=marker,
                markersize=markersize,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
            )

    for ind in [0, 1]:
        ax[ind].tick_params(axis='both', which='major', labelsize=14)
        ax[ind].set_xlabel(r'Costs', fontsize=20)
        ax[ind].set_ylim(1e-15, 1e-1)
        ax[ind].minorticks_off()

    ax[0].set_ylabel(r"$||e_{y}||_\infty$", fontsize=20)
    ax[1].set_ylabel(r"$||e_{z}||_\infty$", fontsize=20)
    ax[0].legend(frameon=False, fontsize=12, loc='upper right', ncols=2)

    fig.savefig(f"data/{problems[0].__name__}/plotCosts_{nSweeps=}_{QI=}_{M=}_{conv_type}_{lintol=}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
