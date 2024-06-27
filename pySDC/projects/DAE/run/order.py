import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import (
    LinearTestSPP,
    LinearTestSPPMinion,
    DiscontinuousTestSPP,
)

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEEmbedded,
    LinearTestDAEMinionEmbedded,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEEmbedded

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        LinearTestSPP,
        # LinearTestDAEEmbedded,
        # DiscontinuousTestSPP,
        # DiscontinuousTestDAEEmbedded
    ]
    sweepers = [generic_implicit, genericImplicitEmbedded]

    if sweepers[1].__name__.startswith("genericImplicit"):
        startsWith = True

    assert startsWith, "To store files correctly, set one of the DAE sweeper into list at index 1!"

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    conv_type = 'increment'

    # parameters for convergence
    maxiter = 6

    # hook class to be used
    hook_class = {
        'generic_implicit': [LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation],
        'genericImplicitEmbedded': [
            LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable
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

    eps_list = [10 ** (-m) for m in range(2, 12)]
    epsValues = {
        'generic_implicit': eps_list,
        'genericImplicitEmbedded': [0.0],
    }

    t0 = 0.0
    Tend = 1.0
    # nSteps = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    # dtValues = (Tend - t0) / nSteps
    dtValues = np.logspace(-2.5, 0.0, num=40)

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
    colors = list(reversed(colors))

    fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
    figIter, axIter = plt.subplots(1, 1, figsize=(9.5, 9.5))
    for problem, sweeper in zip(problems, sweepers):
        epsilons = epsValues[sweeper.__name__]

        for i, eps in enumerate(epsilons):
            errorsDiff, errorsAlg = [], []
            nIters = []

            for dt in dtValues:
                print(eps, dt)
                problem_params = {
                    'newton_tol': newton_tol,
                }
                if not eps == 0.0:
                    problem_params = {
                        'eps': eps,
                        'newton_tol': newton_tol,
                    }

                if not conv_type == 'increment':
                    residual_type = conv_type
                    restol = 1e-13
                    e_tol = -1
                else:
                    residual_type = None
                    restol = -1
                    e_tol = 1e-12

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
                    maxiter=maxiter,
                    max_restarts=max_restarts,
                    tol_event=tol_event,
                    alpha=alpha,
                    residual_type=residual_type,
                    e_tol=e_tol,
                )

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
                    maxiter=maxiter,
                    max_restarts=max_restarts,
                    tol_event=tol_event,
                    alpha=alpha,
                    residual_type=residual_type,
                )

                stats, _ = controllerRun(
                    description=description,
                    controller_params=controller_params,
                    controller=controller,
                    t0=t0,
                    Tend=t0+dt,#Tend,
                    exact_event_time_avail=None,
                )

                errDiffValues = np.array(get_sorted(stats, type='e_global_post_step', sortby='time'))
                errAlgValues = np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time'))
                nIter = np.mean(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])

                errorsDiff.append(max(errDiffValues[:, 1]))
                errorsAlg.append(max(errAlgValues[:, 1]))
                nIters.append(nIter)

            color = colors[i] if not eps == 0.0 else 'k'
            ax[0].loglog(
                dtValues,
                errorsDiff,
                color=color,
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                solid_capstyle='round',
                label=rf'$\varepsilon=${eps}',
            )
            ax[1].loglog(
                dtValues,
                errorsAlg,
                color=color,
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                solid_capstyle='round',
            )
            axIter.semilogx(
                dtValues,
                nIters,
                color=color,
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                solid_capstyle='round',
                label=rf'$\varepsilon=${eps}',
            )

    p = 2 * M - 1
    for ind in [0, 1]:
        fac = 10.0 if ind == 0 else 100.0
        orderReference = [fac * (dt) ** p for dt in dtValues]
        ax[ind].loglog(
            dtValues,
            orderReference,
            color='gray',
            linestyle='--',
            linewidth=4.0,
            dash_capstyle='round',
            label=f'Ref. order $p={p}$',
        )
        ax[ind].tick_params(axis='both', which='major', labelsize=14)
        ax[ind].set_xlabel(r'$\Delta t$', fontsize=20)
        ax[ind].set_ylim(1e-14, 1e1)
        ax[ind].minorticks_off()

    ax[0].set_ylabel(r"$||e_{y}||_\infty$", fontsize=20)
    ax[1].set_ylabel(r"$||e_{z}||_\infty$", fontsize=20)
    ax[0].legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    axIter.tick_params(axis='both', which='major', labelsize=14)
    axIter.set_xlabel(r'$\Delta t$', fontsize=20)
    axIter.set_xscale('log', base=10)
    axIter.minorticks_off()
    axIter.set_ylim(1, maxiter + 5)

    axIter.set_ylabel(r"(Mean) number of SDC iterations", fontsize=20)
    axIter.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/{problems[0].__name__}/plotOrderAccuracy_QI={QI}_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    figIter.savefig(f"data/{problems[0].__name__}/plotIterations_QI={QI}_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(figIter)


if __name__ == "__main__":
    main()
