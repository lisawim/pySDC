import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP, LinearTestSPPMinion

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded
from pySDC.projects.DAE.problems.LinearTestDAEMinion import LinearTestDAEMinionEmbedded
from pySDC.projects.DAE.problems.LinearTestDAE import LinearTestDAEEmbedded

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.implementations.hooks.log_work import LogWork
from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        LinearTestSPP,
        LinearTestDAEEmbedded,
    ]
    sweepers = [generic_implicit, genericImplicitEmbedded]

    if sweepers[1].__name__.startswith("genericImplicit"):
        startsWith = True

    assert startsWith, "To store files correctly, set one of the DAE sweeper into list at index 1!"
    
    # parameters for convergence
    maxiter = 40

    # sweeper params
    M = 4
    quad_type = 'RADAU-RIGHT'
    QIAll = ['MIN-SR-FLEX']#['IE', 'LU', 'MIN-SR-S', 'MIN-SR-FLEX']

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
    }

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-12

    epsValues = {
        'generic_implicit': [1e-3],
        'genericImplicitEmbedded': [0.0],
    }

    t0 = 0.0
    Tend = 1.0
    nSteps = np.array([2])#np.array([2, 5, 10, 20, 50, 100, 200])  # , 500, 1000])
    dtValues = Tend / nSteps
    # dtValues = np.logspace(-2.5, -0.35, num=10)

    colors = [
        'salmon',
        'palegreen',
        'turquoise',
        'deepskyblue',
        'violet',
    ]

    figDiff, axDiff = plt.subplots(1, 1, figsize=(9.5, 9.5))
    figAlg, axAlg = plt.subplots(1, 1, figsize=(9.5, 9.5))
    for problem, sweeper in zip(problems, sweepers):
        epsilons = epsValues[sweeper.__name__]

        for q, QI in enumerate(QIAll):

            for i, eps in enumerate(epsilons):
                costs = []
                errorsDiff, errorsAlg = [], []

                for dt in dtValues:
                    print(QI, eps, dt)
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
                        maxiter=maxiter,
                        max_restarts=max_restarts,
                        tol_event=tol_event,
                        alpha=alpha,
                    )

                    stats, _ = controllerRun(
                        description=description,
                        controller_params=controller_params,
                        controller=controller,
                        t0=t0,
                        Tend=Tend,
                        exact_event_time_avail=None,
                    )

                    newton = np.array(get_sorted(stats, type='work_newton', sortby='time'))
                    rhs = np.array(get_sorted(stats, type='work_rhs', sortby='time'))

                    cost = sum(newton[:, 1]) + sum(rhs[:, 1])
                    costs.append(cost)

                    errDiffValues = np.array(get_sorted(stats, type='e_global_post_step', sortby='time'))
                    errAlgValues = np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time'))

                    errorsDiff.append(max(errDiffValues[:, 1]))
                    errorsAlg.append(max(errAlgValues[:, 1]))

            linestyle = 'solid' if eps == 0.0 else 'dashed'
            solid_capstyle = 'round' if linestyle == 'solid' else None
            dash_capstyle = 'round' if linestyle == 'dashed' else None
            axDiff.loglog(
                costs,
                errorsDiff,
                color=colors[q],
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=rf'{QI} - $\varepsilon=${eps}',
            )
            axAlg.loglog(
                costs,
                errorsAlg,
                color=colors[q],
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=rf'{QI} - $\varepsilon=${eps}',
            )

    for ax_wrapper in [axDiff, axAlg]:
        ax_wrapper.tick_params(axis='both', which='major', labelsize=14)
        ax_wrapper.set_xlabel(r'Costs', fontsize=20)
        # ax_wrapper.set_xlim(1e2, 5e4)
        ax_wrapper.set_ylim(1e-14, 1e1)
        ax_wrapper.minorticks_off()
        if ax_wrapper == axDiff:
            ax_wrapper.set_ylabel(r"$||e_{diff}||_\infty$", fontsize=20)
        else:
            ax_wrapper.set_ylabel(r"$||e_{alg}||_\infty$", fontsize=20)

        ax_wrapper.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    figDiff.savefig(f"data/{problems[0].__name__}/{sweepers[1].__name__}/plotCostsDiffQI_allM={M}_eps={epsValues['generic_implicit'][0]}.png", dpi=300, bbox_inches='tight')
    plt.close(figDiff)

    figAlg.savefig(f"data/{problems[0].__name__}/{sweepers[1].__name__}/plotCostsAlgQI_all_M={M}_eps={epsValues['generic_implicit'][0]}.png", dpi=300, bbox_inches='tight')
    plt.close(figAlg)


if __name__ == "__main__":
    main()
