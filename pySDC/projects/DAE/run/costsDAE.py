import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained, genericImplicitEmbedded
from pySDC.projects.DAE.problems.LinearTestDAEMinion import LinearTestDAEMinionConstrained, LinearTestDAEMinionEmbedded

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.implementations.hooks.log_work import LogWork
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        # LinearTestDAEEmbedded,
        # LinearTestDAEConstrained,
        LinearTestDAEMinionEmbedded,
        LinearTestDAEMinionConstrained,
    ]
    sweepers = [genericImplicitEmbedded, genericImplicitConstrained]

    # sweeper params
    M = 4
    quad_type = 'RADAU-RIGHT'
    QIAll = ['IE', 'LU', 'MIN-SR-S']

    # parameters for convergence
    restol = 1e-13
    nSweeps = 2*M - 1#70

    # hook class to be used
    hook_class = [
        LogWork,
        LogGlobalErrorPostStepDifferentialVariable,
        LogGlobalErrorPostStepAlgebraicVariable,
    ]
    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-12

    colors = [
        'salmon',
        'palegreen',
        'turquoise',
        'deepskyblue',
        'violet',
    ]

    t0 = 0.0
    Tend = 2.0
    nSteps = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    dtValues = Tend / nSteps

    fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
    for problem, sweeper in zip(problems, sweepers):

        for q, QI in enumerate(QIAll):
            costs = []
            errorsDiff, errorsAlg = [], []

            for dt in dtValues:
                print(QI, dt)
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

            linestyle = 'solid' if sweeper.__name__ == 'genericImplicitConstrained' else 'dashed'
            solid_capstyle = 'round' if linestyle == 'solid' else None
            dash_capstyle = 'round' if linestyle == 'dashed' else None
            label = f'{QI}-constrained' if sweeper.__name__ == 'genericImplicitConstrained' else f'{QI}-embedded'
            ax[0].loglog(
                costs,
                errorsDiff,
                color=colors[q],
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
            )

            ax[1].loglog(
                costs,
                errorsAlg,
                color=colors[q],
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
            )

    for ind in [0, 1]:
        ax[ind].tick_params(axis='both', which='major', labelsize=14)
        ax[ind].set_xlabel(r'Costs', fontsize=20)
        ax[ind].set_ylim(1e-15, 1e2)
        ax[ind].minorticks_off()

    ax[0].set_ylabel(r"$||e_{diff}||_\infty$", fontsize=20)
    ax[1].set_ylabel(r"$||e_{alg}||_\infty$", fontsize=20)
    ax[0].legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/LinearTestSPPMinion/plotCostsSDCForDAEs_{nSweeps}sweeps_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()
