import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained, genericImplicitEmbedded
from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAEMinionConstrained, LinearTestDAEMinionEmbedded

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
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
    nSweeps = 70

    # hook class to be used
    hook_class = [LogSolution]

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
    # dtValues = np.logspace(-3.0, -1.1, num=16)

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 9.5))
    for problem, sweeper in zip(problems, sweepers):

        for q, QI in enumerate(QIAll):
            algConstr = []

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

                u_val = get_sorted(stats, type='u', sortby='time')
                t = [me[0] for me in u_val]
                u = [me[1] for me in u_val]

                P = controller.MS[0].levels[0].prob
                g = P.algebraicConstraints(u[-1], t[-1])

                algConstr.append(abs(g))

            linestyle = 'solid' if sweeper.__name__ == 'genericImplicitConstrained' else 'dashed'
            solid_capstyle = 'round' if linestyle == 'solid' else None
            dash_capstyle = 'round' if linestyle == 'dashed' else None
            label = f'{QI}-constrained' if sweeper.__name__ == 'genericImplicitConstrained' else f'{QI}-embedded'
            ax.loglog(
                dtValues,
                algConstr,
                color=colors[q],
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
            )

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'$\Delta t$', fontsize=20)
    ax.set_ylim(1e-16, 1e-6)
    ax.minorticks_off()

    ax.set_ylabel(r"$||g(y, z)||_\infty$", fontsize=20)
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/LinearTestSPPMinion/plotAlgConstraints_{nSweeps}sweeps_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main()