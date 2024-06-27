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

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        # testequation0d,
        # SPPchatGPT,
        # chatGPTDAEEmbedded,
        # chatGPTDAEConstrained,
        # LinearTestSPP,
        LinearTestDAEEmbedded,
        LinearTestDAEConstrained,
        # LinearTestSPPMinion,
        # LinearTestDAEMinionEmbedded,
        # LinearTestDAEMinionConstrained,
        # DiscontinuousTestSPP,
        # DiscontinuousTestDAEEmbedded,
        # DiscontinuousTestDAEConstrained,
    ]
    # sweepers = [generic_implicit]
    sweepers = [genericImplicitEmbedded, genericImplicitConstrained]

    # sweeper params
    M = 6
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'

    # parameters for convergence
    nSweeps = 25#7
    conv_type = 'initial_rel'#'increment'

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

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 9.5))
    for problem, sweeper in zip(problems, sweepers):
        epsilons = epsValues[sweeper.__name__]

        for i, eps in enumerate(epsilons):
            algConstr = []

            for dt in dtValues:
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
                    hook_class=hook_class,
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

                u_val = get_sorted(stats, type='u', sortby='time')
                t = [me[0] for me in u_val]
                u = [me[1] for me in u_val]

                P = controller.MS[0].levels[0].prob
                g = P.algebraicConstraints(u[-1], t[-1])

                algConstr.append(abs(g))

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
            print(algConstr)
            ax.loglog(
                dtValues,
                algConstr,
                color=color,
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                solid_capstyle='round',
                label=rf'$\varepsilon=${eps}',
            )

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'$\Delta t$', fontsize=20)
    ax.set_ylim(1e-14, 1e1)
    ax.minorticks_off()

    ax.set_ylabel(r"$||g(y, z)||_\infty$", fontsize=20)
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/{problems[0].__name__}/plotSolutionManifold_QI={QI}_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
