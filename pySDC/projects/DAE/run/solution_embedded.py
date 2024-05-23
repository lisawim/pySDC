import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import EmbeddedLinearTestDAE, EmbeddedLinearTestDAEMinion, EmbeddedDiscontinuousTestDAE

from pySDC.projects.DAE.sweepers.genericImplicitEmbedded import genericImplicitEmbedded2
from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAEIntegralFormulation2, LinearTestDAEMinionIntegralFormulation2
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEIntegralFormulation2

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        # EmbeddedLinearTestDAE,
        # LinearTestDAEIntegralFormulation2,
        # EmbeddedDiscontinuousTestDAE,
        # DiscontinuousTestDAEIntegralFormulation2,
        EmbeddedLinearTestDAEMinion,
        LinearTestDAEMinionIntegralFormulation2,
    ]
    sweepers = [generic_implicit, genericImplicitEmbedded2]
    
    # parameters for convergence
    restol = 1e-12
    maxiter = 60

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'

    # hook class to be used
    hook_class = [LogSolution]

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-14

    eps_list = [1.0]
    epsValues = {
        'generic_implicit': eps_list,
        'genericImplicitEmbedded2': [0.0],
    }
    
    t0 = 0.0
    dt = 1e-2
    Tend = np.pi

    colors = [
        'lightcoral',
        'indianred',
        'firebrick',
        'brown',
        'darkred',
        'maroon',
    ]

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 9.5))
    for problem, sweeper in zip(problems, sweepers):
        epsilons = epsValues[sweeper.__name__]

        for i, eps in enumerate(epsilons):
            print(eps)
            problem_params = {
                'newton_tol': newton_tol,
            }
            if not eps == 0.0:
                problem_params = {'eps': eps}

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

            u_val = get_sorted(stats, type='u', sortby='time')
            t = np.array([me[0] for me in u_val])
            if not eps == 0.0:
                u = np.array([me[1] for me in u_val])
            else:
                y = np.array([me[1].diff[0] for me in u_val])
                z = np.array([me[1].alg[0] for me in u_val])
                u = np.concatenate((y, z), axis=1)

            color = colors[i] if not eps == 0.0 else 'k'
            ax.plot(t, u[:, -1], color=color, linewidth=4.0, solid_capstyle='round', label=rf'$\varepsilon=${eps}')

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylabel(r'$z$', fontsize=20)
    ax.legend(frameon=False, fontsize=12, loc='lower right', ncols=2)
    ax.minorticks_off()

    fig.savefig(f"data/{problems[0].__name__}/plotSolutionEmbedded.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
