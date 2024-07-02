import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import SPPchatGPT, LinearTestSPP, LinearTestSPPMinion, DiscontinuousTestSPP
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded, genericImplicitConstrained
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

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        # SPPchatGPT,
        # chatGPTDAEEmbedded,
        # testequation0d,
        # LinearTestSPP,
        # LinearTestDAEEmbedded,
        DiscontinuousTestSPP,
        # DiscontinuousTestDAEEmbedded,
        DiscontinuousTestDAEConstrained,
        # LinearTestSPPMinion,
        # LinearTestDAEMinionEmbedded,
    ]
    sweepers = [generic_implicit, genericImplicitConstrained]
    
    # parameters for convergence
    restol = 1e-13
    maxiter = 30#12

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
    newton_tol = 1e-12

    eps_list = [1e-3, 1e-4, 1e-5]#[1e-6]#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # eps_list = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]#[0.01775, 0.0176, 0.016, 0.014, 0.012, 0.01]#[0.003, 0.00093, 0.0009, 0.0005, 0.0001, 1e-5]
    epsValues = {
        'generic_implicit': eps_list,
        'genericImplicitEmbedded': [0.0],
        'genericImplicitConstrained': [0.0],
    }
    
    t0 = 1.0#0.0
    dt = 1e-2
    Tend = 2.0#0.1#1.0#2 * np.pi

    colors = [
        'lightsalmon',
        'lightcoral',
        'indianred',
        'firebrick',
        'brown',
        'darkred',
        'maroon',
    ]

    figAll, axAll = plt.subplots(2, 2, figsize=(17.0, 17.0))
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 9.5))
    for problem, sweeper in zip(problems, sweepers):
        epsilons = epsValues[sweeper.__name__]

        for i, eps in enumerate(epsilons):
            print(eps)
            problem_params = {
                'newton_tol': newton_tol,
            }
            if not eps == 0.0:
                # problem_params = {'eps': eps}
                problem_params = {
                    'eps': eps,
                    # 'u0': 1.0,
                    # 'lambdas': np.array([1, 1 / eps]),
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
                uLast = u
            else:
                y = np.array([me[1].diff[0] for me in u_val])
                z = np.array([me[1].alg[0] for me in u_val])
                u = np.concatenate((y, z), axis=1)
                # u1 = np.array([me[1].diff[0] for me in u_val])
                # u2 = np.array([me[1].diff[1] for me in u_val])
                # u3 = np.array([me[1].diff[2] for me in u_val])
                # u4 = np.array([me[1].alg[0] for me in u_val])
                # uFirst = np.concatenate((u1, u2), axis=1)
                # uSecond = np.concatenate((uFirst, u3), axis=1)
                # uLast = np.concatenate((uSecond, u4), axis=1)

            color = colors[i] if not eps == 0.0 else 'k'
            # ax.plot(t, u[:, 0], color=color, linewidth=4.0, linestyle='dashed', dash_capstyle='round', label=rf'$y$ - $\varepsilon=${eps}')
            ax.plot(t, u[:, -1], color=color, linewidth=4.0, solid_capstyle='round', label=rf'$\varepsilon=${eps}')
            # axAll[0, 0].plot(t, uLast[:, 0], color=color, linewidth=4.0, solid_capstyle='round', label=rf'$\varepsilon=${eps}')
            # axAll[0, 1].plot(t, uLast[:, 1], color=color, linewidth=4.0, solid_capstyle='round')
            # axAll[1, 0].plot(t, uLast[:, 2], color=color, linewidth=4.0, solid_capstyle='round')
            # axAll[1, 1].plot(t, uLast[:, -1], color=color, linewidth=4.0, solid_capstyle='round')

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'$t$', fontsize=20)
    ax.set_ylabel(r'$z$', fontsize=20)
    # ax.set_ylim(-1.5, 1.5)
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)
    ax.minorticks_off()

    # axAll[0, 0].set_ylabel(r"$u_1$", fontsize=20)
    # axAll[0, 1].set_ylabel(r"$u_2$", fontsize=20)
    # axAll[1, 0].set_ylabel(r"$u_3$", fontsize=20)
    # axAll[1, 1].set_ylabel(r"$u_4$", fontsize=20)
    # for i in [0, 1]:
    #     for j in [0, 1]:
    #         if i == 0 and j == 1:
    #             axAll[i, j].set_ylim(-100, 600)
    #         else:
    #             axAll[i, j].set_ylim(-1.5, 1.5)
    #         axAll[i, j].set_xlabel(r'$t$', fontsize=20)
    #         axAll[i, j].minorticks_off()
    # axAll[0, 0].legend(frameon=False, fontsize=12, loc='lower right', ncols=2)

    fig.savefig(f"data/{problems[0].__name__}/plotSolutionEmbedded.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # figAll.savefig(f"data/{problems[0].__name__}/plotSolutionEmbeddedAll.png", dpi=300, bbox_inches='tight')
    # plt.close(figAll)


if __name__ == "__main__":
    main()
