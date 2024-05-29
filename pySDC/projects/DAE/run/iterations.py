import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP, LinearTestSPPMinion, DiscontinuousTestSPP

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained, genericImplicitEmbedded
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEEmbedded,
    LinearTestDAEConstrained,
    LinearTestDAEMinionConstrained,
    LinearTestDAEMinionEmbedded,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEEmbedded, DiscontinuousTestDAEConstrained

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun
from pySDC.projects.DAE.run.costs import get_restol

from pySDC.implementations.hooks.log_work import LogWork
from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        LinearTestSPP,
        LinearTestDAEEmbedded,
        LinearTestDAEConstrained,
        # LinearTestSPPMinion,
        # LinearTestDAEMinionEmbedded,
        # LinearTestDAEMinionConstrained,
        # DiscontinuousTestSPP,
        # DiscontinuousTestDAEEmbedded,
        # DiscontinuousTestDAEConstrained,
    ]
    sweepers = [generic_implicit, genericImplicitEmbedded, genericImplicitConstrained]

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    residual_type = 'initial_rel'

    # parameters for convergence
    nSweeps = 6

    # hook class to be used
    hook_class = [LogWork]

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-12

    eps_list = [10 ** (-m) for m in range(1, 11)]
    epsValues = {
        'generic_implicit': eps_list,
        'genericImplicitEmbedded': [0.0],
        'genericImplicitConstrained': [0.0],
    }

    t0 = 0.0
    # nSteps = np.array([2, 5, 10, 20, 50, 100, 200])  # , 500, 1000])
    # dtValues = Tend / nSteps
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

    fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
    for problem, sweeper in zip(problems, sweepers):
        epsilons = epsValues[sweeper.__name__]

        for i, eps in enumerate(epsilons):
            nIters, nItersNewton = [], []

            for dt in dtValues:

                problem_params = {
                    'newton_tol': newton_tol,
                }
                if not eps == 0.0:
                    problem_params = {'eps': eps}
                
                restol = 3e-10 if not eps == 0.0 else 1e-10

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
                )

                stats, _ = controllerRun(
                    description=description,
                    controller_params=controller_params,
                    controller=controller,
                    t0=t0,
                    Tend=t0+dt,
                    exact_event_time_avail=None,
                )

                nIter = np.sum(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])
                nIters.append(nIter)

                nIterNewton = np.sum(np.array(get_sorted(stats, type='work_newton', sortby='time'))[:, 1])
                nItersNewton.append(nIterNewton)

            color = colors[i] if not eps == 0.0 else 'k'
            if eps != 0.0:
                linestyle = 'solid'
                label=rf'$\varepsilon=${eps}'
            elif eps == 0.0 and sweeper.__name__ == 'genericImplicitEmbedded':
                linestyle = 'solid'
                label=rf'$\varepsilon=${eps} -' + r'$\mathtt{SDC-E}$'
            else:
                linestyle = 'dashed'
                label=rf'$\varepsilon=${eps} - ' + r'$\mathtt{SDC-C}$'
            solid_capstyle = 'round' if linestyle == 'solid' else None
            dash_capstyle = 'round' if linestyle == 'dashed' else None
            ax[0].semilogx(
                dtValues,
                nIters,
                color=color,
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
                label=label,
            )
            ax[1].semilogx(
                dtValues,
                nItersNewton,
                color=color,
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle=linestyle,
                solid_capstyle=solid_capstyle,
                dash_capstyle=dash_capstyle,
            )

    for ind in [0, 1]:
        ax[ind].tick_params(axis='both', which='major', labelsize=14)
        ax[ind].set_xlabel(r'$\Delta t$', fontsize=20)
        ax[ind].set_xscale('log', base=10)
        ax[ind].minorticks_off()
    ax[0].set_ylim(1, nSweeps + 5)

    ax[0].set_ylabel(r"Number of SDC iterations", fontsize=20)
    ax[1].set_ylabel(r"Number of Newton iterations", fontsize=20)
    ax[0].legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/{problems[0].__name__}/plotIterationsNewtonAndSDC_{nSweeps}sweeps_QI={QI}_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()