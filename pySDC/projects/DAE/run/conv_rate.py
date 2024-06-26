import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP,LinearTestSPPMinion, DiscontinuousTestSPP
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded, genericImplicitConstrained
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEEmbedded,
    LinearTestDAEConstrained,
    LinearTestDAEMinionEmbedded,
    LinearTestDAEMinionConstrained,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEEmbedded, DiscontinuousTestDAEConstrained

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostIter, LogGlobalErrorPostIterPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg

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
        testequation0d,
        # LinearTestSPP,
        # LinearTestDAEEmbedded,
        # LinearTestDAEConstrained,
        # LinearTestSPPMinion,
        # LinearTestDAEMinionEmbedded,
        # LinearTestDAEMinionConstrained,
        # DiscontinuousTestSPP,
        # DiscontinuousTestDAEEmbedded,
        # DiscontinuousTestDAEConstrained,
    ]
    sweepers = [generic_implicit]#, genericImplicitEmbedded, genericImplicitConstrained]

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'

    # parameters for convergence
    nSweeps = 20#7
    residual_type = 'initial_rel'

    # hook class to be used
    hook_class = []

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-12

    epsList = [1e-3, 1e-7, 1e-11]#[10 ** (-m) for m in range(2, 12)]
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
    colors = list(reversed(colors))

    for i, dt in enumerate(dtValues):
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 9.5))

        for problem, sweeper in zip(problems, sweepers):
            epsLoop = epsValues[sweeper.__name__]

            for e, eps in enumerate(epsLoop):
                print(f"{dt=}, {eps=}")
                if not eps == 0.0:
                    problem_params = {
                        # 'eps': eps,
                        # 'newton_tol': newton_tol,
                        'lambdas': np.array([1, 1 / eps]),
                        'u0': 1.0,
                    }
                else:
                    problem_params = {
                        'newton_tol': newton_tol,
                    }

                restol = -1#1e-11#5e-12#5e-6#1e-10#5e-7 if not eps == 0.0 else 1e-10

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

                resIter = [get_sorted(stats, iter=k, type='residual_post_iteration', sortby='time') for k in range(1, nSweeps + 1)]
                resIter = nestedListIntoSingleList(resIter)

                conv_rate = [resIter[k + 1] / resIter[k] for k in range(1, len(resIter) - 1)]

                if eps != 0.0:
                    linestyle = 'solid'
                    label=rf'$\varepsilon=${eps}'
                    markersize=10
                elif eps == 0.0 and sweeper.__name__ == 'genericImplicitEmbedded':
                    linestyle = 'solid'
                    label=rf'$\varepsilon=${eps} -' + r'$\mathtt{SDC-E}$'
                    markersize=10
                else:
                    linestyle = 'dashed'
                    label=rf'$\varepsilon=${eps} - ' + r'$\mathtt{SDC-C}$'
                    markersize=None

                solid_capstyle = 'round' if linestyle == 'solid' else None
                dash_capstyle = 'round' if linestyle == 'dashed' else None

                ax.set_title(rf"Convergence rate for $\Delta t=${dt}")
                ax.plot(
                    np.arange(1, len(conv_rate) + 1),
                    conv_rate,
                    # color=color,
                    marker='o',#marker,
                    markersize=markersize,
                    linewidth=4.0,
                    linestyle=linestyle,
                    solid_capstyle=solid_capstyle,
                    dash_capstyle=dash_capstyle,
                    label=label,
                )

        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel(r'Iteration $k$', fontsize=20)
        ax.set_xlim(0, nSweeps + 1)
        ax.set_ylim(0, 2.5)
        ax.minorticks_off()

        ax.set_ylabel(r"$||r^{k+1}||_\infty / ||r^k||_\infty$", fontsize=20)
        ax.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

        fig.savefig(f"data/{problems[0].__name__}/{i}_plotConvergenceRate_QI={QI}_M={M}_dt={dt}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
