import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP,LinearTestSPPMinion, DiscontinuousTestSPP

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded, genericImplicitConstrained
from pySDC.projects.DAE.problems.TestDAEs import (
    LinearTestDAEEmbedded,
    LinearTestDAEConstrained,
    LinearTestDAEMinionEmbedded,
    LinearTestDAEMinionConstrained,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEEmbedded, DiscontinuousTestDAEConstrained

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg

from pySDC.helpers.stats_helper import get_sorted


def main():
    problem = LinearTestSPP
    sweeper = generic_implicit

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'

    # parameters for convergence
    nSweeps = 20
    residual_type = 'initial_rel'

    # hook class to be used
    hook_class = [LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation]

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tolerances = [1e-12]#[10 ** (-m) for m in range(7, 13)]

    # epsValues = [1e-11, 1e-10, 1e-9, 
    epsValues = [1e-6]

    t0 = 0.0
    Tend = 1.0
    nSteps = np.array([100, 200])#np.array([2, 5, 10, 20, 50, 100, 200])#, 500, 1000])
    dtValues = (Tend - t0) / nSteps

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

    for eps in epsValues:
        for i, dt in enumerate(dtValues):
            fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
            figRes, axRes = plt.subplots(1, 1, figsize=(9.5, 9.5))

            for n, newton_tol in enumerate(newton_tolerances):
                print(eps, dt, newton_tol)
                if not eps == 0.0:
                    problem_params = {
                        'newton_tol': newton_tol,
                        'eps': eps
                    }

                else:
                    problem_params = {
                        'newton_tol': 1e-12,
                    }

                restol = 1e-13#1e-11

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
                    Tend=4*dt,#Tend,
                    exact_event_time_avail=None,
                )

                errDiff = np.array(get_sorted(stats, type='e_global_post_step', sortby='time'))
                errAlg = np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time'))

                res = np.array(get_sorted(stats, type='residual_post_step', sortby='time'))

                meanNiters = np.mean(np.array(get_sorted(stats, type='niter', recomputed=None, sortby='time'))[:, 1])

                ax[0].set_title(rf"Differential error for $\Delta t=${dt} for $\varepsilon=${eps} - mean number of iter.: {meanNiters}")
                ax[0].semilogy(
                    errDiff[:, 0],
                    errDiff[:, 1],
                    color=colors[n],
                    linewidth=4.0,
                    linestyle='solid',
                    solid_capstyle='round',
                    label=f"tol={newton_tol}",
                )

                ax[1].set_title(rf"Algebraic error for $\Delta t=${dt} for $\varepsilon=${eps} - mean number of iter.: {meanNiters}")
                ax[1].semilogy(
                    errAlg[:, 0],
                    errAlg[:, 1],
                    color=colors[n],
                    linewidth=4.0,
                    linestyle='solid',
                    solid_capstyle='round',
                )

                axRes.set_title(rf"Residual for $\Delta t=${dt} for $\varepsilon=${eps} - mean number of iter.: {meanNiters}")
                axRes.semilogy(
                    res[:, 0],
                    res[:, 1],
                    color=colors[n],
                    linewidth=4.0,
                    linestyle='solid',
                    solid_capstyle='round',
                    label=f"tol={newton_tol}",
                )

            for ax_wrapper in [ax[0], ax[1], axRes]:
                ax_wrapper.tick_params(axis='both', which='major', labelsize=14)
                ax_wrapper.set_xlabel(r'Time $t$', fontsize=20)
                if not ax_wrapper == axRes:
                    ax_wrapper.set_ylim(1e-16, 1e1)
                else:
                    ax_wrapper.set_ylim(1e-13, 1e-1)
                ax_wrapper.set_yscale('log', base=10)
                ax_wrapper.minorticks_off()

            ax[0].set_ylabel(r"$e_{diff}(t)$", fontsize=20)
            ax[1].set_ylabel(r"$e_{alg}(t)$", fontsize=20)
            axRes.set_ylabel(r"r(t)", fontsize=20)
            ax[0].legend(frameon=False, fontsize=12, loc='upper right', ncols=2)
            axRes.legend(frameon=False, fontsize=12, loc='upper right', ncols=2)

            fig.savefig(f"data/{problem.__name__}/{i}_plotErrorOverTime_{QI=}_{M=}_{dt=}_{restol=}_{eps=}_{nSweeps=}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            figRes.savefig(f"data/{problem.__name__}/{i}_plotResidualOverTime_{QI=}_{M=}_{dt=}_{restol=}_{eps=}_{nSweeps=}.png", dpi=300, bbox_inches='tight')
            plt.close(figRes)


if __name__ == "__main__":
    main()
