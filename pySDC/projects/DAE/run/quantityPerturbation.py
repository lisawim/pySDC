import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate

from pySDC.helpers.stats_helper import get_sorted


def main():
    problem = LinearTestSPP
    sweeper = generic_implicit

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    conv_type = 'increment'

    # parameters for convergence
    maxiter = 14

    # hook class to be used
    hook_class = [LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation, LogEmbeddedErrorEstimate]

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-12

    t0 = 0.0
    Tend = 1.0

    epsValues = np.logspace(-11, 1, num=40)#[10 ** (-m) for m in range(2, 12)]
    # dtValues = [5 * 10 ** (-m) for m in range(1, 5)]
    t0 = 0.0
    Tend = 1.0
    nSteps = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    dtValues = (Tend - t0) / nSteps

    errDiff, errAlg = np.zeros((len(dtValues), len(epsValues))), np.zeros((len(dtValues), len(epsValues)))
    res, embedErr = np.zeros((len(dtValues), len(epsValues))), np.zeros((len(dtValues), len(epsValues)))
    for d, dt in enumerate(dtValues):
        for e, eps in enumerate(epsValues):
            print(f"{dt=}, {eps=}")
            problem_params = {
                'lintol': 1e-11,
                'eps': eps,
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
                hook_class=hook_class,
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
            resValues = np.array(get_sorted(stats, type='residual_post_step', sortby='time'))
            if not conv_type == 'increment':
                type = 'error_embedded_estimate_post_step'
                embedErrValues = np.array(get_sorted(stats, type=type, sortby='time'))
                embedErr[d, e] = max(embedErrValues[:, 1])

            errDiff[d, e] = max(errDiffValues[:, 1])
            errAlg[d, e] = max(errAlgValues[:, 1])
            res[d, e] = max(resValues[:, 1])

    plotErrorPerturbation(
        dtValues,
        epsValues,
        errDiff,
        errAlg,
        epsValues,
        (1e-15, 1e0),
        problem.__name__,
        f'plotErrorsPerturbation_{QI=}_{M=}.png',
    )

    plotResidualAndEmbeddedErrorPerturbation(
        dtValues,
        epsValues,
        res,
        None,
        (1e-15, 1e0),
        r"$||r||_\infty$",
        problem.__name__,
        f'plotResidualPerturbation_{QI=}_{M=}.png',
    )

    plotResidualAndEmbeddedErrorPerturbation(
        dtValues,
        epsValues,
        embedErr,
        None,
        (1e-15, 1e0),
        'Embedded error',
        problem.__name__,
        f'plotEmbeddedErrorPerturbation_{QI=}_{M=}.png',
    )


def plotErrorPerturbation(dtValues, epsValues, errDiff, errAlg, xLim, yLim, prob_cls_name, file_name):
    fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
    for d, dt in enumerate(dtValues):
        ax[0].loglog(epsValues, errDiff[d, :], label=rf"$\Delta t=${dt}")
        ax[1].loglog(epsValues, errAlg[d, :])

    for ax_wrapper in [ax[0], ax[1]]:
        ax_wrapper.tick_params(axis='both', which='major', labelsize=14)
        ax_wrapper.set_xlabel(r'Parameter $\varepsilon$', fontsize=20)
        ax_wrapper.set_yscale('log', base=10)
        # ax.xscale('log', base=10)
        # if xLim is not None:
        #     ax.set_xlim(xLim[0], xLim[-1])

        #     powersOfTen = generate_powers_of_ten(xLim)
        #     ax.set_xticks(powersOfTen)
        #     ax.set_xticklabels([i for i in powersOfTen])

        if yLim is not None:
            ax_wrapper.set_ylim(yLim[0], yLim[-1])

        ax_wrapper.minorticks_off()


    ax[0].set_ylabel(r"$||e_{y}^k||_\infty$", fontsize=20)
    ax[1].set_ylabel(r"$||e_{z}^k||_\infty$", fontsize=20)
    ax[0].legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/{prob_cls_name}/{file_name}", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotResidualAndEmbeddedErrorPerturbation(dtValues, epsValues, quantity, xLim, yLim, yLabel, prob_cls_name, file_name):
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 9.5))
    for d, dt in enumerate(dtValues):
        ax.loglog(epsValues, quantity[d, :], label=rf"$\Delta t=${dt}")

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=20)
    ax.set_yscale('log', base=10)
    # ax.xscale('log', base=10)
    # if xLim is not None:
    #     ax.set_xlim(xLim[0], xLim[-1])

    #     powersOfTen = generate_powers_of_ten(xLim)
    #     ax.set_xticks(powersOfTen)
    #     ax.set_xticklabels([i for i in powersOfTen])

    if yLim is not None:
        ax.set_ylim(yLim[0], yLim[-1])

    ax.minorticks_off()


    ax.set_ylabel(yLabel, fontsize=20)
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/{prob_cls_name}/{file_name}", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
