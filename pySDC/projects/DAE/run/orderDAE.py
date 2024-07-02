import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded, genericImplicitConstrained
from pySDC.projects.DAE.problems.LinearTestDAEMinion import LinearTestDAEMinionEmbedded, LinearTestDAEMinionConstrained

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        LinearTestDAEMinionEmbedded,
        LinearTestDAEMinionConstrained,
    ]
    sweepers = [genericImplicitEmbedded, genericImplicitConstrained]

    # sweeper params
    M = 5
    quad_type = 'RADAU-RIGHT'
    QI = 'IE'

    # parameters for convergence
    restol = 2e-12
    maxiter = 80

    # hook class to be used
    hook_class = [
        LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable
    ]

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-14

    t0 = 0.0
    Tend = 1.0
    # nSteps = np.array([2, 5, 10, 20, 50, 100, 200])  # , 500, 1000])
    # dtValues = Tend / nSteps
    dtValues = np.logspace(-2.2, 0.0, num=6)#np.logspace(-2.2, 0.0, num=30)

    fig, ax = plt.subplots(1, 2, figsize=(17.0, 9.5))
    for problem, sweeper in zip(problems, sweepers):
        errorsDiff, errorsAlg = [], []

        for dt in dtValues:
            print(dt)
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

            errDiffValues = np.array(get_sorted(stats, type='e_global_post_step', sortby='time'))
            errAlgValues = np.array(get_sorted(stats, type='e_global_algebraic_post_step', sortby='time'))

            errorsDiff.append(max(errDiffValues[:, 1]))
            errorsAlg.append(max(errAlgValues[:, 1]))

        color = 'k'
        if sweeper.__name__ == 'genericImplicitEmbedded':
            linestyle = 'solid'
            label=r'$\mathtt{SDC-E}$'
        else:
            linestyle = 'dashed'
            label=r'$\mathtt{SDC-C}$'
        solid_capstyle = 'round' if linestyle == 'solid' else None
        dash_capstyle = 'round' if linestyle == 'dashed' else None
        ax[0].loglog(
            dtValues,
            errorsDiff,
            color=color,
            # marker='*',
            # markersize=10.0,
            linewidth=4.0,
            linestyle=linestyle,
            solid_capstyle=solid_capstyle,
            dash_capstyle=dash_capstyle,
            label=label
        )
        ax[1].loglog(
            dtValues,
            errorsAlg,
            color=color,
            # marker='*',
            # markersize=10.0,
            linewidth=4.0,
            linestyle=linestyle,
            solid_capstyle=solid_capstyle,
            dash_capstyle=dash_capstyle,
            label=label
        )

    p = 2 * M - 1
    for ind in [0, 1]:
        fac = 10.0 if ind == 0 else 100.0
        orderReference = [fac * (dt) ** p for dt in dtValues]
        ax[ind].loglog(
            dtValues,
            orderReference,
            color='gray',
            linestyle='--',
            linewidth=4.0,
            dash_capstyle='round',
            label=f'Ref. order $p={p}$',
        )
        ax[ind].tick_params(axis='both', which='major', labelsize=14)
        ax[ind].set_xlabel(r'$\Delta t$', fontsize=20)
        ax[ind].set_ylim(1e-14, 1e1)
        ax[ind].minorticks_off()

    ax[0].set_ylabel(r"$||e_{diff}||_\infty$", fontsize=20)
    ax[1].set_ylabel(r"$||e_{alg}||_\infty$", fontsize=20)
    ax[0].legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    fig.savefig(f"data/{problems[0].__name__}/plotOrderAccuracyDAE_QI={QI}_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    main()
