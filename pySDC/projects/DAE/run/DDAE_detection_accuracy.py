import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.RungeKuttaDAE import BackwardEulerDAE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable


def getSolutionDetection(dt, tol_SE, alpha):
    # initialize level parameters
    level_params = {
        'restol': -1,
        'dt': dt,
    }

    # initialize problem parameters
    problem_params = {
        'newton_tol': 1e-14,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 3,
        'QI': 'LU',
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 1,
    }

    # initialize controller parameters
    hook_class = [
        LogSolution, LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable
    ]
    controller_params = {
        'logger_level': 30,
        'hook_class': hook_class,
    }

    # convergence controllers
    convergence_controllers = {}
    switch_estimator_params = {
        'tol': tol_SE,
        'alpha': alpha,
    }
    convergence_controllers.update({SwitchEstimator: switch_estimator_params})
    max_restarts = 200
    restarting_params = {
        'max_restarts': max_restarts,
        'crash_after_max_restarts': False,
    }
    convergence_controllers.update({BasicRestartingNonMPI: restarting_params})

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': DiscontinuousTestDAE,
        'problem_params': problem_params,
        'sweeper_class': BackwardEulerDAE,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 1.0
    Tend = 5.5

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    t_switch_exact = P.t_switch_exact

    Path("data").mkdir(parents=True, exist_ok=True)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats, t_switch_exact


def main():
    filename = "data" + "/" + "detection_accuracy_error.png"
    filename_event = "data" + "/" + "detection_accuracy_event_error.png"

    dt = 0.005
    alphas = np.linspace(0.9, 1.0, num=10)
    tolerances_SE = [10 ** (-m) for m in range(8, 13)]

    fig, ax = plt.subplots(1, 2, figsize=(20, 9))
    fig_event, ax_event = plt.subplots(1, 1, figsize=(9, 9))
    for tol_SE in tolerances_SE:
        errDiffValues, errAlgValues = [], []
        tSwitchValues = []

        for alpha in alphas:
            print(f"... Detection for {alpha=} with tolerance {tol_SE} ...")
            solutionStats, t_switch_exact = getSolutionDetection(dt, tol_SE, alpha)

            switches = get_sorted(solutionStats, type='switch', sortby='time', recomputed=False)
            if len(switches) >= 1:
                event_found = True
                t_switches = [t[1] for t in switches]
                t_switch_found = t_switches[-1]
                eventErr = abs(t_switch_exact - t_switch_found)
            else:
                event_found = False
                eventErr = 1
            print(f"Event found? {event_found}\n")
            errDiff = np.array(get_sorted(solutionStats, type='e_global_differential_post_step', recomputed=False))[:, 1]
            errAlg = np.array(get_sorted(solutionStats, type='e_global_algebraic_post_step', recomputed=False))[:, 1]

            tSwitchValues.append(eventErr)
            errDiffValues.append(max(errDiff))
            errAlgValues.append(max(errAlg))

        ax[0].semilogy(alphas, errDiffValues, linewidth=3.5, solid_capstyle='round', label=rf"$tol_S$={tol_SE}")
        ax[1].semilogy(alphas, errAlgValues, linewidth=3.5, solid_capstyle='round')

        ax_event.semilogy(alphas, tSwitchValues, linewidth=3.5, solid_capstyle='round', label=rf"$tol_S$={tol_SE}")

    ax[0].set_xlabel(r'Parameter $\alpha$', fontsize=22)
    ax[1].set_xlabel(r'Parameter $\alpha$', fontsize=22)
    ax_event.set_xlabel(r'Parameter $\alpha$', fontsize=22)
    ax[0].set_ylabel(r'$L_\infty$ error in $y$', fontsize=22)
    ax[1].set_ylabel(r'$L_\infty$ error in $z$', fontsize=22)
    ax_event.set_ylabel(r'$|t^{ex}_{switch} - t^{SE}_{switch}|$', fontsize=22)
    ax[0].set_ylim(1e-3, 1e2)
    ax[1].set_ylim(1e-3, 1e2)
    ax_event.set_ylim(1e-15, 1e1)
    ax[0].legend(loc='best', frameon=False, fontsize=16)
    ax_event.legend(loc='best', frameon=False, fontsize=16)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)

    fig_event.savefig(filename_event, dpi=400, bbox_inches='tight')
    plt.close(fig_event)


if __name__ == "__main__":
    main()
