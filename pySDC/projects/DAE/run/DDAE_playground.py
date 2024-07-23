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


def simulateDAE(use_SE=True):
    r"""
    Main function where things will be done. Here, the problem class ``DiscontinuousTestDAE`` is simulated using
    the ``SemiImplicitDAE`` sweeper, where only the differential variable is integrated using spectral quadrature.
    The problem usually contains a discrete event, but the simulation interval is chosen so that the state function
    does not have a sign change yet (the discrete event does not yet occur). Thus, the solution is smooth.
    """
    # initialize level parameters
    level_params = {
        'restol': -1,
        'dt': 0.01,
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
        'logger_level': 15,
        'hook_class': hook_class,
    }

    # convergence controllers
    convergence_controllers = {}
    switch_estimator_params = {
        'tol': 1e-12,
        'alpha': 1.0,
    }
    if use_SE:
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

    print(f"u at end time {Tend}: {uend}")

    switches = get_sorted(stats, type='switch', sortby='time', recomputed=False)
    if len(switches) >= 1:
        t_switches = [t[1] for t in switches]
        t_switch_found = t_switches[-1]
        event_err = abs(t_switch_exact - t_switch_found)
        event_found = True
    else:
        event_err = None
        event_found = False

    print(f"Event found? {event_found} -- Event time error: {event_err}")

    errDiff = max(np.array(get_sorted(stats, type='e_global_differential_post_step', recomputed=False))[:, 1])
    errAlg = max(np.array(get_sorted(stats, type='e_global_algebraic_post_step', recomputed=False))[:, 1])
    # print(get_sorted(stats, type='e_global_algebraic_post_step', recomputed=False))

    print(f"Differential error: {errDiff}")
    print(f"Algebraic error: {errAlg}")
    print("Hallo")

    u_val = get_sorted(stats, type='u', sortby='time')
    t = np.array([me[0] for me in u_val])
    y = np.array([me[1].diff[0] for me in u_val])
    z = np.array([me[1].alg[0] for me in u_val])

    plt.figure(figsize=(8.5, 8.5))
    plt.plot(t, y, label='Differential variable y')
    plt.plot(t, z, label='Algebraic variable z')
    plt.legend(frameon=False, fontsize=12, loc='upper left')
    plt.xlabel(r"Time $t$")
    plt.ylabel(r"Solution $y(t)$, $z(t)$")

    plt.savefig("data/RK_solution.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    simulateDAE()
