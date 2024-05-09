import numpy as np
from pathlib import Path
import cProfile
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.RungeKuttaDAE import BackwardEulerDAE, TrapezoidalRuleDAE
from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAE
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.hooks.log_solution import LogSolution


def run():
    r"""
    Run the simulation.
    """
    # initialize level parameters
    dt = np.logspace(-2.0, 0.0, num=20)[-1]
    level_params = {
        'restol': -1,
        'dt': dt,
    }

    # initialize problem parameters
    problem_params = {
        'newton_tol': 1e-15,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 3,
        'QI': 'IE',
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 1,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogSolution],
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': LinearTestDAE,
        'problem_params': problem_params,
        'sweeper_class': TrapezoidalRuleDAE,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 0.0
    Tend = 4.0#dt

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uex = P.u_exact(Tend)

    Path("data").mkdir(parents=True, exist_ok=True)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    print(f"Numerical diff error: {abs(uex.diff - uend.diff)}")
    print(f"Numerical alg error: {abs(uex.alg - uend.alg)}")

    # plotSolution(stats)


def plotSolution(stats):
    r"""
    Here, the solution of the DAE is plotted.

    Parameters
    ----------
    stats : dict
        Contains all the statistics of the simulation.
    """

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

    plt.savefig('data/solution_playgroundRK.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run()