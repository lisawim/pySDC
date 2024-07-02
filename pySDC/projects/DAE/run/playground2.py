import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded
from pySDC.projects.DAE.problems.LinearTestDAE import (
    LinearTestDAEEmbedded,
    LinearTestDAEConstrained,
)
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted


def run():
    # initialize level parameters
    level_params = {
        'restol': 1e-12,
        'dt': 0.1,
    }

    # initialize problem parameters
    problem_params = {
        'newton_tol': 1e-12,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 2,
        'QI': 'IE',
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 2 * sweeper_params['num_nodes'] - 1,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogSolution],
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': LinearTestDAEEmbedded,
        'problem_params': problem_params,
        'sweeper_class': genericImplicitEmbedded,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 0.0
    Tend = 2 * level_params['dt']

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    Path("data").mkdir(parents=True, exist_ok=True)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    plotSolution(stats)


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

    plt.savefig('data/LinearTestSPP/solution.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run()