import dill
from pathlib import Path

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPPMinion

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted


def generateDescription(dt, M, eps):
    r"""
    Generates the description for one run.

    Parameters
    ----------
    dt : float
        Time step size for simulation.
    M : int
        Number of collocation nodes.
    eps : float
        Perturbation parameter for problem class.

    Returns
    -------
    description : dict
        Description containing all the stuff for simulation.
    controller_params : dict
        Controller parameters.
    """

    # initialize level parameters
    level_params = {
        'restol': 1e-11,
        'dt': dt,
    }

    problem_params = {
        'newton_tol': 1e-14,
        'eps': eps,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': 'LU',
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 20,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogSolution],
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': LinearTestSPPMinion,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    return description, controller_params


def computeSolution():
    r"""
    Executes the simulation.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    t0 = 0.0
    Tend = 1.0

    epsValues = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]

    for eps in epsValues:
        if eps == 1e-9 or eps == 1e-10:
            dt = 1e-5
            M = 6
        elif eps == 1e-8:
            dt = 1e-6
            M = 6
        elif eps == 1e-7:
            dt = 5e-6
            M = 3
        elif eps == 1e-6:
            dt = 1e-8
            M = 3
        elif eps == 1e-5:
            dt = 1e-7
            M = 3
        else:
            raise NotImplementedError(f"No time step size implemented for eps={eps}!")

        description, controller_params = generateDescription(dt, M, eps)

        # instantiate controller
        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=dt)  # change Tend=Tend!!!

        u_val = get_sorted(stats, type='u', sortby='time')

        fname = 'data/{}_M={}_dt={}_eps={}.dat'.format(description['problem_class'].__name__, M, dt, eps)
        f = open(fname, 'wb')
        dill.dump(u_val, f)
        f.close()


if __name__ == "__main__":
    computeSolution()
