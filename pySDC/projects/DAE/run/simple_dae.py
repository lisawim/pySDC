from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
from pySDC.projects.DAE.sweepers.implicit_Euler_DAE import implicit_Euler_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.helpers.stats_helper import get_sorted


def get_description(dt, nvars, problem_class, newton_tol, sweeper=implicit_Euler_DAE, quad_type='LOBATTO', num_nodes=2):
    """
    Returns the description for one simulation run.
    Args:
        dt (float): time step size for computation
        nvars (int): number of variables of the problem
        problem_class (pySDC.core.Problem.ptype_dae): problem class that wants to be simulated
        newton_tol (float): Tolerance for solving the nonlinear system of DAE solver
        quad_type (str): Quadrature type
        num_nodes (int): Number of nodes inside a time step

    Returns:
        description (dict): contains all information for a controller run
        controller_params (dict): specific parameters for the controller
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = quad_type
    sweeper_params['num_nodes'] = num_nodes

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = newton_tol  # tolerance for implicit solver
    problem_params['nvars'] = nvars

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 40

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [approx_solution_hook, error_hook]

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description, controller_params


def controller_run(t0, Tend, controller_params, description):
    """
    Executes a controller run for time interval to be specified in the arguments
    Args:
        t0 (float): initial time of simulation
        Tend (float): end time of simulation
        controller_params (dict): parameters needed for the controller
        description (dict): contains all information for a controller run

    Returns:
        stats (dict): raw statistics from a controller run
    """

    # instantiate the controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return stats


def main():
    """
    Main function, it executes the simulation of a problem class using a DAE sweeper
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    nvars = 3
    problem_class = simple_dae_1
    newton_tol = 1e-12

    t0 = 0.0
    Tend = 1.0
    dt_list = [2 ** (-m) for m in range(2, 12)]

    for dt_item in dt_list:
        print(f'Controller run -- Simulation for step size: {dt_item}')

        description, controller_params = get_description(dt_item, nvars, problem_class, newton_tol)

        stats = controller_run(t0, Tend, controller_params, description)


if __name__ == "__main__":
    main()
