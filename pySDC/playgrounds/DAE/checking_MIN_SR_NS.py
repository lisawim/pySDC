from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE
from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAE
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.projects.DAE.misc.hooksDAE import (
    LogGlobalErrorPostStepDifferentialVariable,
    LogGlobalErrorPostStepAlgebraicVariable,
    LogGlobalErrorPostIterDiff,
    LogGlobalErrorPostIterAlg,
)


def run(dt, num_nodes, QI):
    # initialize level parameters
    level_params = {
        'restol': -1,
        'e_tol': -1,
        'dt': dt,
    }

    # initialize problem parameters
    problem_params = {
        'solver_type': "direct",
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': num_nodes,
        'QI': QI,
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 1000,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [
            # LogGlobalErrorPostStepDifferentialVariable,
            # LogGlobalErrorPostStepAlgebraicVariable,
            LogGlobalErrorPostIterDiff,
            LogGlobalErrorPostIterAlg,
        ],
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': LinearTestDAE,
        'problem_params': problem_params,
        'sweeper_class': FullyImplicitDAE,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    t0 = 0.0
    Tend = t0 + dt

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uex = P.u_exact(t0 + dt)

    Path("data").mkdir(parents=True, exist_ok=True)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    err = abs(uex - uend)
    err_diff = abs(uex.diff[0] - uend.diff[0])
    err_alg = abs(uex.alg[0] - uend.alg[0])
    print(f"Error: {err}")
    print(f"Diff. error: {err_diff}")
    print(f"Alg. error : {err_alg} \n")

    err_diff_values = [me[1] for me in get_sorted(stats, type=f"e_global_differential_post_iteration", sortby="iter")]
    err_alg_values = [me[1] for me in get_sorted(stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]

    print(err_diff_values)
    print(err_alg_values)


if __name__ == "__main__":
    QI = "MIN-SR-NS"
    num_nodes = 9
    dt = 1e-2
    run(dt, num_nodes, QI)