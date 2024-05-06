import numpy as np
import scipy as sp
import scipy.sparse.linalg as spla
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.problem_classes.odeScalar import ProtheroRobinson
from pySDC.playgrounds.GMRES_SDC.sweeper_classes.GMRES_SDC import GMRES_SDC
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.playgrounds.GMRES_SDC.hooks.log_GMRES_residual import LogGMRESResidualPostStep


def non_f(eps, t):
    return 1 / eps * np.cos(t) - np.sin(t)


def main():
    r"""
    Computes one time step by solving the preconditioned linear system using GMRES. The relative residual
    from GMRES iterations for different numbers of restarts are then plotted.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    # initialize level parameters
    dt = 1.0
    restol = 1e-12
    level_params = {
        'restol': restol,
        'dt': dt,
    }

    eps = 1e-5
    problem_params = {
        'epsilon': eps,
    }

    # initialize sweeper parameters
    M = 12
    QI = 'EE'
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': QI,
        'initial_guess': 'spread',
    }

    # initialize step parameters
    maxiter = 1
    step_params = {
        'maxiter': maxiter,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogGMRESResidualPostStep],
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': ProtheroRobinson,
        'problem_params': problem_params,
        'sweeper_class': GMRES_SDC,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    t0, Tend = 0.0, dt

    plt.figure(figsize=(8.5, 8.5))
    marker = ['s', 'o', 'd', '*', '>', '<']

    num_restarts = [1, 2, 3, 4, 6, 12]
    for i, restart in enumerate(num_restarts):
        description['sweeper_params'].update({'k0': restart})

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

        # get initial values on finest level
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)
        uex = P.u_exact(Tend)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        print(f"Numerical error after iterations is {abs(uex - uend)}")
        pr_norm = [me[1] for me in get_sorted(stats, type='relative_residual_gmres_post_step', sortby='time')][0]
        print(pr_norm)
        plt.semilogy(np.arange(1, len(pr_norm) + 1), pr_norm, marker=marker[i], label=rf"$k_0$ = {restart}")

    plt.legend(frameon=False, fontsize=12, loc='upper right')
    plt.yscale('log', base=10)
    plt.xlabel("Iterations in GMRES", fontsize=16)
    plt.ylabel("Relative residual", fontsize=16)
    plt.savefig(f"data/relativeResidualRestarts_{QI}", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
