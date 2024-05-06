import numpy as np
from scipy.sparse.linalg import gmres
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.problem_classes.odeScalar import ProtheroRobinson
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.GMRES_SDC.sweeper_classes.GMRES_SDC import GMRES_SDC
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep


def main():
    Path("data").mkdir(parents=True, exist_ok=True)

    # initialize level parameters
    level_params = {
        'restol': 1e-12,
    }

    problem_params = {
        'epsilon': 0.02,#1e-5,
    }

    # initialize sweeper parameters
    QI = 'IE'
    M = 3
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': QI,
        'initial_guess': 'spread',
    }

    # initialize step parameters
    maxiter = 12
    step_params = {
        'maxiter': maxiter,
    }

    # initialize controller parameters
    controller_params = {
        'logger_level': 30,
        'hook_class': [LogGlobalErrorPostStep],
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

    t0, Tend = 0.0, 10.0#10.0
    dt_list = np.logspace(-2.0, 1.0, num=10)
    restarts = [1]#np.arange(1, M + 2, 1)

    plt.figure(figsize=(8.5, 8.5))
    for k0 in restarts:
        err_norms = []
        for dt in dt_list:
            description['level_params'].update({'dt': dt})
            description['sweeper_params'].update({'k0': k0})

            controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)
            uex = P.u_exact(Tend)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            print(f"Numerical error after iterations is {abs(uex - uend)}")

            errors = [me[1] for me in get_sorted(stats, type='e_global_post_step', sortby='time')]
            err_norms.append(max(errors))

        order_ref = [err_norms[-1] * (dt_list[m] / dt_list[-1]) ** (2 * M - 1) for m in range(len(dt_list))]

        plt.loglog(dt_list, err_norms, marker='*', label=rf"$k_0$ = {k0}")
        plt.loglog(dt_list, order_ref, color='k', linestyle='solid')
    plt.legend(frameon=False, fontsize=12, loc='upper right')
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.ylim(1e-16, 1e1)
    plt.xlabel(r"Time step size $\Delta t$", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.savefig(f"data/plotOrderAccuracy_{QI}", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
