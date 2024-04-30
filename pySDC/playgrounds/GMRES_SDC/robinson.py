import numpy as np
from scipy.sparse.linalg import gmres
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.core.Step import step
from pySDC.implementations.problem_classes.odeScalar import ProtheroRobinson
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


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
    level_params = {
        'restol': 1e-9,
        'dt': dt,
    }

    eps = 0.02
    problem_params = {
        'epsilon': eps,
    }

    # initialize sweeper parameters
    M = 12
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': M,
        'QI': 'EE',
        'initial_guess': 'spread',
    }

    # initialize step parameters
    step_params = {
        'maxiter': 10,
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': ProtheroRobinson,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }

    S = step(description=description)

    L = S.levels[0]
    P = L.prob

    L.status.time = 0.0
    L.u[0] = P.u_exact(L.time)
    L.sweep.predict()

    QImat = generic_implicit(sweeper_params).QI[1:, 1:]
    Qmat = generic_implicit(sweeper_params).coll.Qmat[1:, 1:]

    nodes = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(M)]

    u0full = np.array([L.u[m].flatten() for m in range(1, M + 1)]).flatten()
    ffull = np.array([non_f(eps, tau) for tau in nodes])

    LHSmat = np.matmul(np.linalg.inv(np.eye(M) - dt * (- 1 / eps) * QImat), np.eye(M) - dt * (- 1 / eps) * Qmat)
    RHSvec = np.linalg.inv(np.eye(M) - dt * (- 1 / eps) * QImat).dot(u0full + dt * Qmat.dot(ffull))

    plt.figure(figsize=(8.5, 8.5))
    marker = ['s', 'o', 'd', '*', '>', '<']

    num_restarts = np.arange(1, 13, 2)
    for i, restart in enumerate(num_restarts):
        res_restarts = []
        callback_pr_norm = lambda res: res_restarts.append(res)

        sol = gmres(
            LHSmat,
            RHSvec.flatten(),
            x0=u0full.flatten(),
            restart=restart,
            atol=1e-14,
            rtol=0,
            maxiter=5,
            callback=callback_pr_norm,
            callback_type='pr_norm',
        )[0]

        plt.semilogy(np.arange(1, len(res_restarts) + 1), res_restarts, marker=marker[i], label=rf"$k_0$ = {restart}")
    
        unew = sol[-1]
        uex = P.u_exact(L.time + dt)

        err = abs(uex - unew)
        print(f"Numerical error is {err}")

    plt.legend(frameon=False, fontsize=12, loc='upper right')
    plt.yscale('log', base=10)
    plt.xlabel("Iterations in GMRES", fontsize=16)
    plt.ylabel("Relative residual", fontsize=16)
    plt.savefig("data/relativeResidualRestarts", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
