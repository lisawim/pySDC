import numpy as np
import dill

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution

if __name__ == "__main__":
    problem_name = "ANDREWS-SQUEEZER"
    problem_type = "fullyImplicitDAE"
    eps = 0.0

    QI = "RadauIIA7"

    t0 = 0.0
    dt = 1e-7
    Tend = 0.031

    hook_class = [LogSolution]

    kwargs = {"maxiter": 1, "nsweeps": 1, "e_tol": -1}

    # Let's do the simulation to get results
    solution_stats = computeSolution(
        problemName=problem_name,
        t0=t0,
        dt=dt,
        Tend=Tend,
        nNodes=4,
        QI=QI,
        problemType=problem_type,
        hookClass=hook_class,
        eps=eps,
        **kwargs,
    )

    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t_solve = np.array([me[0] for me in u_val])
    u_diff_solve = np.array([me[1].diff[: 14] for me in u_val])
    u_alg_solve = np.array([me[1].alg[: 13] for me in u_val])

    np.save("t_solve_andrews.npy", t_solve)
    np.save("u_diff_solve_andrews.npy", u_diff_solve)
    np.save("u_alg_solve_andrews.npy", u_alg_solve)

    print("Results saved successfully!")
