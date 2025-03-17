import numpy as np
import dill

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getEndTime

if __name__ == "__main__":
    problem_name = "MICHAELIS-MENTEN"
    problem_type = "fullyImplicitDAE"
    eps = 0.0

    QI = "RadauIIA7"

    t0 = 0.0
    dt = 1e-7
    Tend = 0.021

    hook_class = [LogSolution]

    kwargs = {"maxiter": 1, "nsweeps": 1, "e_tol": -1}

    # Let's do the simulation to get results
    solution_stats = computeSolution(
        problemName=problem_name,
        t0=t0,
        dt=dt,
        Tend=getEndTime(problem_name),
        nNodes=4,
        QI=QI,
        problemType=problem_type,
        hookClass=hook_class,
        eps=eps,
        **kwargs,
    )

    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t = np.array([round(me[0], 13) for me in u_val])
    y = np.array([me[1].diff[0] for me in u_val])
    z = np.array([me[1].alg[0] for me in u_val])

    results = {t_item: (y_item, z_item) for t_item, y_item, z_item in zip(t, y, z)}

    with open(f"refSol_SciPy_michaelisMentenDAE_RadauIIA7_{dt=}.dat", "wb") as f:
        dill.dump(results, f)
