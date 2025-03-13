import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getEndTime

if __name__ == "__main__":
    problem_name = "MICHAELIS-MENTEN"
    problem_type = "fullyImplicitDAE"
    eps = 0.0

    QI = "RadauIIA7"

    t0 = 0.0
    dt = 1e-5
    Tend = 0.02

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
