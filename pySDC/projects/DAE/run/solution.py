import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import getEndTime, computeSolution, getColor, getLabel, getMarker, Plotter


def run(epsList, problemName, dt=1e-2):
    r"""
    Plots the solution of a problem for the SPP and corresponding DAEs.

    Parameters
    ----------
    epsList : list
        List of parameter :math:`\varepsilon`.
    problemName : str
        Name of the problem.
    dt : float
        Time step size.
    """

    nNodes = 20

    QI = "MIN-SR-NS"#"LU"
    dt = 1e-1#1e-5
    t0 = 0.0
    Tend = getEndTime(problemName)

    hook_class = [LogSolution]

    # Define a dictionary with problem types and their respective parameters
    problems = {
        # "SPP": epsList,
        "embeddedDAE": [0.0],
        # "fullyImplicitDAE": [0.0],
        # "constrainedDAE": [0.0],
    }

    solutionPlotter = Plotter(nrows=1, ncols=2, figsize=(18, 6))
    for problemType, epsValues in problems.items():
        for i, eps in enumerate(epsValues):
            print(f"\nComputing solution with time step size {dt} for {problemType} with {eps=} with {nNodes} nodes...\n")

            solutionStats = computeSolution(
                problemName=problemName,
                t0=t0,
                dt=dt,
                Tend=Tend,
                nNodes=nNodes,
                QI=QI,
                problemType=problemType,
                hookClass=hook_class,
                eps=eps,
                newton_tol=1e-14,
                solver_type="direct",
            )

            u_val = get_sorted(solutionStats, type="u", sortby="time")

            t = np.array([me[0] for me in u_val])
            if not eps == 0.0:
                # u = np.array([me[1] for me in u_val])
                # y = u[:, 0]
                # z = u[:, -1]
                y = np.array([me[1][0] for me in u_val])
                z = np.array([me[1][1] for me in u_val])
            else:
                y = np.array([me[1].diff[0] for me in u_val])
                z = np.array([me[1].alg[0] for me in u_val])
                # u = np.concatenate((y, z), axis=1)

            color, problemLabel = getColor(problemType, i, QI), getLabel(problemType, eps, QI)
            solutionPlotter.plot(t, y, subplot_index=0, color=color, label=problemLabel)
            solutionPlotter.plot(t, z, subplot_index=1, color=color)

    solutionPlotter.set_xlabel(r"$t$", subplot_index=None)
    solutionPlotter.set_ylabel(r"$y$", subplot_index=0)
    solutionPlotter.set_ylabel(r"$z$", subplot_index=1)

    solutionPlotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=6, fontsize=18)

    filename = "data" + "/" + f"{problemName}" + "/" + f"solution.png"
    solutionPlotter.save(filename)


if __name__ == "__main__":
    run([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], 'LINEAR-TEST')
    # run([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], 'MICHAELIS-MENTEN')
    # run([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'MICHAELIS-MENTEN')
    # run([10 ** (-m) for m in range(1, 5)], 'VAN-DER-POL', dt=1e-5)