import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import getEndTime, computeSolution, getColor, getMarker, Plotter


def run():
    r"""
    Plots the numerical solution especially for the Van der pol problem.

    TODO: Extend script for DAE case:

    - for-loop for ODE case as well as DAE case(s)
    - also numerical DAE solutions are plotted with correct labels
    """

    problemName = 'VAN-DER-POL'
    nNodes = 3

    QI = 'LU'

    use_adaptivity = True
    e_tol_adaptivity = 1e-7

    t0 = 0.0
    dt = 1e-3
    Tend = getEndTime(problemName)

    hook_class = [LogSolution]

    epsList = [10 ** (-m) for m in range(1, 5)]

    # Define a dictionary with problem types and their respective parameters
    problems = {
        'SPP': epsList,
        # 'embeddedDAE': [0.0],
        # 'constrainedDAE': [0.0],
    }

    solutionPlotter = Plotter(nrows=2, ncols=1, orientation='vertical', figsize=(18, 16))
    for problemType, epsValues in problems.items():
        for i, eps in enumerate(epsValues):
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
                use_adaptivity=use_adaptivity,
                e_tol_adaptivity=e_tol_adaptivity,
            )

            u_val = get_sorted(solutionStats, type='u', sortby='time')
            t = np.array([me[0] for me in u_val])
            if not eps == 0.0:
                u = np.array([me[1] for me in u_val])
            else:
                y = np.array([me[1].diff[0] for me in u_val])
                z = np.array([me[1].alg[0] for me in u_val])
                u = np.concatenate((y, z), axis=1)

            color, marker = getColor(problemType, i), getMarker(problemType)
            solutionPlotter.plot(t, u[:, 0], subplot_index=0, color=color, marker=marker, label=rf'$\varepsilon=${eps}')
            solutionPlotter.plot(t, u[:, -1], subplot_index=1, color=color, marker=marker)

    solutionPlotter.set_xlabel(r'$t$', subplot_index=None)
    solutionPlotter.set_ylabel(r'$y$', subplot_index=0)
    solutionPlotter.set_ylabel(r'$z$', subplot_index=1)

    solutionPlotter.set_legend(subplot_index=0, loc='lower left')

    filename = "data" + "/" + f"{problemName}" + "/" + f"solution.png"
    solutionPlotter.save(filename)


if __name__ == "__main__":
    run()
