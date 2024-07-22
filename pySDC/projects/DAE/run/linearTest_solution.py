import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run import getEndTime, computeSolution, getColor, getLabel, getMarker, Plotter


def run():
    problemName = 'LINEAR-TEST'
    nNodes = 3

    QI = 'LU'

    t0 = 0.0
    dt = 1e-2
    Tend = getEndTime(problemName)

    hook_class = [LogSolution]

    # Define the list of epsilon values for 'SPP' problem
    epsList = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    # Define a dictionary with problem types and their respective parameters
    problems = {
        'SPP': epsList,
        'embeddedDAE': [0.0],
        'constrainedDAE': [0.0]
    }

    solutionPlotter = Plotter(nrows=1, ncols=2, orientation='horizontal', figsize=(30, 10))
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
            )

            u_val = get_sorted(solutionStats, type='u', sortby='time')
            t = np.array([me[0] for me in u_val])
            if not eps == 0.0:
                u = np.array([me[1] for me in u_val])
            else:
                y = np.array([me[1].diff[0] for me in u_val])
                z = np.array([me[1].alg[0] for me in u_val])
                u = np.concatenate((y, z), axis=1)

            color, marker, problemLabel = getColor(problemType, i), getMarker(problemType), getLabel(problemType)
            label = rf'$\varepsilon=${eps}' + problemLabel
            solutionPlotter.plot(t, u[:, 0], subplot_index=0, color=color, marker=marker, label=label)
            solutionPlotter.plot(t, u[:, -1], subplot_index=1, color=color, marker=marker)

    solutionPlotter.set_xlabel(r'$t$', subplot_index=None)
    solutionPlotter.set_ylabel(r'$y$', subplot_index=0)
    solutionPlotter.set_ylabel(r'$z$', subplot_index=1)

    solutionPlotter.set_legend(subplot_index=0, loc='upper right')

    filename = "data" + "/" + f"{problemName}" + "/" + f"solution.png"
    solutionPlotter.save(filename)


if __name__ == "__main__":
    run()
