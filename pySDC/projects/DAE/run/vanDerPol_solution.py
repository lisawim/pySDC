import numpy as np
from mpi4py import MPI

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.core.errors import ProblemError

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run import getEndTime, computeSolution, getColor, getMarker, Plotter


def run():
    r"""
    Plots the numerical solution especially for the Van der pol problem. This script is intended
    to run in parallel to accelerate the execution.

    TODO: Extend script for DAE case:

    - for-loop for ODE case as well as DAE case(s)
    - also numerical DAE solutions are plotted with correct labels
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        if rank == 0:
            raise ProblemError("This script is intended to run in parallel!")

    problemName = 'VAN-DER-POL'
    nNodes = comm.Get_size()

    QI = 'MIN-SR-S'
    problemType = 'SPP'

    t0 = 0.0
    dt = 1e-3
    Tend = getEndTime(problemName)

    hook_class = [LogSolution]

    epsList = [10 ** (-m) for m in range(1, 5)]
    if rank == 0:
        solutionPlotter = Plotter(nrows=2, ncols=1, orientation='vertical', figsize=(18, 16))

    for i, eps in enumerate(epsList):
        solutionStats = computeSolution(
            problemName=problemName,
            t0=t0,
            dt=dt,
            Tend=Tend,
            nNodes=nNodes,
            QI=QI,
            problemType=problemType,
            useMPI=True,
            hookClass=hook_class,
            eps=eps,
            comm=comm,
        )

        if rank == 0:
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

    if rank == 0:
        solutionPlotter.set_xlabel(r'$t$', subplot_index=None)
        solutionPlotter.set_ylabel(r'$y$', subplot_index=0)
        solutionPlotter.set_ylabel(r'$z$', subplot_index=1)

        solutionPlotter.set_legend(subplot_index=0, loc='lower left')

        filename = "data" + "/" + f"{problemName}" + "/" + f"solution.png"
        solutionPlotter.save(filename)


if __name__ == "__main__":
    run()
