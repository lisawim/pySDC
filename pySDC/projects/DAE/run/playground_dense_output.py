import numpy as np

from pySDC.projects.DAE.misc.log_solution_dense_output import LogSolutionDenseOutput
from pySDC.projects.DAE import DenseOutput

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run import getEndTime, computeSolution, getColor, getMarker, Plotter


def main():
    problemName = 'VAN-DER-POL'
    nNodes = 3

    QI = 'LU'
    problemType = 'SPP'

    t0 = 0.0
    dt = 1e-1
    Tend = 0.5#getEndTime(problemName)

    eps = 1e-1

    hook_class = [LogSolutionDenseOutput]

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

    u_val_dense_output = get_sorted(solutionStats, type='log_solution_dense_output', sortby='time')
    nodes_val_dense_output = get_sorted(solutionStats, type='log_nodes_dense_output', sortby='time')

    print(nodes_val_dense_output)
    print(u_val_dense_output)


if __name__ == "__main__":
    main()
