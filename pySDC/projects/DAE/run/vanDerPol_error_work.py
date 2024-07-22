import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_work import LogWork
from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run import getEndTime, computeSolution, getColor, getMarker, Plotter


def run():
    problemName = 'VAN-DER-POL'
    nNodes = 3

    QI = 'LU'
    problemType = 'SPP'

    t0 = 0.0
    dt = 1e-3
    Tend = getEndTime(problemName)

    hook_class = [LogSolution, LogWork, LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation]

    epsList = [10 ** (-m) for m in range(1, 5)]
    workPlotter = Plotter(nrows=3, ncols=2, figsize=(30, 26))
    for i, eps in enumerate(epsList):
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
        u = np.array([me[1] for me in u_val])

        errDiffValues = np.array(get_sorted(solutionStats, type='e_global_post_step', sortby='time'))
        errAlgValues = np.array(get_sorted(solutionStats, type='e_global_algebraic_post_step', sortby='time'))

        newton = np.array(get_sorted(solutionStats, type='work_newton', sortby='time'))
        nIter = np.array(get_sorted(solutionStats, type='niter', sortby='time'))

        color, marker = getColor(problemType, i), getMarker(problemType)
        workPlotter.plot(t, u[:, 0], subplot_index=0, color=color, marker=marker, label=rf'$\varepsilon=${eps}')
        workPlotter.plot(t, u[:, -1], subplot_index=1, color=color, marker=marker)

        workPlotter.plot(t, errDiffValues[:, 1], subplot_index=2, color=color, marker=marker, plot_type='semilogy')
        workPlotter.plot(t, errAlgValues[:, 1], subplot_index=3, color=color, marker=marker, plot_type='semilogy')

        workPlotter.plot(t, newton[:, 1], subplot_index=4, color=color, marker=marker, plot_type='semilogy')
        workPlotter.plot(t, nIter[:, 1], subplot_index=5, color=color, marker=marker)

    workPlotter.set_xlabel(r'$t$', subplot_index=None)
    workPlotter.set_ylabel(r'$y$', subplot_index=0)
    workPlotter.set_ylabel(r'$z$', subplot_index=1)
    workPlotter.set_ylabel(r'global error in $y$', subplot_index=2)
    workPlotter.set_ylabel(r'global error in $z$', subplot_index=3)
    workPlotter.set_ylabel(r'Newton iterations in each step', subplot_index=4)
    workPlotter.set_ylabel(r'SDC iterations in each step', subplot_index=5)

    workPlotter.set_ylim((1e-15, 1e1), subplot_index=2)
    workPlotter.set_ylim((1e-15, 1e1), subplot_index=3)
    workPlotter.set_ylim((1e0, 6.5e3), subplot_index=4)
    workPlotter.set_ylim((1, 25), subplot_index=5)

    workPlotter.set_legend(subplot_index=0, loc='lower left')

    filename = "data" + "/" + f"{problemName}" + "/" + f"error_work.png"
    workPlotter.save(filename)


if __name__ == "__main__":
    run()
