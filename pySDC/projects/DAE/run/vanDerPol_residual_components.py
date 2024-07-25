import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.playgrounds.DAE.log_residual_components import LogResidualComponentsPostStep

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import getEndTime, computeSolution, getColor, getMarker, Plotter


def run():
    problemName = 'VAN-DER-POL'
    nNodes = 3

    QI = 'LU'
    problemType = 'SPP'

    t0 = 0.0
    dt = 1e-3
    Tend = getEndTime(problemName)

    # for this script set residual tolerance and e_tol=-1 instead
    restol = 1e-13
    e_tol = -1

    logResidualComp = True
    hook_class = [LogSolution, LogResidualComponentsPostStep]
    skip_residual_computation = ()

    kwargs = {
        # 'restol': restol,
        # 'e_tol': e_tol,
        'logResidualComp': logResidualComp,
        'skip_residual_computation': skip_residual_computation,
    }

    epsList = [10 ** (-m) for m in range(1, 5)]
    residualPlotter = Plotter(nrows=2, ncols=2, figsize=(30, 18))

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
            **kwargs,
        )

        u_val = get_sorted(solutionStats, type='u', sortby='time')
        t = np.array([me[0] for me in u_val])
        u = np.array([me[1] for me in u_val])

        residual_comp_val = get_sorted(solutionStats, type='residual_comp_post_step', sortby='time')
        residual_comp = np.array([me[1] for me in residual_comp_val])

        color, marker = getColor(problemType, i), getMarker(problemType)
        residualPlotter.plot(t, u[:, 0], subplot_index=0, color=color, marker=marker, label=rf'$\varepsilon=${eps}')
        residualPlotter.plot(t, u[:, -1], subplot_index=1, color=color, marker=marker)

        residualPlotter.plot(t, residual_comp[:, 0], subplot_index=2, color=color, marker=marker, plot_type='semilogy')
        residualPlotter.plot(t, residual_comp[:, -1], subplot_index=3, color=color, marker=marker, plot_type='semilogy')

    residualPlotter.set_xlabel(r'$t$', subplot_index=None)
    residualPlotter.set_ylabel(r'$y$', subplot_index=0)
    residualPlotter.set_ylabel(r'$z$', subplot_index=1)
    residualPlotter.set_ylabel(r'Residual component in $f(y, z)$', subplot_index=2)
    residualPlotter.set_ylabel(r'Residual component in $g(y, z)$', subplot_index=3)

    residualPlotter.set_ylim((1e-16, 1e-10), subplot_index=2)
    residualPlotter.set_ylim((1e-16, 1e-10), subplot_index=3)

    residualPlotter.set_legend(subplot_index=0, loc='lower left')

    filename = "data" + "/" + f"{problemName}" + "/" + f"residual_components.png"
    residualPlotter.save(filename)


if __name__ == "__main__":
    run()
