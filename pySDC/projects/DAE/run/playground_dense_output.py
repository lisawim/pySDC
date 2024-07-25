import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.DAE.misc.log_solution_dense_output import LogSolutionDenseOutput
from pySDC.projects.DAE import DenseOutput

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getMarker, Plotter


def main():
    problemName = 'VAN-DER-POL'
    nNodes = 3

    QI = 'LU'
    problemType = 'constrainedDAE'
    # problemType = 'SPP'

    t0 = 0.0
    dt = 1e-2
    Tend = 0.7#0.804

    eps = 0.0
    # eps = 1e-10

    hook_class = [LogSolution, LogSolutionDenseOutput]

    use_adaptivity = True
    e_tol_adaptivity = 100 * np.finfo(float).eps

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

    # get values of u and corresponding times for all nodes
    u_dense = get_sorted(solutionStats, type='u_dense', sortby='time', recomputed=False)
    nodes_dense = get_sorted(solutionStats, type='nodes_dense', sortby='time')
    sol = DenseOutput(nodes_dense, u_dense)

    # choose evaluation interval
    t_eval = [t0 + i * dt for i in range(int(Tend / dt) + 1)]
    u_eval = [sol.__call__(t_item) for t_item in t_eval]

    # Usual logged solution serves as reference
    u_ref_val = get_sorted(solutionStats, type='u', sortby='time')

    t = np.array([me[0] for me in u_ref_val])
    if not eps == 0.0:
        u = np.array([me[1] for me in u_ref_val])
        u_eval = np.array(u_eval)
    else:
        y = np.array([me[1].diff[0] for me in u_ref_val])
        y_eval = np.array([me.diff[0] for me in u_eval])

        z = np.array([me[1].alg[0] for me in u_ref_val])
        z_eval = np.array([me.alg[0] for me in u_eval])

        u = np.column_stack((y, z))
        u_eval = np.column_stack((y_eval, z_eval))

    denseOutputPlotter = Plotter(nrows=2, ncols=1, orientation='vertical', figsize=(18, 16))

    color, marker = getColor(problemType, -1), getMarker(problemType)
    denseOutputPlotter.plot(t, u[:, 0], subplot_index=0, color=color, marker=marker, label=rf'$\varepsilon=${eps}')
    denseOutputPlotter.plot(t, u[:, -1], subplot_index=1, color=color, marker=marker)
    denseOutputPlotter.plot(t_eval, u_eval[:, 0], subplot_index=0, color='g', marker='d', label=rf'$\varepsilon=${eps} - Dense output')
    denseOutputPlotter.plot(t_eval, u_eval[:, -1], subplot_index=1, color='g', marker='d')

    denseOutputPlotter.set_xlabel(r'$t$', subplot_index=None)
    denseOutputPlotter.set_ylabel(r'$y$', subplot_index=0)
    denseOutputPlotter.set_ylabel(r'$z$', subplot_index=1)

    denseOutputPlotter.set_legend(subplot_index=0, loc='lower left')

    filename = "data" + "/" + f"{problemName}" + "/" + f"solution_dense_output.png"
    denseOutputPlotter.save(filename)


if __name__ == "__main__":
    main()
