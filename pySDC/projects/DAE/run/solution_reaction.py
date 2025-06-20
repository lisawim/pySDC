import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAE

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

    nNodes = 3

    QI = "LU"
    dt = 1e-2
    t0 = 0.0

    nvars = 64
    kwargs = {"nvars": nvars}

    hook_class = [LogSolution]

    # Define a dictionary with problem types and their respective parameters
    problem_type = "semiImplicitDAE"
    eps = 0.0

    prob = ReactionDiffusionPDAE(nvars=nvars)

    solutionPlotter = Plotter(nrows=2, ncols=2, figsize=(18, 18))

    solutionStats = computeSolution(
        problemName=problemName,
        t0=t0,
        dt=dt,
        Tend=getEndTime(problemName),
        nNodes=nNodes,
        QI=QI,
        problemType=problem_type,
        hookClass=hook_class,
        eps=eps,
        newton_tol=1e-14,
        **kwargs,
    )

    u_stats = get_sorted(solutionStats, type="u", sortby="time")
    u_val = [me[1] for me in u_stats]
    t = [me[0] for me in u_stats]
    u = np.array([me.diff[:nvars] for me in u_val])
    v = np.array([me.diff[nvars:] for me in u_val])
    w = np.array([me.alg[:nvars] for me in u_val])

    x = prob.xvalues

    # color = getColor(problemType, i, QI)
    # solutionPlotter.plot(x, u[-1], subplot_index=0, color=color)
    # solutionPlotter.plot(x, v[-1], subplot_index=1, color=color)
    # solutionPlotter.plot(x, w[-1], subplot_index=2, color=color)

    fig, ax = plt.subplots()
    line_u, = ax.plot([], [], label="u(x,t)")
    line_v, = ax.plot([], [], linestyle="dashed", label="v(x,t)")
    line_w, = ax.plot([], [], label="w(x,t)")
    ax.set_xlim(0, 1)
    ax.set_ylim(np.min([u,v,w]), np.max([u,v,w]))
    # ax.set_ylim(np.min([u,v]), np.max([u,v]))
    ax.set_xlabel("x")
    ax.set_ylabel("Concentration")
    ax.legend()
    title = ax.set_title("")

    def init():
        line_u.set_data([], [])
        line_v.set_data([], [])
        line_w.set_data([], [])
        title.set_text("")
        return line_u, line_v, line_w, title

    def animate(i):
        line_u.set_data(x, u[i])
        line_v.set_data(x, v[i])
        line_w.set_data(x, w[i])
        title.set_text(f"t = {t[i]:.2f}")
        return line_u, line_v, line_w, title
    
    ani = animation.FuncAnimation(
        fig, animate, frames=len(t), init_func=init,
        blit=True, interval=50
    )

    filename_mp4 = "data" + "/" + f"{problemName}" + "/" + "reaction_diffusion.mp4"
    ani.save(filename_mp4, fps=20)

    # solutionPlotter.set_xlabel(r"$x$", subplot_index=None)
    # solutionPlotter.set_ylabel(r"$u(x,T)$", subplot_index=0)
    # solutionPlotter.set_ylabel(r"$v(x,T)$", subplot_index=1)
    # solutionPlotter.set_ylabel(r"$w(x,T)$", subplot_index=2)

    # solutionPlotter.adjust_layout(num_subplots=3)

    # filename = "data" + "/" + f"{problemName}" + "/" + f"solution.png"
    # solutionPlotter.save(filename)


if __name__ == "__main__":
    # run([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], 'TELEGRAPHER')
    run([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], 'REACTION-DIFFUSION')
