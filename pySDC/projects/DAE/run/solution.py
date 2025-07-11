import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import getEndTime, computeSolution, getColor, getLabel, getMarker, Plotter


def u_exact(t, problem_name="LINEAR-STIFF"):
    if problem_name == "LINEAR-STIFF":
        u_ex = np.array([np.cos(t), np.exp(t), np.sin(t), -np.cos(t)])
    else:
        raise NotImplementedError
    
    return u_ex


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

    QI = "IE"#"LU"
    dt = 1e-2#1e-5
    t0 = 0.0
    Tend = getEndTime(problemName)

    hook_class = [LogSolution]

    # Define a dictionary with problem types and their respective parameters
    problems = {
        # "SPP": epsList,
        "embeddedDAE": [0.0],
        # "fullyImplicitDAE": [0.0],
        # "constrainedDAE": [0.0],
        # "semiImplicitDAE": [0.0],
    }

    # solutionPlotter = Plotter(nrows=1, ncols=2, figsize=(18, 6))
    solutionPlotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))
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
            )

            u_val = get_sorted(solutionStats, type="u", sortby="time")

            niters = [me[1] for me in get_sorted(solutionStats, type="niter", sortby="time")]
            print(niters)
            t = np.array([me[0] for me in u_val])
            # if not eps == 0.0:
            #     # u = np.array([me[1] for me in u_val])
            #     # y = u[:, 0]
            #     # z = u[:, -1]
            #     y = np.array([me[1][0] for me in u_val])
            #     z = np.array([me[1][1] for me in u_val])
            # else:
            #     y = np.array([me[1].diff[0] for me in u_val])
            #     z = np.array([me[1].alg[0] for me in u_val])
            #     # u = np.concatenate((y, z), axis=1)

            y1 = np.array([me[1].diff[0] for me in u_val])
            y2 = np.array([me[1].diff[1] for me in u_val])
            y3 = np.array([me[1].diff[2] for me in u_val])
            y4 = np.array([me[1].alg[0] for me in u_val])

            u_ex = np.array([u_exact(t_item, problem_name=problemName) for t_item in t])
            print(u_ex.shape)
            color, problemLabel = getColor(problemType, i, QI), getLabel(problemType, eps, QI)
            solutionPlotter.plot(t, y1, subplot_index=0, color=color, label=problemLabel)
            solutionPlotter.plot(t, u_ex[:, 0], subplot_index=0, color="black", linestyle="dashed")

            solutionPlotter.plot(t, y2, subplot_index=1, color=color)
            solutionPlotter.plot(t, u_ex[:, 1], subplot_index=1, color="black", linestyle="dashed")

            solutionPlotter.plot(t, y3, subplot_index=2, color=color)
            solutionPlotter.plot(t, u_ex[:, 2], subplot_index=2, color="black", linestyle="dashed")

            solutionPlotter.plot(t, y4, subplot_index=3, color=color)
            solutionPlotter.plot(t, u_ex[:, 3], subplot_index=3, color="black", linestyle="dashed")

    solutionPlotter.set_xlabel(r"$t$", subplot_index=None)
    solutionPlotter.set_ylabel(r"$y$", subplot_index=0)
    solutionPlotter.set_ylabel(r"$z$", subplot_index=1)

    solutionPlotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=6, fontsize=18)

    filename = "data" + "/" + f"{problemName}" + "/" + f"solution.png"
    solutionPlotter.save(filename)


if __name__ == "__main__":
    # run([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], 'LINEAR-TEST')
    run([0.0], 'LINEAR-STIFF')
    # run([0.0], "ANDREWS-SQUEEZER")
    # run([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01], 'MICHAELIS-MENTEN')
    # run([1e-1, 1e-2, 1e-3, 1e-4, 1e-5], 'MICHAELIS-MENTEN')
    # run([10 ** (-m) for m in range(1, 5)], 'VAN-DER-POL', dt=1e-5)