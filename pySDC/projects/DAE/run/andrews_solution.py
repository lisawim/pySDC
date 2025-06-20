import numpy as np
from pathlib import Path

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_work import LogWork

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import getEndTime, computeSolution, getColor, getLabel, getMarker, Plotter


def run(problemName, dt=1e-7):
    r"""
    Plots the solution of a problem for system of DAEs especially for Andrews' squeezer.

    Parameters
    ----------
    problemName : str
        Name of the problem.
    dt : float
        Time step size.
    """

    nNodes = 4

    QI = "LU"

    t0 = 0.0
    Tend = getEndTime(problemName)

    hook_class = [LogSolution, LogWork]

    problemType = "semiImplicitDAE" # "constrainedDAE"  # "constrainedDAE"
    eps = 0.0

    # kwargs = {"maxiter": 1, "nsweeps": 1, "e_tol": -1}

    solutionStats = computeSolution(
        problemName=problemName,
        t0=t0,
        dt=dt,
        Tend=0.031,#Tend,
        nNodes=nNodes,
        QI=QI,
        problemType=problemType,
        hookClass=hook_class,
        eps=eps,
        index=1,
        # **kwargs,
    )

    u_val = get_sorted(solutionStats, type="u", sortby="time")
    t_solve = np.array([me[0] for me in u_val])
    u_diff_solve = np.array([me[1].diff[: 14] for me in u_val])
    u_alg_solve = np.array([me[1].alg[: 13] for me in u_val])

    # np.save("/Users/lisa/Projects/Python/pySDC/pySDC/projects/DAE/data/t_solve_andrews_3.npy", t_solve)
    # np.save("/Users/lisa/Projects/Python/pySDC/pySDC/projects/DAE/data/u_diff_solve_andrews_3.npy", u_diff_solve)
    # np.save("/Users/lisa/Projects/Python/pySDC/pySDC/projects/DAE/data/u_alg_solve_andrews_3.npy", u_alg_solve)

    # print("Results saved successfully!")

    # nIter = np.sum([me[1] for me in get_sorted(solutionStats, type='niter', sortby='time')])
    # newton = np.sum([me[1] for me in get_sorted(solutionStats, type='work_newton', sortby='time')])
    # print([me[1] for me in get_sorted(solutionStats, type='niter', sortby='time')])
    # print(f"SDC iterations in total: {nIter}")
    # print(f"Newton iterations in total: {newton}")

    q_ex = [
        0.1581077119629904e2,
        -0.1575637105984298e2,
        0.4082224013073101e-1,
        -0.5347301163226948,
        0.5244099658805304,
        0.5347301163226948,
        0.1048080741042263*10,
    ]

    t = np.array([me[0] for me in u_val])

    q1 = np.array([me[1].diff[0] for me in u_val])
    q2 = np.array([me[1].diff[1] for me in u_val])
    q3 = np.array([me[1].diff[2] for me in u_val])
    q4 = np.array([me[1].diff[3] for me in u_val])
    q5 = np.array([me[1].diff[4] for me in u_val])
    q6 = np.array([me[1].diff[5] for me in u_val])
    q7 = np.array([me[1].diff[6] for me in u_val])

    path_to_data = Path("/Users/lisa/Projects/Python/pySDC/pySDC/projects/DAE/data/")
    t = np.load(path_to_data / "t_solve_andrews_3.npy")
    u_diff_solve = np.load(path_to_data / "u_diff_solve_andrews_3.npy")
    u_alg_solve = np.load(path_to_data / "u_alg_solve_andrews_3.npy")

    assert np.allclose(q1, u_diff_solve[:, 0], atol=1e-14)
    assert np.allclose(q2, u_diff_solve[:, 1], atol=1e-14)
    assert np.allclose(q3, u_diff_solve[:, 2], atol=1e-14)
    assert np.allclose(q4, u_diff_solve[:, 3], atol=1e-14)
    assert np.allclose(q5, u_diff_solve[:, 4], atol=1e-14)
    assert np.allclose(q6, u_diff_solve[:, 5], atol=1e-14)
    assert np.allclose(q7, u_diff_solve[:, 6], atol=1e-14)

    q1 = u_diff_solve[:, 0]
    q2 = u_diff_solve[:, 1]
    q3 = u_diff_solve[:, 2]
    q4 = u_diff_solve[:, 3]
    q5 = u_diff_solve[:, 4]
    q6 = u_diff_solve[:, 5]
    q7 = u_diff_solve[:, 6]

    q_sol = [q1[-1], q2[-1], q3[-1], q4[-1], q5[-1], q6[-1], q7[-1]]

    err = [abs(u_ex - u_sol) for u_ex, u_sol in zip(q_ex, q_sol)]
    print(err)

    # Transform solution to get plot as in Hairer & Wanner (1996)
    q1 = ((q1 + np.pi) % (2 * np.pi)) - np.pi
    q2 = ((q2 + np.pi) % (2 * np.pi)) - np.pi
    q3 = ((q3 + np.pi) % (2 * np.pi)) - np.pi
    q4 = ((q4 + np.pi) % (2 * np.pi)) - np.pi
    q5 = ((q5 + np.pi) % (2 * np.pi)) - np.pi
    q6 = ((q6 + np.pi) % (2 * np.pi)) - np.pi
    q7 = ((q7 + np.pi) % (2 * np.pi)) - np.pi

    solutionPlotter = Plotter(nrows=1, ncols=1, figsize=(20, 10))

    problemLabel = getLabel(problemType, eps, QI)
    solutionPlotter.plot(t, q1, label=r"$\beta$")  # label=r"$q_1$")
    solutionPlotter.plot(t, q2, label=r"$\Theta$")  # label=r"$q_2$")
    solutionPlotter.plot(t, q3, label=r"$\gamma$")  # label=r"$q_3$")
    solutionPlotter.plot(t, q4, label=r"$\Phi$")  # label=r"$q_4$")
    solutionPlotter.plot(t, q5, label=r"$\delta$")  # label=r"$q_5$")
    solutionPlotter.plot(t, q6, label=r"$\Omega$")  # label=r"$q_6$")
    solutionPlotter.plot(t, q7, label=r"$\varepsilon$")  # label=r"$q_7$")

    solutionPlotter.set_title(problemLabel)

    solutionPlotter.set_xlabel(r'$t$', subplot_index=None)
    solutionPlotter.set_ylabel(r'Solution', subplot_index=0)

    solutionPlotter.set_xlim((0.0, 0.03))
    solutionPlotter.set_ylim((-0.7, 0.7))

    solutionPlotter.set_legend(subplot_index=0, loc='upper right')

    filename = "data" + "/" + f"{problemName}" + "/" + f"solution.png"
    solutionPlotter.save(filename)

    # solutionPlotter2 = Plotter(nrows=4, ncols=2, figsize=(35, 30))
    # solutionPlotter2.plot(t, q1, subplot_index=0, label=r"$\beta$")  # label=r"$q_1$")
    # solutionPlotter2.plot(t, q2, subplot_index=1, label=r"$\Theta$")  # label=r"$q_2$")
    # solutionPlotter2.plot(t, q3, subplot_index=2, label=r"$\gamma$")  # label=r"$q_3$")
    # solutionPlotter2.plot(t, q4, subplot_index=3, label=r"$\Phi$")  # label=r"$q_4$")
    # solutionPlotter2.plot(t, q5, subplot_index=4, label=r"$\delta$")  # label=r"$q_5$")
    # solutionPlotter2.plot(t, q6, subplot_index=5, label=r"$\Omega$")  # label=r"$q_6$")
    # solutionPlotter2.plot(t, q7, subplot_index=6, label=r"$\varepsilon$")  # label=r"$q_7$")

    # solutionPlotter2.set_legend(subplot_index=None, loc='upper right')
    # solutionPlotter2.set_xlim((0.0, 0.03), subplot_index=None)
    # solutionPlotter2.set_ylim((-2.0, 2.0), subplot_index=0)
    # solutionPlotter2.set_ylim((-2.0, 2.0), subplot_index=1)
    # solutionPlotter2.set_ylim((0.0, 0.4), subplot_index=2)
    # solutionPlotter2.set_ylim((-0.6, 0.2), subplot_index=3)
    # solutionPlotter2.set_ylim((0.4, 0.6), subplot_index=4)
    # solutionPlotter2.set_ylim((-0.2, 0.6), subplot_index=5)
    # solutionPlotter2.set_ylim((1.0, 1.2), subplot_index=6)

    # filename2 = "data" + "/" + f"{problemName}" + "/" + f"solution2.png"
    # solutionPlotter2.save(filename2)


if __name__ == "__main__":
    run('ANDREWS-SQUEEZER')