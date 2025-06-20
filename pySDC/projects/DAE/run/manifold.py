import numpy as np

from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.implementations.hooks.log_solution import LogSolution, LogSolutionAfterIteration
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter


def algebraic_constraints(t, u, problem_name, problem_type):
    r"""
    Returns the value of the algebraic constraints for a specific problem.

    Parameters
    ----------
    t : float
        Time.
    u : mesh or MeshDAE
        Numerical solution.
    problem_name : str
        Name of problem.
    problem_type : str
        Type of problem.

    Returns
    -------
    g : float or np.1darray
        Value of algebraic constraints.
    """

    assert problem_name == "PROTHERO-ROBINSON"

    z = u[1] if problem_type in ["SPP", "SPP-yp"] else u.alg[0]
    g = -z + np.cos(t)
    return g


def finalize_plot(k: int, dt, plotter, num_nodes, problem_name, QI_list, hook_for, solver_type=""):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    k : int
        Case number
    dt : float
        Time step size.
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    num_nodes : int
        Number of collocation nodes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    solver_type : str
        Solver type used to solve system at each node (used for file name).
    """

    xlabel = hook_for if hook_for == "iteration" else "time"
    plotter.set_xlabel(xlabel, subplot_index=None)

    for q, QI in enumerate(QI_list):
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=24)

        plotter.set_ylabel(r"||g(y, z)||", subplot_index=q)

        plotter.set_yscale(scale="log", subplot_index=q)

    plotter.sync_ylim(min_y_set=1e-15)

    plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=6, fontsize=22)

    plotter.adjust_layout(num_subplots=len(QI_list))

    solve = f"_{solver_type}" if solver_type == "direct" else ""
    filename = "data" + "/" + f"{problem_name}" + "/" + f"manifold_{hook_for}_{num_nodes=}_{dt=}_case{k}{solve}.png"
    plotter.save(filename)


"""Main routine"""
if __name__ == "__main__":
    # problem_name = "LINEAR-TEST"
    # problem_name = "MICHAELIS-MENTEN"
    problem_name = "PROTHERO-ROBINSON"

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = 10

    solver_type = ""
    kwargs = {
        "e_tol": -1,
        "maxiter": 80,
        "solver_type": solver_type,
        # "logger_level": 15,
    }

    case = 6

    problems = get_problem_cases(k=case, problem_name=problem_name)

    hook_for = "iteration"  # "step"

    if hook_for == "iteration":
        sortby = "iter"
        hook_class = [LogSolutionAfterIteration]
    elif hook_for == "step":
        sortby = "time"
        hook_class = [LogSolution]

    t0 = 0.0
    dt = 1e-2
    Tend = t0 + dt if hook_for == "iteration" else getEndTime(problem_name)

    manifold_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                print(f"\n{QI} with {num_nodes} nodes: Running test for {problem_type} with {eps=} using {dt=}...\n")

                # Let's do the simulation to get results
                solution_stats = computeSolution(
                    problemName=problem_name,
                    t0=t0,
                    dt=dt,
                    Tend=Tend,
                    nNodes=num_nodes,
                    QI=QI,
                    problemType=problem_type,
                    hookClass=hook_class,
                    eps=eps,
                    **kwargs,
                )

                x = [me[0] for me in get_sorted(solution_stats, type=f"u", sortby=sortby)]
                u = [me[1] for me in get_sorted(solution_stats, type=f"u", sortby=sortby)]

                if hook_for == "iteration":
                    alg_const = [
                        abs(algebraic_constraints(Tend, u_iter, problem_name, problem_type))
                        for u_iter in u
                    ]
                elif hook_for == "step":
                    alg_const = [
                        abs(algebraic_constraints(x_item, u_item, problem_name, problem_type))
                        for x_item, u_item in zip(x, u)
                    ]

                # Define plotting-related stuff and use shortcuts
                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker = res["marker"]
                markersize = res["markersize"]

                # Get things done in plot
                manifold_plotter = plot_result(
                    manifold_plotter, x, alg_const, q, color, None, None, linestyle, problem_label,
                )

    finalize_plot(case, dt, manifold_plotter, num_nodes, problem_name, QI_list, hook_for, solver_type)
