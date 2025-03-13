from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate, LogEmbeddedErrorEstimatePostIter

from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter


def finalize_plot(k: int, dt, plotter, num_nodes, problem_name, QI_list, hook_for, solver_type):
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
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17}

    xlabel = "iteration" if hook_for == "_post_iteration" else "time"
    plotter.set_xlabel(xlabel, subplot_index=None)

    for q, QI in enumerate(QI_list):
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=24)

        plotter.set_ylabel("increment", subplot_index=q)

        plotter.set_ylim((1e-15, 1e5), scale="log", subplot_index=q)
        # plotter.set_yscale(scale="log", subplot_index=q)

    bbox_pos = bbox_position[k]
    plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)

    plotter.adjust_layout(num_subplots=len(QI_list))

    increment_for = "iteration" if hook_for == "_post_iteration" else "run"
    solve = f"_{solver_type}" if solver_type == "direct" else ""
    filename = "data" + "/" + f"{problem_name}" + "/" + f"estimate_embedded_error_{increment_for}_{num_nodes=}_{dt=}_case{k}{solve}.png"
    plotter.save(filename)


"""Main routine"""
if __name__ == "__main__":
    # problem_name = "LINEAR-TEST"
    problem_name = "MICHAELIS-MENTEN"

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = 14

    solver_type = "newton"
    kwargs = {
        "e_tol": -1,
        "maxiter": 200,
        "solver_type": solver_type,
    }

    t0 = 0.0
    dt = 1e-2#np.logspace(-2.5, 0.0, num=11)[0]

    case = 6

    problems = get_problem_cases(k=case)

    hook_for = "_post_iteration"  # ""
    sortby = "iter" if hook_for == "_post_iteration" else "time"
    hook_class = [LogEmbeddedErrorEstimatePostIter] if hook_for == "_post_iteration" else [LogEmbeddedErrorEstimate]

    err_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                print(f"\n{QI}: Running test for {problem_type} with {eps=}...\n")

                # Let's do the simulation to get results
                solution_stats = computeSolution(
                    problemName=problem_name,
                    t0=t0,
                    dt=dt,
                    Tend=t0 + dt if hook_for == "_post_iteration" else getEndTime(problem_name),
                    nNodes=num_nodes,
                    QI=QI,
                    problemType=problem_type,
                    hookClass=hook_class,
                    eps=eps,
                    **kwargs,
                )

                # Get error values along iterations
                x = [me[0] for me in get_sorted(solution_stats, type=f"error_embedded_estimate{hook_for}", sortby=sortby)]
                embedded_err_values = [me[1] for me in get_sorted(solution_stats, type=f"error_embedded_estimate{hook_for}", sortby=sortby)]

                # Define plotting-related stuff and use shortcuts
                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker = res["marker"]
                markersize = res["markersize"]

                # Get things done in plot
                err_plotter = plot_result(
                    err_plotter, x, embedded_err_values, q, color, marker, markersize, linestyle, problem_label, markevery=100
                )

    finalize_plot(case, dt, err_plotter, num_nodes, problem_name, QI_list, hook_for, solver_type)
