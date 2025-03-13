import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter
from pySDC.projects.DAE.run.error import get_problem_cases, get_error_label, plot_result

from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep


def finalize_plot(k: int, plotter, num_nodes, problem_name, QI_list):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    k : int
        Case number
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    num_nodes : int
        Number of collocation nodes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    """

    plotter.set_xlabel("time step sizes", subplot_index=None)

    err_label = get_error_label(problem_name)

    for q, QI in enumerate(QI_list):
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=24)

        plotter.set_ylabel(err_label, subplot_index=q)

        plotter.set_xscale(scale="log", subplot_index=q)

        plotter.set_ylim((1e-15, 1e1), scale="log", subplot_index=q)

    plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=6, fontsize=22)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"order_{num_nodes=}_case{k}.png"
    plotter.save(filename)


"""Main routine"""
if __name__ == "__main__":
    problem_name = "LINEAR-TEST"
    # problem_name = "MICHAELIS-MENTEN"

    QI_list = ["IE", "LU", "MIN-SR-S", "MIN-SR-FLEX"]
    num_nodes = 3

    p = 2 * num_nodes - 1

    t0 = 0.0
    dt_list = np.logspace(-1.0, 0.0, num=11)

    case = 4

    problems = get_problem_cases(k=case)

    hook_class = [LogGlobalErrorPostStep]

    order_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                errors = []

                for dt in dt_list:
                    print(f"\n{QI} with {num_nodes} nodes: Running test for {problem_type} with {eps=} using {dt=}...\n")

                    # Let's do the simulation to get results
                    solution_stats = computeSolution(
                        problemName=problem_name,
                        t0=t0,
                        dt=dt,
                        Tend=getEndTime(problem_name),
                        nNodes=num_nodes,
                        QI=QI,
                        problemType=problem_type,
                        hookClass=hook_class,
                        eps=eps,
                        e_tol=1e-7,
                    )

                    err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]

                    errors.append(max(err_values))

                # Define plotting-related stuff and use shortcuts
                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker = res["marker"]
                markersize = res["markersize"]

                order_plotter = plot_result(
                    order_plotter, dt_list, errors, q, color, marker, markersize, linestyle, problem_label
                )

        # Reference order
        order_plotter = plot_result(
            order_plotter, dt_list, [dt ** p for dt in dt_list], q, "lightgrey", None, None, "dotted", ""
        )

    finalize_plot(case, order_plotter, num_nodes, problem_name, QI_list)
