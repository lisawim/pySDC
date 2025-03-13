import numpy as np
from mpi4py import MPI

from pySDC.implementations.hooks.log_work import LogWork

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.projects.DAE.run.mpi_test import QI_SERIAL, QI_PARALLEL
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter

def finalize_plot(k, num_nodes, problem_name, QI_list, iter_plotter):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    dt : float
        Time step size.
    k : int
        Case number.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    iter_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17}

    for q, QI in enumerate(QI_list):
        iter_plotter.set_xlabel("time step sizes", subplot_index=q)

        iter_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

        iter_plotter.set_yscale(scale="log")

    iter_plotter.set_ylabel("number of Newton iterations", subplot_index=None)

    iter_plotter.sync_ylim(min_y_set=1e0)

    iter_plotter.set_grid()

    iter_plotter.adjust_layout(num_subplots=len(QI_list))

    bbox_pos = bbox_position[k]
    iter_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"newton_iterations_along_step_sizes_{num_nodes=}_case{k}.png"
    iter_plotter.save(filename)


def newton_iterations_along_step_sizes():
    QI_list = QI_SERIAL + QI_PARALLEL
    num_nodes = 3

    # problem_name = "LINEAR-TEST"
    problem_name = "MICHAELIS-MENTEN"

    solver_type = "newton"
    kwargs = {"solver_type": solver_type}

    t0 = 0.0
    dt_list = [10 ** (-m) for m in range(2, 8)]

    case = 6

    problems = get_problem_cases(k=case, problem_name=problem_name)

    hook_class = [LogWork]

    iter_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                newton_iters = []

                for dt in dt_list:
                    print(f"\n{QI} with {num_nodes} nodes: Running test for {problem_type} with {eps=} using {dt=}...")

                    solution_stats = computeSolution(
                        problemName=problem_name,
                        t0=t0,
                        dt=dt,
                        Tend=getEndTime(problem_name),
                        nNodes=num_nodes,
                        QI=QI,
                        problemType=problem_type,
                        eps=eps,
                        hookClass=hook_class,
                        **kwargs,
                    )

                    newton_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="work_newton", sortby="time")])
                    newton_iters.append(newton_iter)

                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker = res["marker"]
                markersize = res["markersize"]

                iter_plotter = plot_result(
                    iter_plotter,
                    dt_list,
                    newton_iters,
                    q,
                    color,
                    marker,
                    markersize,
                    linestyle,
                    problem_label,
                    plot_type="semilogy",
                )

    finalize_plot(case, num_nodes, problem_name, QI_list, iter_plotter)


if __name__ == "__main__":
    newton_iterations_along_step_sizes()
