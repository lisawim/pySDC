import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.projects.DAE.run.mpi_test import QI_SERIAL, QI_PARALLEL, run_serial_test, run_parallel_tests
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter


def finalize_plot(dt, k, problem_name, QI_list, iter_plotter, num_nodes_list=None, appendix="along_nodes"):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    dt : float
        Time step size.
    k : int
        Case number.
    num_nodes_list : list
        List contains different number of collocation nodes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    iter_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17}

    for q, QI in enumerate(QI_list):
        if appendix == "along_nodes":
            iter_plotter.set_xticks(num_nodes_list[::4], subplot_index=q)
            iter_plotter.set_xlabel("number of nodes", subplot_index=q)
        elif appendix == "along_step_sizes":
            iter_plotter.set_xlabel("time step sizes", subplot_index=q)

        iter_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

        iter_plotter.set_yscale(scale="log")

    iter_plotter.set_ylabel("number of iterations", subplot_index=None)

    # if problem_name == "LINEAR-TEST":
    #     iter_plotter.set_ylim((5e2, 5e3), scale="log")
    # elif problem_name == "MICHAELIS-MENTEN":
    #     iter_plotter.set_ylim((5e2, 1.4e5), scale="log")

    iter_plotter.sync_ylim(min_y_set=1e0)

    iter_plotter.set_grid()

    iter_plotter.adjust_layout(num_subplots=len(QI_list))

    bbox_pos = bbox_position[k]
    iter_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"iterations_{appendix}_{dt=}_case{k}.png"
    iter_plotter.save(filename)


def iterations_along_nodes():
    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    QI_list = QI_SERIAL + QI_PARALLEL
    num_processes_list = range(2, global_size + 1)

    # problem_name = "LINEAR-TEST"
    problem_name = "MICHAELIS-MENTEN"

    solver_type = "direct"
    kwargs = {"solver_type": solver_type}

    t0 = 0.0
    dt = 1e-4#np.logspace(-2.5, 0.0, num=11)[0]

    case = 6

    problems = get_problem_cases(k=case, problem_name=problem_name)

    results_dict = {} if global_rank == 0 else None

    # Plot serial SDC schemes
    if global_rank == 0:
        iter_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

        for q, QI_ser in enumerate(QI_SERIAL):
            results_dict, global_rank = run_serial_test(dt, global_rank, num_processes_list, problems, problem_name, QI_ser, results_dict, t0, **kwargs)

            for problem_type, eps_values in problems.items():
                for i, eps in enumerate(eps_values):
                    color, res = getColor(problem_type, i, QI_ser), getMarker(problem_type, i, QI_ser)
                    problem_label, linestyle = getLabel(problem_type, eps, QI_ser), get_linestyle(problem_type, QI_ser)
                    marker = res["marker"]
                    markersize = res["markersize"]

                    num_iters = [results_dict[num_processes][problem_type][eps]["num_iter"] for num_processes in num_processes_list]

                    iter_plotter = plot_result(
                        iter_plotter,
                        num_processes_list,
                        num_iters,
                        q,
                        color,
                        marker,
                        markersize,
                        linestyle,
                        problem_label,
                        plot_type="semilogy",
                        markevery=4,
                    )

    global_comm.Barrier()

    results_dict = run_parallel_tests(dt, global_comm, global_rank, num_processes_list, problems, problem_name, QI_PARALLEL, results_dict, t0, **kwargs)

    if global_rank == 0:
        for QI_par in QI_PARALLEL:
            for problem_type, eps_values in problems.items():
                for i, eps in enumerate(eps_values):
                    q = 2 if QI_par == "MIN-SR-S" else 3
                    color, res = getColor(problem_type, i, QI_ser), getMarker(problem_type, i, QI_ser)
                    problem_label, linestyle = getLabel(problem_type, eps, QI_ser), get_linestyle(problem_type, QI_ser)
                    marker = res["marker"]
                    markersize = res["markersize"]

                    num_iters = [results_dict[num_processes][problem_type][eps][QI_par]["num_iter"] for num_processes in num_processes_list]

                    iter_plotter = plot_result(
                        iter_plotter,
                        num_processes_list,
                        num_iters,
                        q,
                        color,
                        marker,
                        markersize,
                        linestyle,
                        problem_label,
                        plot_type="semilogy",
                        markevery=4,
                    )

        finalize_plot(dt, case, problem_name, QI_list, iter_plotter, num_nodes_list=num_processes_list)

    MPI.Finalize()


def iterations_along_step_sizes():
    QI_list = QI_PARALLEL  # QI_SERIAL + QI_PARALLEL
    num_nodes = 3

    # problem_name = "LINEAR-TEST"
    problem_name = "MICHAELIS-MENTEN"

    solver_type = "direct"
    kwargs = {"solver_type": solver_type}

    t0 = 0.0
    dt_list = [1e-2]  # [10 ** (-m) for m in range(2, 8)]

    case = 4  # 6

    problems = get_problem_cases(k=case, problem_name=problem_name)

    iter_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                num_iters = []

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
                        **kwargs,
                    )

                    num_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="niter", sortby="time")])
                    num_iters.append(num_iter)

                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker = res["marker"]
                markersize = res["markersize"]

                iter_plotter = plot_result(
                    iter_plotter,
                    dt_list,
                    num_iters,
                    q,
                    color,
                    marker,
                    markersize,
                    linestyle,
                    problem_label,
                    plot_type="semilogy",
                )

    # finalize_plot(dt, case, problem_name, QI_list, iter_plotter, appendix="along_step_sizes")


if __name__ == "__main__":
    # iterations_along_nodes()
    iterations_along_step_sizes()    
