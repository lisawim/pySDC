from mpi4py import MPI
import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.projects.DAE.run.mpi_test import QI_PARALLEL, compute_speedup_and_efficiency, run_serial_test, run_parallel_tests
from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter


def finalize_plot(dt, num_processes_list, problem_name, k):
    mpi_plotter.set_ylabel("speedup", subplot_index=0, fontsize=20)
    mpi_plotter.set_ylabel("efficiency", subplot_index=1, fontsize=20)

    # mpi_plotter.set_ylim((0, 13), subplot_index=0)
    mpi_plotter.set_ylim((0, 1), subplot_index=1)

    mpi_plotter.set_xticks(num_processes_list[::4])
    mpi_plotter.set_xlabel("number of processes", fontsize=20)

    mpi_plotter.set_grid(True, subplot_index=None)

    mpi_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=20)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"mpi_test_semi_implicit_{dt=}_case={k}.png"
    mpi_plotter.save(filename)


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    # problem_name = "LINEAR-TEST"
    problem_name = "MICHAELIS-MENTEN"

    num_processes_list = range(2, global_size + 1)

    QI_ser = "LU"
    QI_PARALLEL_MIN = QI_PARALLEL.copy()
    if "MIN-SR-NS" not in QI_PARALLEL_MIN:
        QI_PARALLEL_MIN.append("MIN-SR-NS")

    t0 = 0.0
    dt = 1e-2

    case = 9

    kwargs = {"e_tol": 1e-13}

    problems = get_problem_cases(k=case)

    results_dict = {} if global_rank == 0 else None

    results_dict, global_rank = run_serial_test(
        dt, global_rank, num_processes_list, problems, problem_name, QI_ser, results_dict, t0, **kwargs
    )

    global_comm.Barrier()

    results_dict = run_parallel_tests(
        dt, global_comm, global_rank, num_processes_list, problems, problem_name, QI_PARALLEL_MIN, results_dict, t0, **kwargs
    )

    if global_rank == 0:
        mpi_plotter = Plotter(nrows=1, ncols=2, figsize=(12, 6))

        speedups, efficiencies = compute_speedup_and_efficiency(num_processes_list, problems, results_dict, QI_PARALLEL_MIN)

        for QI_par in QI_PARALLEL_MIN:
            for problem_type, epsValues in problems.items():
                for i, eps in enumerate(epsValues):
                    if QI_par == "MIN-SR-S":
                        color = getColor(problem_type, i, QI_par)
                    else:
                        if problem_type == "constrainedDAE":
                            color = "mediumslateblue"
                        else:
                            color = "firebrick"

                    res = getMarker(problem_type, i, QI_par)
                    if QI_par == "MIN-SR-S":
                        marker = res["marker"]
                    else:
                        if problem_type == "constrainedDAE":
                            marker = "o"
                        else: marker = "H"

                    markersize = res["markersize"]

                    if QI_par == "MIN-SR-S":
                        linestyle = get_linestyle(problem_type, QI_par)
                    else:
                        if problem_type == "constrainedDAE":
                            linestyle = "dashdot"
                        else:
                            linestyle = "dashed"

                    problem_label = getLabel(problem_type, eps, QI_par)
                    label = f"{QI_par}-{problem_label}"

                    mpi_plotter = plot_result(
                        mpi_plotter,
                        num_processes_list,
                        speedups[QI_par][problem_type][eps],
                        0,
                        color,
                        marker,
                        markersize,
                        linestyle,
                        label,
                        plot_type="plot",
                        markevery=4,
                    )

                    mpi_plotter = plot_result(
                        mpi_plotter,
                        num_processes_list,
                        efficiencies[QI_par][problem_type][eps],
                        1,
                        color,
                        marker,
                        markersize,
                        linestyle,
                        label,
                        plot_type="plot",
                        markevery=4,
                    )

        finalize_plot(dt, num_processes_list, problem_name, case)
