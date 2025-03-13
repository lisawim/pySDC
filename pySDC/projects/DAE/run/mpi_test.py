from mpi4py import MPI
import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter

QI_SERIAL = ["IE", "LU"]
QI_PARALLEL = ["MIN-SR-S"]

def run_test_and_split_communicator(
        num_processes, global_comm, global_rank, problemName, t0, dt, Tend, QI, problemType, useMPI, eps, hookClass=[], **kwargs
):
    r"""
    In this function the speed-up test is done. Here, the communicator is then splitted. Number of collocation nodes
    is adapted as well.

    Parameters
    ----------
    num_processes : int
        Number of processes.
    global_comm : MPI.COMM_WORLD
        Global communicator to be split.
    global_rank : MPI.COMM_WORLD
        Current rank that passes this function

    Returns
    -------
    """

    if global_rank < num_processes:
        # Split the communicator to create a new communicator for this test
        sub_comm = global_comm.Split(color=1, key=global_rank)
        sub_rank = sub_comm.Get_rank()

        sub_nNodes = sub_comm.Get_size()

        # Perform the computation with the sub-communicator
        solutionStats = computeSolution(
            problemName=problemName,
            t0=t0,
            dt=dt,
            Tend=Tend,
            nNodes=sub_nNodes,
            QI=QI,
            problemType=problemType,
            useMPI=useMPI,
            eps=eps,
            comm=sub_comm,
            hookClass=hookClass,
            **kwargs,
        )

        timingRun = get_sorted(solutionStats, type="timing_run")[0][1]
        timingRunFull = sub_comm.reduce(timingRun, op=MPI.MAX, root=0)

        num_iter = np.sum([me[1] for me in get_sorted(solutionStats, type="niter", sortby="time")])

        sub_comm.Free()

        # Only the root of sub_comm returns the collected data
        if sub_rank == 0:
            return {
                "timingRun": timingRunFull,
                "num_iter": num_iter,
            }
        else:
            return None

    else:
        # Split the communicator to exclude this process
        global_comm.Split(color=MPI.UNDEFINED, key=global_rank)
        return None

def run_serial_test(dt, global_rank, num_processes_list, problems, problem_name, QI_ser, results_dict, t0, **kwargs):
    for num_tasks in num_processes_list:
        # Number of processes/tasks is number of collocation nodes
        num_nodes = num_tasks

        if global_rank == 0:
            results_dict[num_tasks] = {} if global_rank == 0 else None

            for problemType, epsValues in problems.items():
                results_dict[num_tasks][problemType] = {}

                for eps in epsValues:
                    results_dict[num_tasks][problemType][eps] = {"serial": 0, "parallel": 0, "num_iter": 0}

            for problemType, epsValues in problems.items():
                for eps in epsValues:
                    print(f"\n{QI_ser}: Running serial test for {problemType} with {eps=} using {num_nodes} nodes...")

                    solution_stats = computeSolution(
                        problemName=problem_name,
                        t0=t0,
                        dt=dt,
                        Tend=getEndTime(problem_name),
                        nNodes=num_nodes,
                        QI=QI_ser,
                        problemType=problemType,
                        useMPI=False,
                        eps=eps,
                        **kwargs,
                    )

                    timing_run = np.array(get_sorted(solution_stats, type="timing_run", sortby="time"))
                    results_dict[num_tasks][problemType][eps]["serial"] += timing_run[0][1]
                    print(f"Runtime: {results_dict[num_tasks][problemType][eps]['serial']}\n")

                    num_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="niter", sortby="time")])
                    results_dict[num_tasks][problemType][eps]["num_iter"] = num_iter

    return results_dict, global_rank

def run_parallel_tests(dt, global_comm, global_rank, num_processes_list, problems, problem_name, QI_PARALLEL, results_dict, t0, **kwargs):
    for QI_par in QI_PARALLEL:
        for num_processes in num_processes_list:

            for problemType, epsValues in problems.items():
                for eps in epsValues:
                    if global_rank == 0:
                        print(f"\n{QI_par}: Running parallel test for {problemType} with {eps=} using {num_processes} nodes...")

                    global_comm.Barrier()  # Ensure all processes reach this point before continuing

                    result = run_test_and_split_communicator(
                        num_processes=num_processes,
                        global_comm=global_comm,
                        global_rank=global_rank,
                        problemName=problem_name,
                        t0=t0,
                        dt=dt,
                        Tend=getEndTime(problem_name),
                        QI=QI_par,
                        problemType=problemType,
                        useMPI=True,
                        eps=eps,
                        **kwargs,
                    )

                    global_comm.Barrier()  # Ensure all processes finish this test before continuing

                    if global_rank == 0 and result is not None:
                        timing_run = result["timingRun"]
                        num_iter = result["num_iter"]

                        if QI_par not in results_dict[num_processes][problemType][eps]:
                            results_dict[num_processes][problemType][eps][QI_par] = {"runtime": 0, "num_iter": 0}

                        results_dict[num_processes][problemType][eps][QI_par]["runtime"] += timing_run
                        results_dict[num_processes][problemType][eps][QI_par]["num_iter"] = num_iter

                        runtime = results_dict[num_processes][problemType][eps][QI_par]["runtime"]
                        num_iter = results_dict[num_processes][problemType][eps][QI_par]["num_iter"]
                        print(f"Runtime ({QI_par}): {runtime} -- Number of iterations: {num_iter}\n")

    return results_dict

def compute_speedup_and_efficiency(num_processes_list, problems, results_dict, QI_PARALLEL):
    speedups = {}
    efficiencies = {}

    # Prepare data for plotting
    for QI_par in QI_PARALLEL:
        speedups[QI_par] = {problemType: {eps: [] for eps in epsValues} for problemType, epsValues in problems.items()}
        efficiencies[QI_par] = {problemType: {eps: [] for eps in epsValues} for problemType, epsValues in problems.items()}

    for num_processes in num_processes_list:
        for problemType, epsValues in problems.items():
            for eps in epsValues:
                for QI_par in QI_PARALLEL:
                    if QI_par in results_dict[num_processes][problemType][eps]:
                        timings = results_dict[num_processes][problemType][eps]
                        speedup = timings["serial"] / timings[QI_par]["runtime"]
                        speedups[QI_par][problemType][eps].append(speedup)

                        efficiency = speedup / num_processes
                        efficiencies[QI_par][problemType][eps].append(efficiency)

                        print(f"{num_processes} nodes for {problemType} with {eps} ({QI_par}): Speedup is {speedup} -- efficiency is {efficiency}\n")

    return speedups, efficiencies

def check_num_processes(problem_name, global_comm, global_rank, global_size):
    if problem_name == "LINEAR-TEST":
        if not (2 <= global_size <= 20):
            if global_rank == 0:
                raise ProblemError("This test requires between 2 and 20 processes!")
            global_comm.Abort(1)  # Ensures all processes exit

    elif problem_name == "MICHAELIS-MENTEN":
        if not (2 <= global_size <= 14):
            if global_rank == 0:
                raise ProblemError("This test requires between 2 and 20 processes!")
            global_comm.Abort(1)  # Ensures all processes exit

def finalize_plot(dt, num_processes_list, problem_name, k):
    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.31}

    for QI_par in QI_PARALLEL:
        if len(QI_PARALLEL) == 2:
            q = 0 if QI_par == "MIN-SR-S" else 2

            mpi_plotter.set_title(rf"$Q_\Delta=${QI_par}", subplot_index=q, fontsize=20)
            mpi_plotter.set_title(rf"$Q_\Delta=${QI_par}", subplot_index=q + 1, fontsize=20)
        else:
            q = 0

        mpi_plotter.set_ylabel("speedup", subplot_index=q, fontsize=20)
        mpi_plotter.set_ylabel("efficiency", subplot_index=q + 1, fontsize=20)

        mpi_plotter.set_ylim((0, 10), subplot_index=q)
        mpi_plotter.set_ylim((0, 1), subplot_index=q + 1)

        mpi_plotter.set_yscale(subplot_index=0)

    mpi_plotter.set_xticks(num_processes_list[::4])
    mpi_plotter.set_xlabel("number of processes", fontsize=20)

    mpi_plotter.set_grid(True, subplot_index=None)

    bbox_pos = bbox_position[k]
    mpi_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=20)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"mpi_test_{dt=}_case={k}.png"
    mpi_plotter.save(filename)


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    # problem_name = "LINEAR-TEST"
    problem_name = "MICHAELIS-MENTEN"

    check_num_processes(problem_name, global_comm, global_rank, global_size)

    num_processes_list = range(2, global_size + 1)

    QI_ser = "LU"

    t0 = 0.0
    dt = 1e-2

    case = 6

    kwargs = {"e_tol": 1e-13}

    problems = get_problem_cases(k=case)

    results_dict = {} if global_rank == 0 else None

    results_dict, global_rank = run_serial_test(
        dt, global_rank, num_processes_list, problems, problem_name, QI_ser, results_dict, t0, **kwargs
    )

    global_comm.Barrier()

    results_dict = run_parallel_tests(
        dt, global_comm, global_rank, num_processes_list, problems, problem_name, QI_PARALLEL, results_dict, t0, **kwargs
    )

    if global_rank == 0:
        nrows = len(QI_PARALLEL)
        figsize_y = 12 if len(QI_PARALLEL) == 2 else 6
        mpi_plotter = Plotter(nrows=nrows, ncols=2, figsize=(12, figsize_y))

        speedups, efficiencies = compute_speedup_and_efficiency(num_processes_list, problems, results_dict, QI_PARALLEL)

        for QI_par in QI_PARALLEL:
            subplot_index = 0 if QI_par == "MIN-SR-S" else 2
            for problem_type, epsValues in problems.items():
                for i, eps in enumerate(epsValues):
                    color, res = getColor(problem_type, i, QI_par), getMarker(problem_type, i, QI_par)
                    problem_label, linestyle = getLabel(problem_type, eps, QI_par), get_linestyle(problem_type, QI_par)
                    marker = res["marker"]
                    markersize = res["markersize"]

                    mpi_plotter = plot_result(
                        mpi_plotter,
                        num_processes_list,
                        speedups[QI_par][problem_type][eps],
                        subplot_index,
                        color,
                        marker,
                        markersize,
                        linestyle,
                        problem_label,
                        plot_type="plot",
                        markevery=4,
                    )

                    mpi_plotter = plot_result(
                        mpi_plotter,
                        num_processes_list,
                        efficiencies[QI_par][problem_type][eps],
                        subplot_index + 1,
                        color,
                        marker,
                        markersize,
                        linestyle,
                        problem_label,
                        plot_type="plot",
                        markevery=4,
                    )

        finalize_plot(dt, num_processes_list, problem_name, case)

