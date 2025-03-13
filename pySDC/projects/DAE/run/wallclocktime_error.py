from mpi4py import MPI
import numpy as np

from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep

from pySDC.projects.DAE.run.error import get_error_label, get_problem_cases, plot_result
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter


def compute_speedup_factors(serial_times, parallel_times):
    """Compute smallest and largest speedup factors."""
    speedup_factors = np.array(serial_times) / np.array(parallel_times)
    return np.min(speedup_factors), np.max(speedup_factors)

def choose_time_step_sizes(problem_name):
    """Choose specific time step sizes depending on problem."""
    if problem_name == "LINEAR-TEST":
        dt_list = np.logspace(-2.5, 0.0, num=11)
    elif problem_name == "MICHAELIS-MENTEN":
        dt_list = [10 ** (-m) for m in range(2, 8)]
    else:
        raise NotImplementedError()
    return dt_list

def run_parallel_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs):
    err, time = [], []

    for dt in dt_list:

        comm.Barrier()

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
            comm=comm,
            useMPI=True,
            **kwargs,
        )

        comm.Barrier()

        err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]
        err.append(max(err_values))

        if rank == 0:
            print(f"For {dt=} error is {max(err_values)}")

        timing_run = get_sorted(solution_stats, type="timing_run")[0][1]
        timing_run_full = comm.reduce(timing_run, op=MPI.MAX)
        time.append(timing_run_full)

    return err, time
    

def run_serial_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs):
    err, time = [], []
    
    for dt in dt_list:
        if rank == 0:

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
                comm=comm,
                useMPI=False,
                **kwargs,
            )

            err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]
            err.append(max(err_values))

            timing_run = np.array(get_sorted(solution_stats, type="timing_run", sortby="time"))
            time.append(timing_run[0][1])

            print(f"For {dt=} error is {max(err_values)}")

        comm.Barrier()  # Ensure synchronization between ranks

    return err, time


def finalize_plot(k, num_nodes, problem_name, QI_list, work_plotter, solver_type):
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
    work_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17}

    err_label = get_error_label(problem_name)

    for q, QI in enumerate(QI_list):
        work_plotter.set_xlabel("wall-clock time", subplot_index=q)

        work_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

        # work_plotter.set_xlim((8e-3, 2e1), subplot_index=q)
        work_plotter.set_xscale(scale="log", subplot_index=q)

        # work_plotter.set_ylim((1e-14, 1e0), subplot_index=q)
        work_plotter.set_yscale(scale="log", subplot_index=q)

    work_plotter.sync_xlim(min_x_set=1e-2)
    work_plotter.sync_ylim(min_y_set=1e-15)

    work_plotter.set_ylabel(f"{err_label}")

    work_plotter.set_grid()

    work_plotter.adjust_layout(num_subplots=len(QI_list))

    bbox_pos = bbox_position[k]
    work_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)
    print("save")
    filename = "data" + "/" + f"{problem_name}" + "/" + f"wallclocktime_error_case{k}_{num_nodes=}_{solver_type}.png"
    work_plotter.save(filename)


def compute_work_vs_error(case, comm, num_nodes, rank, problem_name, QI_list, do_plotting=True, dt_list=None):
    t0 = 0.0

    if dt_list is None:
        dt_list = choose_time_step_sizes(problem_name)

    print(dt_list)

    solver_type = "newton"
    kwargs = {"e_tol": 1e-13, "solver_type": solver_type}

    problems = get_problem_cases(k=case, problem_name=problem_name)

    hook_class = [LogGlobalErrorPostStep]

    results = []

    for problem_type, epsValues in problems.items():
        for i, eps in enumerate(epsValues):
            serial_times = None
            parallel_times_min_sr_s = None
            parallel_times_min_sr_flex = None

            for q, QI in enumerate(QI_list):

                if QI == "LU":
                    err, time = run_serial_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs)
                    serial_times = time
                elif QI == "MIN-SR-S":
                    err, time = run_parallel_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs)
                    parallel_times_min_sr_s = time
                elif QI == "MIN-SR-FLEX":
                    err, time = run_parallel_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs)
                    parallel_times_min_sr_flex = time
                else:
                    err, time = run_serial_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs)

                if rank == 0:
                    results.append((q, time, err, problem_type, i, QI, eps))
                    print(f"{QI}: For {problem_type} with {eps=} the runtimes are: {time}\n")

                if rank == 0 and serial_times and parallel_times_min_sr_s:
                    smallest_speedup_min_sr_s, largest_speedup_min_sr_s = compute_speedup_factors(serial_times, parallel_times_min_sr_s)
                    print(f"MIN-SR-S: For {problem_type} with {eps=}: Smallest speedup {smallest_speedup_min_sr_s:.2f}, Largest speedup {largest_speedup_min_sr_s:.2f}")

                if rank == 0 and serial_times and parallel_times_min_sr_flex:
                    smallest_speedup_min_sr_flex, largest_speedup_min_sr_flex = compute_speedup_factors(serial_times, parallel_times_min_sr_flex)
                    print(f"MIN-SR-FLEX: For {problem_type} with {eps=}: Smallest speedup {smallest_speedup_min_sr_flex:.2f}, Largest speedup {largest_speedup_min_sr_flex:.2f}")

                comm.Barrier()

    if rank == 0 and do_plotting:
        work_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

        for q, time, err, problem_type, i, QI, eps in results:
            color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
            problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
            marker, markersize = res["marker"], res["markersize"]
            plot_result(work_plotter, time, err, q, color, marker, markersize, linestyle, problem_label)

        finalize_plot(case, num_nodes, problem_name, QI_list, work_plotter, solver_type)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    problem_name = "MICHAELIS-MENTEN"

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = size

    case = 6

    compute_work_vs_error(case, comm, num_nodes, rank, problem_name, QI_list)

    MPI.Finalize()
