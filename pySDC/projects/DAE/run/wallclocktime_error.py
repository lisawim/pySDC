from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_work import LogWork
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep, LogLocalErrorPostStep
from pySDC.projects.DAE.misc.hooksDAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable
from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep as LogGlobalErrorFirstVariable
from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStepPerturbation

from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter
from pySDC.projects.DAE.run.error import get_error_label, get_problem_cases, plot_result
from pySDC.helpers.stats_helper import get_sorted


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

def run_parallel_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, error_type="global", log_time=True, **kwargs):
    err, work = [], []
    err_diff, err_alg = [], []

    for dt in dt_list:
        if rank == 0:
            print(f"\n{QI}: Running test for {problem_type} with {eps=} using {dt=}...\n")

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

        err_diff_values = [me[1] for me in get_sorted(solution_stats, type="e_global_differential_post_step", sortby="time")]
        err_alg_values = [me[1] for me in get_sorted(solution_stats, type="e_global_algebraic_post_step", sortby="time")]
        if len(err_diff_values) > 0 and len(err_alg_values) > 0:
            err_diff.append(max(err_diff_values))
            err_alg.append(max(err_alg_values))

        err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_{error_type}_post_step", sortby="time")]
        if len(err_values) > 0:
            err.append(max(err_values))

        if rank == 0:
            plt.figure()
            plt.plot([me[0] for me in get_sorted(solution_stats, type="e_global_differential_post_step", sortby="time")], err_diff_values, label="y err")
            plt.plot([me[0] for me in get_sorted(solution_stats, type="e_global_differential_post_step", sortby="time")], err_alg_values, label="z err")
            plt.yscale("log", base=10)
            plt.ylim((1e-15, 1e-9))
            plt.legend(loc="best")
            plt.show()

            num_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="niter", sortby="time")])
            print(f"Number of iterations: {num_iter}")

            newton_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="work_newton", sortby="time")])
            print(f"Number of Newton iterations: {newton_iter}")

        if log_time:
            timing_run = get_sorted(solution_stats, type="timing_run")[0][1]
            timing_step = get_sorted(solution_stats, type="timing_step")
            timing_run_full = comm.reduce(timing_run, op=MPI.MAX)
            work.append(timing_run_full)
        else:
            newton_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="work_newton", sortby="time")])
            newton_iter_full = comm.reduce(newton_iter, op=MPI.MAX)
            work.append(newton_iter_full)

    return err, err_diff, err_alg, work


def run_serial_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, error_type="global", log_time=True, **kwargs):
    err, work = [], []
    err_diff, err_alg = [], []

    for dt in dt_list:
        if rank == 0:
            print(f"\n{QI}: Running test for {problem_type} with {eps=} using {dt=}...\n")
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

            err_diff_values = [me[1] for me in get_sorted(solution_stats, type="e_global_differential_post_step", sortby="time")]
            err_alg_values = [me[1] for me in get_sorted(solution_stats, type="e_global_algebraic_post_step", sortby="time")]

            print(f"{err_diff_values=}")
            print(max(err_alg_values))

            u_val = get_sorted(solution_stats, type="u", sortby="time")

            t = np.array([0.0] + [me[0] for me in u_val])

            y = np.array([me[1].diff[0] for me in u_val])
            z = np.array([me[1].alg[0] for me in u_val])

            plt.figure()
            plt.plot(t, y, label="y")
            plt.plot(t, z, label="z")
            # plt.yscale("log", base=10)
            # plt.ylim((1e-15, 1e-9))
            plt.legend(loc="best")
            plt.show()

            plt.figure()
            plt.plot([me[0] for me in get_sorted(solution_stats, type="e_global_differential_post_step", sortby="time")], err_diff_values, label="y err")
            plt.plot([me[0] for me in get_sorted(solution_stats, type="e_global_differential_post_step", sortby="time")], err_alg_values, label="z err")
            plt.yscale("log", base=10)
            plt.ylim((1e-15, 1e-9))
            plt.legend(loc="best")
            plt.show()

            if len(err_diff_values) > 0 and len(err_alg_values) > 0:
                err_diff.append(max(err_diff_values))
                err_alg.append(max(err_alg_values))

            err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_{error_type}_post_step", sortby="time")]
            if len(err_values) > 0:
                err.append(max(err_values))

            newton_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="work_newton", sortby="time")])
            print(f"Number of Newton iterations: {newton_iter}")

            if log_time:
                timing_run = np.array(get_sorted(solution_stats, type="timing_run", sortby="time"))
                work.append(timing_run[0][1])
            else:
                work.append(newton_iter)

            num_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="niter", sortby="time")])
            print(f"Number of iterations: {num_iter}")

        comm.Barrier()  # Ensure synchronization between ranks

    return err, err_diff, err_alg, work


def finalize_plot(k, num_nodes, problem_name, QI_list, work_plotter, solver_type, error_type="global", label_error="", log_time=True, separate_errors=False):
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
        if log_time:
            work_plotter.set_xlabel("wall-clock time", subplot_index=q)
        else:
            work_plotter.set_xlabel("number of Newton iterations", subplot_index=q)

        work_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

        # work_plotter.set_xlim((8e-3, 2e1), subplot_index=q)
        work_plotter.set_xscale(scale="log", subplot_index=q)

        # work_plotter.set_ylim((1e-14, 1e0), subplot_index=q)
        work_plotter.set_yscale(scale="log", subplot_index=q)

    if log_time:
        work_plotter.sync_xlim(min_x_set=1e-3)
    else:
        work_plotter.sync_xlim(min_x_set=1e1)

    work_plotter.sync_ylim(min_y_set=1e-16)

    if not separate_errors:
        work_plotter.set_ylabel(f"{error_type} error")
    else:
        if label_error == "_diff":
            work_plotter.set_ylabel(f"{error_type} error in y")
        elif label_error == "_alg":
            work_plotter.set_ylabel(f"{error_type} error in z")

    work_plotter.set_grid()

    work_plotter.adjust_layout(num_subplots=len(QI_list))

    bbox_pos = bbox_position[k]
    work_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)
    print("save")
    plot_type = "wallclocktime_error" if log_time else "work_error"
    filename = "data" + "/" + f"{problem_name}" + "/" + f"{plot_type}_case{k}_{num_nodes=}_{solver_type}{label_error}.png"
    work_plotter.save(filename)


def compute_work_vs_error(case, comm, num_nodes, rank, problem_name, QI_list, do_plotting=True, dt_list=None, error_type="global", separate_errors=False, log_time=True):
    t0 = 0.0

    if dt_list is None:
        dt_list = choose_time_step_sizes(problem_name)

    solver_type = "newton"
    kwargs = {
        "e_tol": 1e-13,
        "solver_type": solver_type,
    }

    problems = get_problem_cases(k=case, problem_name=problem_name)

    hook_class = [LogWork, LogSolution]

    results = []
    results_diff, results_alg = [], []

    for problem_type, epsValues in problems.items():
        for i, eps in enumerate(epsValues):
            # Adding hook classes for error(s)
            if error_type == "global":
                if separate_errors:
                    if eps == 0.0:
                        hook_class += [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable]
                    else:
                        hook_class += [LogGlobalErrorFirstVariable, LogGlobalErrorPostStepPerturbation]
                else:
                    hook_class += [LogGlobalErrorPostStep]

            elif error_type == "local":
                hook_class += [LogLocalErrorPostStep]

            serial_times = None
            parallel_times_min_sr_s = None
            parallel_times_min_sr_flex = None

            for q, QI in enumerate(QI_list):

                if QI == "LU":
                    err, err_diff, err_alg, work = run_serial_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, error_type=error_type, log_time=log_time, **kwargs)
                    # serial_times = time
                elif QI == "MIN-SR-S":
                    err, err_diff, err_alg, work = run_parallel_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, error_type=error_type, log_time=log_time, **kwargs)
                    # parallel_times_min_sr_s = time
                elif QI == "MIN-SR-FLEX":
                    err, err_diff, err_alg, work = run_parallel_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, error_type=error_type, log_time=log_time, **kwargs)
                    # parallel_times_min_sr_flex = time
                else:
                    err, err_diff, err_alg, work = run_serial_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, error_type=error_type, log_time=log_time, **kwargs)

                if rank == 0:
                    results.append((q, work, err, problem_type, i, QI, eps))

                    results_diff.append((q, work, err_diff, problem_type, i, QI, eps))

                    results_alg.append((q, work, err_alg, problem_type, i, QI, eps))
                    # print(f"{QI}: For {problem_type} with {eps=} the runtimes are: {time}\n")

                # if rank == 0 and serial_times and parallel_times_min_sr_s:
                #     smallest_speedup_min_sr_s, largest_speedup_min_sr_s = compute_speedup_factors(serial_times, parallel_times_min_sr_s)
                #     print(f"MIN-SR-S: For {problem_type} with {eps=}: Smallest speedup {smallest_speedup_min_sr_s:.2f}, Largest speedup {largest_speedup_min_sr_s:.2f}")

                # if rank == 0 and serial_times and parallel_times_min_sr_flex:
                #     smallest_speedup_min_sr_flex, largest_speedup_min_sr_flex = compute_speedup_factors(serial_times, parallel_times_min_sr_flex)
                #     print(f"MIN-SR-FLEX: For {problem_type} with {eps=}: Smallest speedup {smallest_speedup_min_sr_flex:.2f}, Largest speedup {largest_speedup_min_sr_flex:.2f}")

                comm.Barrier()

    if rank == 0 and do_plotting:
        if separate_errors:
            work_plotter_diff = Plotter(nrows=2, ncols=2, figsize=(12, 12))
            work_plotter_alg = Plotter(nrows=2, ncols=2, figsize=(12, 12))

            for q, work, err_diff, problem_type, i, QI, eps in results_diff:
                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker, markersize = res["marker"], res["markersize"]
                work_plotter_diff = plot_result(work_plotter_diff, work, err_diff, q, color, marker, markersize, linestyle, problem_label)

            finalize_plot(case, num_nodes, problem_name, QI_list, work_plotter_diff, solver_type, error_type=error_type, label_error="_diff", log_time=log_time, separate_errors=separate_errors)

            for q, work, err_alg, problem_type, i, QI, eps in results_alg:
                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker, markersize = res["marker"], res["markersize"]
                work_plotter_alg = plot_result(work_plotter_alg, work, err_alg, q, color, marker, markersize, linestyle, problem_label)

            finalize_plot(case, num_nodes, problem_name, QI_list, work_plotter_alg, solver_type, error_type=error_type, label_error="_alg", log_time=log_time, separate_errors=separate_errors)

        else:
            work_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

            for q, time, err, problem_type, i, QI, eps in results:
                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker, markersize = res["marker"], res["markersize"]
                plot_result(work_plotter, time, err, q, color, marker, markersize, linestyle, problem_label)

            finalize_plot(case, num_nodes, problem_name, QI_list, work_plotter, solver_type, error_type=error_type, log_time=log_time, separate_errors=separate_errors)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    problem_name = "MICHAELIS-MENTEN"

    QI_list = ["LU", "MIN-SR-S"]#["IE", "LU", "MIN-SR-S"]  # ["MIN-SR-S", "LU", "IE"]
    num_nodes = size

    case = 11#4

    dt_list = [1e-5]#None

    error_type = "global"
    separate_errors = True
    log_time = False

    compute_work_vs_error(case, comm, num_nodes, rank, problem_name, QI_list, dt_list=dt_list, error_type=error_type, separate_errors=separate_errors, log_time=log_time)

    MPI.Finalize()
