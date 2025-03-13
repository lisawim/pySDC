from mpi4py import MPI
import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep

from pySDC.projects.DAE.run.error import get_error_label, get_problem_cases, plot_result
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter


def run_test(t0, QI, dt, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs):
    t, z, err_values, timing_step_full = None, None, None, None

    if QI in ["MIN-SR-S", "MIN-SR-FLEX"]:
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
            **kwargs,
        )

        u_stats = get_sorted(solution_stats, type="u", sortby="time")
        t = np.array([me[0] for me in u_stats])
        if not eps == 0.0:
            u = np.array([me[1] for me in u_stats])
            z = u[:, -1]
        else:
            z = np.array([me[1].alg[0] for me in u_stats])

        # err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]

        timing_step = [me[1] for me in get_sorted(solution_stats, type="timing_step", sortby="time")]
        timing_step_full = comm.reduce(timing_step, op=MPI.MAX)

    else:
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
                **kwargs,
            )

            u_stats = get_sorted(solution_stats, type="u", sortby="time")
            t = np.array([me[0] for me in u_stats])
            if not eps == 0.0:
                u = np.array([me[1] for me in u_stats])
                z = u[:, -1]
            else:
                z = np.array([me[1].alg[0] for me in u_stats])

            # err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]

            timing_step_full = [me[1] for me in get_sorted(solution_stats, type="timing_step", sortby="time")]

    comm.Barrier()  # Ensure synchronization between ranks

    return t, z, err_values, timing_step_full


def finalize_plot(k: int, dt, plotter, num_nodes, problem_name, QI_list):
    plotter.set_xlabel("time", subplot_index=None)

    for q, QI in enumerate(QI_list):
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

        plotter.set_ylabel("wall-clock time", subplot_index=q)

        plotter.set_yscale(scale="log", subplot_index=q)

    plotter.sync_ylim()

    plotter.adjust_layout(num_subplots=len(QI_list))

    plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=20)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"effort_{num_nodes=}_{dt=}_case{k}.png"
    plotter.save(filename)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem_name = "MICHAELIS-MENTEN"

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = comm.Get_size()

    t0 = 0.0
    dt = 1e-4

    case = 6

    problems = get_problem_cases(k=case, problem_name=problem_name)

    hook_class = [LogSolution]

    effort_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                # Let's do the simulation to get results
                t, z, err_values, timing_step_full = run_test(t0, QI, dt, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank)

                if rank == 0:
                    color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                    problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                    marker, markersize = res["marker"], res["markersize"]

                    effort_plotter = plot_result(effort_plotter, t, timing_step_full, q, color, None, None, linestyle, problem_label)

    if rank == 0:
        finalize_plot(case, dt, effort_plotter, num_nodes, problem_name, QI_list)
