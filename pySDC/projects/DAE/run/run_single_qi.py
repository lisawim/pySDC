import argparse
import dill
from mpi4py import MPI
import numpy as np
import os

from pySDC.implementations.hooks.log_work import LogWork
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep, LogLocalErrorPostStep
from pySDC.projects.DAE.misc.hooksDAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable
from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep as LogGlobalErrorFirstVariable
from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStepPerturbation

from pySDC.projects.DAE import computeSolution, getEndTime
from pySDC.helpers.stats_helper import get_sorted

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--QI", type=str, required=True)
    parser.add_argument("--problem_type", type=str, required=True)
    parser.add_argument("--problem_name", type=str, default="DPR")
    parser.add_argument("--case", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=3)
    parser.add_argument("--solver_type", type=str, default="constrainedDAE")
    parser.add_argument("--error_type", type=str, default="global")
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--log_time", action="store_true")
    parser.add_argument("--separate_errors", action="store_true")
    return parser.parse_args()


def get_hook_class(error_type, eps, separate_errors):
    hooks = [LogWork]
    if error_type == "global":
        if separate_errors:
            if eps == 0.0:
                hooks += [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable]
            else:
                hooks += [LogGlobalErrorFirstVariable, LogGlobalErrorPostStepPerturbation]
        else:
            hooks += [LogGlobalErrorPostStep]
    elif error_type == "local":
        hooks += [LogLocalErrorPostStep]
    return hooks

def choose_time_step_sizes(problem_name):
    """Choose specific time step sizes depending on problem."""
    if problem_name == "ANDREWS-SQUEEZER":
        Tend = 0.03
        n_steps = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
        dt_list = Tend / n_steps
    elif problem_name == "LINEAR-TEST":
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
            num_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="niter", sortby="time")])
            print(f"Number of iterations: {num_iter}")

            solver_type = kwargs.get("solver_type")
            inner_iter = np.sum([me[1] for me in get_sorted(solution_stats, type=f"work_{solver_type}", sortby="time")])
            print(f"Number of {solver_type} iterations: {inner_iter}")

        if log_time:
            timing_run = get_sorted(solution_stats, type="timing_run")[0][1]
            timing_step = get_sorted(solution_stats, type="timing_step")
            timing_run_full = comm.reduce(timing_run, op=MPI.MAX)
            work.append(timing_run_full)
        else:
            solver_type = kwargs.get("solver_type")
            inner_iter = np.sum([me[1] for me in get_sorted(solution_stats, type=f"work_{solver_type}", sortby="time")])
            inner_iter_full = comm.reduce(inner_iter, op=MPI.MAX)
            work.append(inner_iter_full)

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

            if len(err_diff_values) > 0 and len(err_alg_values) > 0:
                err_diff.append(max(err_diff_values))
                err_alg.append(max(err_alg_values))

            err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_{error_type}_post_step", sortby="time")]
            if len(err_values) > 0:
                err.append(max(err_values))

            solver_type = kwargs.get("solver_type")
            inner_iter = np.sum([me[1] for me in get_sorted(solution_stats, type=f"work_{solver_type}", sortby="time")])

            if log_time:
                timing_run = np.array(get_sorted(solution_stats, type="timing_run", sortby="time"))
                work.append(timing_run[0][1])
            else:
                work.append(inner_iter)

            num_iter = np.sum([me[1] for me in get_sorted(solution_stats, type="niter", sortby="time")])

        comm.Barrier()  # Ensure synchronization between ranks

    return err, err_diff, err_alg, work

def run_single_qi():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    t0 = 0.0

    args = parse_args()

    hook_class = get_hook_class(args.error_type, args.eps, args.separate_errors)

    kwargs = {"solver_type": args.solver_type, "e_tol": -1, "maxiter": 10}

    dt_list_args = choose_time_step_sizes(args.problem_name)

    # print(f"Rank {rank} has {args.num_nodes} nodes")

    if args.QI in ["MIN-SR-S", "MIN-SR-NS"]:
        err, err_diff, err_alg, work = run_parallel_test(
            t0=t0,
            QI=args.QI,
            dt_list=dt_list_args,
            num_nodes=args.num_nodes,
            problem_name=args.problem_name,
            problem_type=args.problem_type,
            hook_class=hook_class,
            eps=args.eps,
            comm=comm,
            rank=rank,
            error_type=args.error_type,
            log_time=args.log_time,
            separate_errors=args.separate_errors,
            **kwargs,
        )

    else:
        err, err_diff, err_alg, work = run_serial_test(
            t0=t0,
            QI=args.QI,
            dt_list=dt_list_args,
            num_nodes=args.num_nodes,
            problem_name=args.problem_name,
            problem_type=args.problem_type,
            hook_class=hook_class,
            eps=args.eps,
            comm=comm,
            rank=rank,
            error_type=args.error_type,
            log_time=args.log_time,
            separate_errors=args.separate_errors,
            **kwargs,
        )

    if rank == 0:
        result_dir = os.environ.get("PYSDC_RESULT_DIR", "data")
        os.makedirs(result_dir, exist_ok=True)

        fname = f"result_{args.QI}_{args.problem_type}_eps{str(args.eps).replace('.', 'p')}.pkl"
        full_path = os.path.join(result_dir, fname)

        with open(full_path, "wb") as f:
            dill.dump((err, err_diff, err_alg, work), f)


if __name__ == "__main__":
    run_single_qi()
