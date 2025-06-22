import argparse
import dill
import os
import numpy as np

from pySDC.projects.DAE import compute_solution

from pySDC.helpers.stats_helper import get_sorted


def parse_args():
    def parse_hook(path: str):
        # Pfad importieren: module.ClassName
        module_name, class_name = path.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)

    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=float, required=True)
    parser.add_argument("--dt_list", nargs="+", type=float, required=True)
    parser.add_argument("--Tend", type=float, required=True)
    parser.add_argument("--QI", type=str, required=True)
    parser.add_argument("--sweeper_type", type=str, required=True)
    parser.add_argument("--problem_name", type=str, default="DPR")
    parser.add_argument("--num_nodes", type=int, default=3)
    parser.add_argument("--use_mpi", action="store_true")
    parser.add_argument("--hook_class", nargs='+', type=parse_hook, default=[])
    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0

    max_errors = [] if rank == 0 else None
    wallclock_times = [] if rank == 0 else None

    if rank == 0:
        print(f"\nRunning {args.QI} with {args.sweeper_type}...\n")

    for dt in args.dt_list:
        if args.use_mpi:
            comm.Barrier()

            solution_stats = compute_solution(
                args.problem_name, args.t0, dt, args.Tend, args.num_nodes, args.QI, args.sweeper_type, hook_class=args.hook_class
            )

            comm.Barrier()

            timing_run = get_sorted(solution_stats, type="timing_run")[0][1]
            timing_run_full = comm.reduce(timing_run, op=MPI.MAX)

            if rank == 0:
                wallclock_times.append(timing_run_full)

        else:
            if rank == 0:
                solution_stats = compute_solution(
                    args.problem_name, args.t0, dt, args.Tend, args.num_nodes, args.QI, args.sweeper_type, hook_class=args.hook_class
                )

                timing_run_full = np.array(get_sorted(solution_stats, type="timing_run", sortby="time"))[0][1]
                wallclock_times.append(timing_run_full)

        if rank == 0:
            err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]
            max_errors.append(max(err_values))

    if rank == 0:
        fname = f"results_experiment_{args.num_nodes}.pkl"
        path = os.path.join(args.output_dir, fname)

        with open(path, "rb") as f:
            all_stats = dill.load(f)

        key = f"{args.sweeper_type}_{args.QI}"
        all_stats[key] = {"max_errors": max_errors, "wc_times": wallclock_times}

        with open(path, "wb") as f:
            dill.dump(all_stats, f)

if __name__ == '__main__':
    main()
