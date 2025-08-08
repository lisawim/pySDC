import argparse
import dill
import os
import numpy as np

from pySDC.projects.DAE import compute_solution
from pySDC.implementations.hooks.log_solution import LogSolution

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
    # parser.add_argument("--skip_residual_computation", type=str, )
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

    hook_class = args.hook_class
    hook_class += [LogSolution]

    # Dummy run to avoid start overhead
    if args.use_mpi:
        comm.Barrier()

        dt_dummy = 1e-4

        _ = compute_solution(
            args.problem_name,
            args.t0,
            dt=dt_dummy,
            Tend=args.t0 + dt_dummy,
            num_nodes=args.num_nodes,
            QI="MIN-SR-NS",
            sweeper_type="constrainedDAE",
            use_mpi=args.use_mpi,
            hook_class=hook_class,
            measure=False,
        )

    if args.use_mpi:
        comm.Barrier()

    all_max_global_error_full = [] if rank == 0 else None
    q_max_final_error_full = [] if rank == 0 else None
    wallclock_times = [] if rank == 0 else None

    if rank == 0:
        print(f"\nRunning {args.QI} with {args.sweeper_type}...\n")

    for dt in args.dt_list:
        if args.use_mpi:
            if rank == 0:
                print(f"- {dt=}..")

            comm.Barrier()

            runtime, solution_stats = compute_solution(
                args.problem_name,
                args.t0,
                dt,
                args.Tend,
                args.num_nodes,
                args.QI,
                args.sweeper_type,
                args.use_mpi,
                hook_class=hook_class,
                measure=True,
            )

            comm.Barrier()

            timing_run = runtime
            timing_run_full = comm.reduce(timing_run, op=MPI.MAX)

            if rank == 0:
                wallclock_times.append(timing_run_full)

        else:
            if rank == 0:
                print(f"- {dt=}..")

                runtime, solution_stats = compute_solution(
                    args.problem_name,
                    args.t0,
                    dt,
                    args.Tend,
                    args.num_nodes,
                    args.QI,
                    args.sweeper_type,
                    args.use_mpi,
                    hook_class=hook_class,
                    measure=True,
                )

                timing_run_full = runtime
                wallclock_times.append(timing_run_full)

        if rank == 0:
            err_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_post_step", sortby="time")]
            all_max_global_error_full.append(max(err_values))

            # Store solution at Tend = 0.03 (for Andrews' problem)
            if args.problem_name == "ANDREWS-SQUEEZER":
                from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import qend_ref_testset

                u_val = get_sorted(solution_stats, type="u", sortby="time")
                t = np.array([me[0] for me in u_val])
                u = np.array([me[1].flatten() for me in u_val])
                q = u[:, : 7]

                i = np.searchsorted(t, args.Tend)
                if i < len(t) and np.isclose(t[i], args.Tend, atol=1e-14):
                    ind = i
                elif i > 0 and np.isclose(t[i-1], args.Tend, atol=1e-14):
                    ind = i - 1
                else:
                    print("No suitable entry found.")

                t_ref = t[ind]
                qend_ref = qend_ref_testset(t_ref)

                qend = q[ind, :]
                qend_max_final_err = max(abs(qend - qend_ref))
                q_max_final_error_full.append(qend_max_final_err)

    if rank == 0:
        fname = f"results_experiment_{args.num_nodes}.pkl"
        path = os.path.join(args.output_dir, fname)

        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as f:
                all_stats = dill.load(f)
        else:
            all_stats = {}

        key = f"{args.sweeper_type}_{args.QI}"
        all_stats[key] = {
            "all_max_global_error": all_max_global_error_full,
            "wc_times": wallclock_times,
        }

        if args.problem_name == "ANDREWS-SQUEEZER":
            all_stats[key].update({"q_max_final_error": q_max_final_error_full})

        with open(path, "wb") as f:
            dill.dump(all_stats, f)

if __name__ == '__main__':
    main()
