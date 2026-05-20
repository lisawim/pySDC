import argparse
import dill
import os
import numpy as np
from typing import Any, Literal, Optional

from pySDC.projects.DAE import compute_solution
from pySDC.projects.DAE.misc.dataclasses import QEndErrorResult, WorkPrecisionResult
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import qend_ref_testset

from pySDC.helpers.stats_helper import get_sorted


def compute_qend_max_final_err(
    solution_stats: Any,
    Tend: float,
    *,
    prefix: str = "",
    q_dim: int = 7,
    atol: float = 1e-14,
    on_fail: Literal["raise", "nan", "none"] = "raise",
) -> Optional[QEndErrorResult]:
    """
    Compute max-norm final error in q(Tend) against reference qend_ref_testset(t_ref).

    Parameters
    ----------
    solution_stats
        Stats object/dict consumed by get_sorted().
    Tend : float
        Target end time to match against time entries in solution_stats.
    q_dim : int
        Number of q-components extracted from u (default: 7 for Andrews).
    atol : float
        Absolute tolerance for time matching.
    on_fail : list
        Behavior if no suitable time entry is found:
        - "raise": raise ValueError
        - "nan": return result with qend_max_final_err = np.nan (and ind = -1)
        - "none": return None

    Returns
    -------
    QEndErrorResult or None
    """

    assert prefix in {"", "_post_iteration"}, f"Unexpected prefix={prefix!r}"

    u_val = get_sorted(solution_stats, type=f"u{prefix}", sortby="time")
    if not u_val:
        msg = f"solution_stats contains no 'u{prefix}' entries."
        if on_fail == "raise":
            raise ValueError(msg)
        if on_fail == "none":
            return None
        return QEndErrorResult(-1, float("nan"), np.array([]), np.array([]), float("nan"))

    t = np.array([me[0] for me in u_val], dtype=float)

    # u: shape (n, dof). flatten() ensures 1D per time entry
    u = np.array([np.asarray(me[1]).flatten() for me in u_val], dtype=float)

    if u.ndim != 2 or u.shape[1] < q_dim:
        raise ValueError(f"Expected u to have at least {q_dim} dofs, got shape {u.shape}.")

    q = u[:, :q_dim]

    # Find index near Tend
    i = int(np.searchsorted(t, Tend))
    ind = None

    if i < len(t) and np.isclose(t[i], Tend, atol=atol):
        ind = i
    elif i > 0 and np.isclose(t[i - 1], Tend, atol=atol):
        ind = i - 1

    if ind is None:
        msg = f"No suitable entry found near Tend={Tend} (atol={atol})."
        if on_fail == "raise":
            raise ValueError(msg)
        if on_fail == "none":
            return None
        return QEndErrorResult(-1, float("nan"), np.array([]), np.array([]), float("nan"))

    t_ref = t[ind]
    qend_ref = np.asarray(qend_ref_testset(t_ref), dtype=float).reshape(-1)
    qend = np.asarray(q[ind, :], dtype=float).reshape(-1)

    if qend_ref.shape[0] != qend.shape[0]:
        raise ValueError(f"Shape mismatch: qend has {qend.shape[0]} entries, qend_ref has {qend_ref.shape[0]}.")

    qend_max_final_err = max(abs(qend - qend_ref))

    return QEndErrorResult(
        ind=ind,
        t_ref=t_ref,
        qend=qend,
        qend_ref=qend_ref,
        qend_error=qend_max_final_err,
    )


def parse_args():
    def parse_hook(path: str):
        # Import the path module.ClassName
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

    hook_class = args.hook_class + [LogSolution]

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
                problem_name=args.problem_name,
                t0=args.t0,
                dt=dt,
                Tend=args.Tend,
                num_nodes=args.num_nodes,
                QI=args.QI,
                sweeper_type=args.sweeper_type,
                use_mpi=args.use_mpi,
                hook_class=hook_class,
                measure=True,
            )

            comm.Barrier()

            timing_run = runtime
            timing_run_full = comm.reduce(timing_run, op=MPI.MAX, root=0)

            if rank == 0:
                wallclock_times.append(timing_run_full)

        else:
            if rank == 0:
                print(f"- {dt=}..")

                runtime, solution_stats = compute_solution(
                    problem_name=args.problem_name,
                    t0=args.t0,
                    dt=dt,
                    Tend=args.Tend,
                    num_nodes=args.num_nodes,
                    QI=args.QI,
                    sweeper_type=args.sweeper_type,
                    use_mpi=args.use_mpi,
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
                res = compute_qend_max_final_err(solution_stats, args.Tend)
                qend_max_final_err = res.qend_error
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
