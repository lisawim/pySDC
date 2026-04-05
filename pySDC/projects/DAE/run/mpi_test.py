import numpy as np
import dill
from mpi4py import MPI
from pathlib import Path
from typing import Any, Callable, Optional
from collections import defaultdict

from pySDC.core.errors import ParameterError
from pySDC.core.hooks import Hooks

from pySDC.projects.DAE import compute_solution
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.misc.dataclasses import ScalingRunStats
from pySDC.projects.DAE.run.utils import set_correct_sweeper_type
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE.misc.methods_config import RADAU_METHODS, RK_METHODS
from pySDC.projects.DAE.run.run_single_experiment import compute_qend_max_final_err

from pySDC.helpers.stats_helper import get_sorted


_PRECOMPUTED_FILENAMES = {
    "ANDREWS-SQUEEZER": "results_scaling_dt={dt}_andrews.pkl",
    "LINEAR-TEST": "results_scaling_dt={dt}_linear.pkl",
    "REACTION-DIFFUSION": "results_scaling_dt={dt}_reaction_diffusion.pkl",
}


def build_filename(dt: float, problem_name: str) -> str:
    """Builds filename for specific problem to access correct stats."""
    try:
        template = _PRECOMPUTED_FILENAMES[problem_name]
    except KeyError as e:
        raise ParameterError(f"Unknown problem_name={problem_name!r} for filename generation") from e
    return template.format(dt=dt)


def get_time_and_index_of_step(t0: float, dt: float, Tend: float, problem_name: str) -> float:
    """Get the time of the first step for a given problem."""
    if problem_name == "ANDREWS-SQUEEZER":
        return Tend, -1
    else:
        return t0 + dt, 0


def _group_by_time_rounded(pairs, *, round_t: int = 14, transform: Callable = lambda x: x):
    """Group a list of (t, value) pairs by rounded t."""

    d: dict[float, list[float]] = defaultdict(list)
    for t, val in pairs:
        d[round(float(t), round_t)].append(transform(val))
    return d


def _filter_pairs_to_one_step(pairs, t_step: float, round_t: int = 14) -> list[tuple[float, float]]:
    """
    Keep only those (t, val) entries that belong to the first time step.
    The "first time step" is defined as the minimal rounded t appearing in the list.
    """

    counter = defaultdict(int)
    keys = []
    vals = []
    for t, v in pairs:
        t_key = round(float(t), round_t)
        if t_key == t_step:
            k = counter[t_key]
            counter[t_key] += 1
            keys.append((t_key, k))
            vals.append(float(v))
        else:
            continue

    return keys, np.asarray(vals, dtype=float)


def max_timing_over_ranks_per_post_iter(all_rank_timings, *, round_t: int = 14):
    """
    all_rank_timings: dict[rank] -> list[(t, tau)] for type="timing_post_iteration".

    Returns
    -------
    keys : list[(t, k)]
        Chronological post-iteration keys.
    tau_max : np.ndarray
        tau_max[j] is the max timing over ranks for keys[j].

    Notes
    -----
    This is the standard "straggler" wallclock proxy: the iteration is done
    when the slowest rank is done.
    """
    per_rank = {r: _group_by_time_rounded(lst, round_t=round_t) for r, lst in all_rank_timings.items()}
    all_times = sorted(set().union(*[set(d.keys()) for d in per_rank.values()]))

    keys: list[tuple[float, int]] = []
    tau_max: list[float] = []

    ranks = list(per_rank.keys())
    for t in all_times:
        lists = [per_rank[r].get(t, []) for r in ranks]
        max_len = max((len(L) for L in lists), default=0)
        for k in range(max_len):
            taus_k = [L[k] for L in lists if len(L) > k]
            if not taus_k:
                continue
            keys.append((t, k))
            tau_max.append(max(taus_k))

    return keys, np.asarray(tau_max, dtype=float)


def max_timing_over_ranks_one_step_per_post_iter(all_rank_timings, t_step: float, idx_step: int, round_t: int = 14):
    """
    all_rank_timings: dict[rank] -> list[(t, tau)] for type="timing_post_iteration".

    Returns only the first time step (minimal t) and all its post-iterations.
    """
    # group per rank by rounded time
    per_rank = {r: _group_by_time_rounded(l, round_t=round_t) for r, l in all_rank_timings.items()}

    # determine the first step time key globally (smallest t across all ranks)
    all_times = sorted(set().union(*[set(d.keys()) for d in per_rank.values()]))
    if not all_times:
        return [], np.asarray([], dtype=float)

    # First or last time step
    t_found = all_times[idx_step]
    if t_found != round(float(t_step), round_t):
        raise ValueError(f"Warning: expected first time step to be around t1={t_step}, but got t0={t_found} after rounding.")

    ranks = list(per_rank.keys())
    lists = [per_rank[r].get(t_step, []) for r in ranks]
    max_len = max((len(l) for l in lists))

    keys: list[tuple[float, int]] = []
    tau_max: list[float] = []

    for k in range(max_len):
        taus_k = [l[k] for l in lists if len(l) > k]
        if not taus_k:
            continue
        keys.append((t_step, k))
        tau_max.append(max(taus_k))  # straggler proxy within this step

    return keys, np.asarray(tau_max, dtype=float)


def cumulative_from_durations(durations: Any) -> np.ndarray:
    """Cumulative sum, returned as float array."""
    return np.cumsum(np.asarray(durations, dtype=float))


def build_accuracy_vs_time_post_iter(
    *,
    all_rank_timings: dict,
    e_global_post_iteration_steps: list,
    t_step: float,
    idx_step: int,
    round_t: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (T_cpu, e) pairs on post-iteration level.

    Parameters
    ----------
    all_rank_timings : dict[rank] -> list[(t, tau)]
        Rank-wise timing lists for "timing_post_iteration".
    e_global_post_iteration : list[(t, e)]
        Global error list for "e_global_post_iteration". Same on all ranks.
    """
    t_keys, tau_max = max_timing_over_ranks_one_step_per_post_iter(all_rank_timings, t_step=t_step, idx_step=idx_step, round_t=round_t)
    e_keys, e_vals = _filter_pairs_to_one_step(e_global_post_iteration_steps, t_step=t_step, round_t=round_t)

    e_map = {k: val for k, val in zip(e_keys, e_vals)}
    
    # Align: keep only keys that exist in both (in timing order)
    keep_tau = []
    keep_e = []
    for j, k in enumerate(t_keys):
        if k in e_map:
            keep_tau.append(float(tau_max[j]))
            keep_e.append(float(e_map[k]))

    T_cpu = cumulative_from_durations(keep_tau)
    e = np.asarray(keep_e, dtype=float)
    return T_cpu, e


def build_accuracy_vs_time_post_iter_serial(
    *,
    e_global_post_iteration_steps: list,
    timing_post_iteration_steps: list,
    t_step: float,
    idx_step: int,
    round_t: int = 14,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Serial convenience wrapper.

    For serial runs there is only one "rank". We still want the exact same
    (t_cpu_one_step, e_global_post_iteration) fields in ScalingRunStats.

    Returns (None, None) if required stats are missing.
    """

    if len(timing_post_iteration_steps) == 0 or len(e_global_post_iteration_steps) == 0:
        return None, None

    all_rank_timings = {0: timing_post_iteration_steps}

    t_cpu_one_step, e_global_one_step = build_accuracy_vs_time_post_iter(
        all_rank_timings=all_rank_timings,
        e_global_post_iteration_steps=e_global_post_iteration_steps,
        t_step=t_step,
        idx_step=idx_step,
        round_t=round_t,
    )
    return t_cpu_one_step, e_global_one_step


def run_warmup_step(
    *,
    global_comm: MPI.Comm,
    global_size: int,
    problem_name: str,
    t0: float,
    dt_dummy: float = 1e-4,
    QI: str = "MIN-SR-NS",
    sweeper_type: str = "constrainedDAE",
) -> None:
    """
    Run a tiny dummy step to avoid one-time overhead (e.g. setup, allocations, JIT)
    from polluting the first measured run.
    """

    global_comm.Barrier()

    _ = compute_solution(
        problem_name,
        t0,
        dt=dt_dummy,
        Tend=t0 + dt_dummy,
        num_nodes=global_size,
        QI=QI,
        sweeper_type=sweeper_type,
        use_mpi=True,
        measure=False,
        comm=global_comm,
    )

    global_comm.Barrier()


def ensure_results_path(problem_name: str, filename: str) -> Path:
    """Create results directory and return full path to the pickle file."""
    output_dir = Path("data") / problem_name / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def _mean_niter(solution_stats: dict) -> float:
    niter = [me[1] for me in get_sorted(solution_stats, type="niter", sortby="time")]
    return float(np.mean(niter))


def run_test_and_split_communicator(
    *,
    problem_name: str,
    t0: float,
    dt: float,
    Tend: float,
    global_comm: MPI.Comm,
    global_rank: int,
    num_nodes: int,
    QI: str,
    sweeper_type: str,
    use_mpi: bool,
    **kwargs: Any,
) -> Optional[ScalingRunStats]:
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
    if global_rank >= num_nodes:
        # Exclude this process from the test.
        global_comm.Split(color=MPI.UNDEFINED, key=global_rank)
        return None, None

    # Split the communicator to create a new communicator for this test
    sub_comm = global_comm.Split(color=1, key=global_rank)
    sub_rank = sub_comm.Get_rank()
    sub_size = sub_comm.Get_size()
    assert sub_size == num_nodes, (sub_size, num_nodes)

    # Perform the computation with the sub-communicator
    runtime, solution_stats = compute_solution(
        problem_name,
        t0,
        dt,
        Tend,
        sub_size,
        QI,
        sweeper_type,
        use_mpi,
        measure=True,
        comm=sub_comm,
        **kwargs,
    )

    t_wall = sub_comm.reduce(runtime, op=MPI.MAX, root=0)

    niter_mean = _mean_niter(solution_stats) if sub_rank == 0 else None

    timing_iteration = get_sorted(solution_stats, type="timing_post_iteration", sortby="time")
    gathered_timings = sub_comm.gather(timing_iteration, root=0)

    timing_step = [me[1] for me in get_sorted(solution_stats, type="timing_post_step", sortby="time")]
    gathered_timings_step = sub_comm.gather(timing_step, root=0)

    if sub_rank == 0:
        e_embedded_steps = [me[1] for me in get_sorted(solution_stats, type=f"error_embedded_estimate", sortby="time")]

        e_global_steps = [me[1] for me in get_sorted(solution_stats, type="e_global_post_step", sortby="time")]

        type = "e_position_end_post_iteration" if problem_name == "ANDREWS-SQUEEZER" else "e_global_post_iteration"
        e_global_post_iteration_steps = get_sorted(solution_stats, type=type, sortby="time")

        # Build (cumulative time, error) pairs on post-iteration level.
        t_step, idx_step = get_time_and_index_of_step(t0=t0, dt=dt, Tend=Tend, problem_name=problem_name)
        all_rank_timings = {r: gathered_timings[r] for r in range(sub_size)}

        t_cpu_one_step, e_global_post_iter_vals = build_accuracy_vs_time_post_iter(
            all_rank_timings=all_rank_timings,
            e_global_post_iteration_steps=e_global_post_iteration_steps,
            t_step=t_step,
            idx_step=idx_step,
        )

        t_cpu_steps = []
        for i in range(len(gathered_timings_step[0])):
            t_max = max(gathered_timings_step[r][i] for r in range(sub_size))
            t_cpu_steps.append(t_max)

        if problem_name == "ANDREWS-SQUEEZER":
            res = compute_qend_max_final_err(solution_stats, Tend)
            qend_error = res.qend_error

        result = ScalingRunStats(
            t_wall=t_wall,
            e_embedded_steps=e_embedded_steps,
            e_global_steps=e_global_steps,
            t_cpu_steps=t_cpu_steps,
            qend_error=(qend_error if problem_name == "ANDREWS-SQUEEZER" else None),
            niter_mean=niter_mean,
            t_cpu_one_step=t_cpu_one_step,
            e_global_one_step=e_global_post_iter_vals,
        )

    else:
        result = None

    sub_comm.Free()
    return result


def run_mpi_test(
    global_comm: MPI.Comm,
    hook_class: list[Hooks],
    problem_name: str,
    dt: float,
    sweepers: list,
    QI_serial_methods: list,
    QI_parallel_methods: list,
    **kwargs: Any,
) -> None:
    """Runs MPI test."""

    assert global_comm is not None

    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    results_path: Optional[Path] = None
    if global_rank == 0:
        # Use explicit filename by default, but keep helper around:
        results_filename = build_filename(dt, problem_name)
        results_path = ensure_results_path(problem_name, results_filename)

    num_processes = range(2, global_size + 1)

    if global_size > num_processes[-1]:
        raise ParameterError(
            f"Only maximum {num_processes[-1]} processes are allowed, but global size is {global_size}"
        )

    results = {} if global_rank == 0 else None

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    # Warm-up once on the full communicator.
    run_warmup_step(global_comm=global_comm, global_size=global_size, problem_name=problem_name, t0=t0)

    is_executed = {name: False for name in RADAU_METHODS + RK_METHODS} if global_rank == 0 else None

    # Serial runs (rank 0 only)
    for QI_ser in QI_serial_methods:
        for sweeper_type in sweepers:
            sweeper_type_eff = sweeper_type

            if global_rank == 0:
                if QI_ser in RADAU_METHODS + RK_METHODS:
                    if is_executed[QI_ser]:
                        continue

                    sweeper_type_eff = set_correct_sweeper_type(sweeper_type=sweeper_type, QI=QI_ser)
                    is_executed[QI_ser] = True

                key_ser = f"{sweeper_type_eff}_{QI_ser}"

                if key_ser not in results:
                    results[key_ser] = {}

                if QI_ser in RADAU_METHODS + RK_METHODS:
                    num_nodes_ref = num_processes[0]

                    runtime, solution_stats = compute_solution(
                        problem_name,
                        t0,
                        dt,
                        Tend,
                        num_nodes_ref,
                        QI_ser,
                        sweeper_type_eff,
                        hook_class=hook_class,
                        use_mpi=False,
                        measure=True,
                    )

                    t_cpu_steps = [me[1] for me in get_sorted(solution_stats, type="timing_post_step", sortby="time")]

                    e_global_steps = [
                        me[1] for me in get_sorted(solution_stats, type="e_global_post_step", sortby="time")
                    ]

                    if problem_name == "ANDREWS-SQUEEZER":
                        res = compute_qend_max_final_err(solution_stats, Tend)
                        qend_error = res.qend_error

                    # Copy runtimes to all nodes in dict
                    for num_nodes in num_processes:
                        results[key_ser][num_nodes] = ScalingRunStats(
                            t_wall=runtime,
                            e_global_steps=e_global_steps,
                            e_embedded_steps=None,
                            t_cpu_steps=t_cpu_steps,
                            qend_error=(
                                float(qend_error) if problem_name == "ANDREWS-SQUEEZER" else None
                            ),
                            niter_mean=None,
                            t_cpu_one_step=None,
                            e_global_one_step=None,
                        )

                else:
                    for num_nodes in num_processes:
                        print(f"... Running for {QI_ser} with {num_nodes=} ...")

                        results[key_ser][num_nodes] = {}

                        runtime, solution_stats = compute_solution(
                            problem_name,
                            t0,
                            dt,
                            Tend,
                            num_nodes,
                            QI_ser,
                            sweeper_type_eff,
                            hook_class=hook_class,
                            use_mpi=False,
                            measure=True,
                        )

                        # --- new: accuracy vs. time (post-iteration level) ---
                        t_step, idx_step = get_time_and_index_of_step(t0, dt, Tend, problem_name)

                        timing_post_iteration_steps = get_sorted(solution_stats, type="timing_post_iteration", sortby="time")

                        type = "e_position_end_post_iteration" if problem_name == "ANDREWS-SQUEEZER" else "e_global_post_iteration"
                        e_global_post_iteration_steps = get_sorted(solution_stats, type=type, sortby="time")

                        t_cpu_one_step, e_global_post_iter_vals = build_accuracy_vs_time_post_iter_serial(
                            e_global_post_iteration_steps=e_global_post_iteration_steps,
                            timing_post_iteration_steps=timing_post_iteration_steps,
                            t_step=t_step,
                            idx_step=idx_step,
                        )

                        t_cpu_steps = [me[1] for me in get_sorted(solution_stats, type="timing_post_step", sortby="time")]

                        e_embedded_steps = [
                            me[1]
                            for me in get_sorted(solution_stats, type=f"error_embedded_estimate", sortby="time")
                        ]

                        e_global_steps = [
                            me[1] for me in get_sorted(solution_stats, type="e_global_post_step", sortby="time")
                        ]

                        if problem_name == "ANDREWS-SQUEEZER":
                            res = compute_qend_max_final_err(solution_stats, Tend)
                            qend_error = res.qend_error

                        results[key_ser][num_nodes] = ScalingRunStats(
                            t_wall=runtime,
                            e_embedded_steps=e_embedded_steps,
                            e_global_steps=e_global_steps,
                            t_cpu_steps=t_cpu_steps,
                            qend_error=(
                                float(qend_error) if problem_name == "ANDREWS-SQUEEZER" else None
                            ),
                            niter_mean=_mean_niter(solution_stats),
                            t_cpu_one_step=t_cpu_one_step,
                            e_global_one_step=e_global_post_iter_vals,
                        )

                # Persist after each serial block for robustness.
                assert results_path is not None
                with results_path.open("wb") as f:
                    dill.dump(results, f)

    # Warm-up once on the full communicator.
    run_warmup_step(global_comm=global_comm, global_size=global_size, problem_name=problem_name, t0=t0)

    # Parallel runs
    for sweeper_type in sweepers:
        for QI_par in QI_parallel_methods:
            key_par = f"{sweeper_type}_{QI_par}"

            if global_rank == 0:
                results[key_par] = {}

            for num_nodes in num_processes:
                if global_rank == 0:
                    print(f"... Running for {QI_par} with {num_nodes=} ...")

                    results[key_par][num_nodes] = {}

                global_comm.Barrier()

                result = run_test_and_split_communicator(
                    problem_name=problem_name,
                    t0=t0,
                    dt=dt,
                    Tend=Tend,
                    global_comm=global_comm,
                    global_rank=global_rank,
                    num_nodes=num_nodes,
                    QI=QI_par,
                    hook_class=hook_class,
                    sweeper_type=sweeper_type,
                    use_mpi=True,
                )

                global_comm.Barrier()

                if global_rank == 0:
                    results[key_par][num_nodes] = result

                    with open(results_path, "wb") as f:
                        dill.dump(results, f)

    if global_rank == 0 and results_path is not None:
        with results_path.open("wb") as f:
            dill.dump(results, f)


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    run_mpi_test(global_comm, **config_linear)
