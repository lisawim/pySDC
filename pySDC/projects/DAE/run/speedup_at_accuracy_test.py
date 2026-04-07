import numpy as np
import dill
from mpi4py import MPI
from pathlib import Path
from typing import Any, Callable, Optional

from pySDC.core.errors import ParameterError
from pySDC.core.hooks import Hooks

from pySDC.projects.DAE import compute_solution
from pySDC.projects.DAE.misc.dataclasses import SpeedupAccuracyRunStats
from pySDC.projects.DAE.run.utils import set_correct_sweeper_type
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE.run.mpi_test import (
    ensure_results_path, _mean_niter, run_warmup_step
)
from pySDC.projects.DAE.misc.methods_config import RADAU_METHODS, RK_METHODS
from pySDC.projects.DAE.run.run_single_experiment import compute_qend_max_final_err

from pySDC.helpers.stats_helper import get_sorted


_PRECOMPUTED_FILENAMES = {
    "ANDREWS-SQUEEZER": "results_speedup_at_acc_dt={dt}_andrews.pkl",
    "LINEAR-TEST": "results_speedup_at_acc_dt={dt}_linear.pkl",
    "REACTION-DIFFUSION": "results_speedup_at_acc_dt={dt}_reaction_diffusion.pkl",
}


def build_filename(dt: float, problem_name: str) -> str:
    """Builds filename for specific problem to access correct stats."""
    try:
        template = _PRECOMPUTED_FILENAMES[problem_name]
    except KeyError as e:
        raise ParameterError(f"Unknown problem_name={problem_name!r} for filename generation") from e
    return template.format(dt=dt)


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
) -> Optional[SpeedupAccuracyRunStats]:
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

    t_wall_stop_at_acc = sub_comm.reduce(runtime, op=MPI.MAX, root=0)

    niter_mean = _mean_niter(solution_stats) if sub_rank == 0 else None

    timing_step = [me[1] for me in get_sorted(solution_stats, type="timing_post_step", sortby="time")]
    gathered_timings_step = sub_comm.gather(timing_step, root=0)

    if sub_rank == 0:
        e_embedded_steps = [me[1] for me in get_sorted(solution_stats, type=f"error_embedded_estimate", sortby="time")]

        e_exact_steps = [me[1] for me in get_sorted(solution_stats, type=f"exact_error", sortby="time")]

        e_global_steps = [me[1] for me in get_sorted(solution_stats, type="e_global_post_step", sortby="time")]

        t_cpu_steps = []
        for i in range(len(gathered_timings_step[0])):
            t_max = max(gathered_timings_step[r][i] for r in range(sub_size))
            t_cpu_steps.append(t_max)

        if problem_name == "ANDREWS-SQUEEZER":
            res = compute_qend_max_final_err(solution_stats, Tend)
            qend_error = res.qend_error

        result = SpeedupAccuracyRunStats(
            t_wall_stop_at_acc=t_wall_stop_at_acc,
            e_embedded_steps=e_embedded_steps,
            e_global_steps=e_global_steps,
            e_exact_steps=e_exact_steps,
            t_cpu_steps=t_cpu_steps,
            qend_error=(qend_error if problem_name == "ANDREWS-SQUEEZER" else None),
            niter_mean=niter_mean,
        )

    else:
        result = None

    sub_comm.Free()
    return result


def run_speedup_at_accuracy_test(
    global_comm: MPI.Comm,
    hook_class: list[Hooks],
    problem_name: str,
    dt: float,
    sweepers: list,
    QI_serial_methods: list,
    QI_parallel_methods: list,
    stop_at_accuracy_for_speedup: bool,
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
                        stop_at_accuracy_for_speedup=stop_at_accuracy_for_speedup,
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
                        results[key_ser][num_nodes] = SpeedupAccuracyRunStats(
                            t_wall_stop_at_acc=runtime,
                            e_global_steps=e_global_steps,
                            e_embedded_steps=None,
                            e_exact_steps=None,
                            t_cpu_steps=t_cpu_steps,
                            qend_error=(qend_error if problem_name == "ANDREWS-SQUEEZER" else None),
                            niter_mean=None,
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
                            stop_at_accuracy_for_speedup=stop_at_accuracy_for_speedup,
                        )

                        t_cpu_steps = [me[1] for me in get_sorted(solution_stats, type="timing_post_step", sortby="time")]

                        e_embedded_steps = [
                            me[1]
                            for me in get_sorted(solution_stats, type=f"error_embedded_estimate", sortby="time")
                        ]

                        e_exact_steps = [
                            me[1] for me in get_sorted(solution_stats, type=f"exact_error", sortby="time")
                        ]

                        e_global_steps = [
                            me[1] for me in get_sorted(solution_stats, type="e_global_post_step", sortby="time")
                        ]

                        if problem_name == "ANDREWS-SQUEEZER":
                            res = compute_qend_max_final_err(solution_stats, Tend)
                            qend_error = res.qend_error

                        results[key_ser][num_nodes] = SpeedupAccuracyRunStats(
                            t_wall_stop_at_acc=runtime,
                            e_embedded_steps=e_embedded_steps,
                            e_global_steps=e_global_steps,
                            e_exact_steps=e_exact_steps,
                            t_cpu_steps=t_cpu_steps,
                            qend_error=(qend_error if problem_name == "ANDREWS-SQUEEZER" else None),
                            niter_mean=_mean_niter(solution_stats),
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
                    stop_at_accuracy_for_speedup=stop_at_accuracy_for_speedup,
                )

                global_comm.Barrier()

                if global_rank == 0:
                    results[key_par][num_nodes] = result

                    with open(results_path, "wb") as f:
                        dill.dump(results, f)

    if global_rank == 0 and results_path is not None:
        with results_path.open("wb") as f:
            dill.dump(results, f)
