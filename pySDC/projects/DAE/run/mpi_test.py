import numpy as np
import dill
from mpi4py import MPI
from pathlib import Path
import os
from typing import Any, Optional, Tuple

from pySDC.core.errors import ParameterError
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
) -> Tuple[Optional[float], Optional[float]]:
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

    if sub_rank == 0:
        e_emb_post_step = [
            me[1] for me in get_sorted(solution_stats, type=f"error_embedded_estimate_post_step", sortby="time")
        ]

        e_global_post_step = [me[1] for me in get_sorted(solution_stats, type="e_global_post_step", sortby="time")]

        if problem_name == "ANDREWS-SQUEEZER":
            res = compute_qend_max_final_err(solution_stats, Tend)
            qend_max_final_err = res.qend_max_final_err

        result = ScalingRunStats(
            t_wall=runtime,
            niter_mean=_mean_niter(solution_stats),
            e_emb_post_step=e_emb_post_step,
            e_global_post_step=e_global_post_step,
            qend_max_final_error=(qend_max_final_err if problem_name == "ANDREWS-SQUEEZER" else None),
        )

    else:
        result = None

    sub_comm.Free()
    return result


def run_mpi_test(
    global_comm: MPI.Comm,
    hook_class: list,
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
        # results_filename = "results_scaling_test.pkl"
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

                    e_global_post_step = [
                        me[1] for me in get_sorted(solution_stats, type="e_global_post_step", sortby="time")
                    ]

                    # Copy runtimes to all nodes in dict
                    for num_nodes in num_processes:
                        results[key_ser][num_nodes] = ScalingRunStats(
                            t_wall=runtime,
                            e_global_post_step=e_global_post_step,
                        )

                else:
                    for num_nodes in num_processes:
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

                        e_emb_post_step = [
                            me[1]
                            for me in get_sorted(
                                solution_stats, type=f"error_embedded_estimate_post_step", sortby="time"
                            )
                        ]

                        e_global_post_step = [
                            me[1] for me in get_sorted(solution_stats, type="e_global_post_step", sortby="time")
                        ]

                        if problem_name == "ANDREWS-SQUEEZER":
                            res = compute_qend_max_final_err(solution_stats, Tend)
                            qend_max_final_err = res.qend_max_final_err

                        results[key_ser][num_nodes] = ScalingRunStats(
                            t_wall=float(runtime),
                            niter_mean=_mean_niter(solution_stats),
                            e_emb_post_step=e_emb_post_step,
                            e_global_post_step=e_global_post_step,
                            qend_max_final_error=(
                                float(qend_max_final_err) if problem_name == "ANDREWS-SQUEEZER" else None
                            ),
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
