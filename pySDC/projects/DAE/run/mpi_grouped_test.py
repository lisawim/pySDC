import dill
from mpi4py import MPI
from pathlib import Path
from typing import Any, Optional

from pySDC.core.errors import ParameterError
from pySDC.projects.DAE import compute_solution
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.mpi_test import ensure_results_path, _mean_niter, run_warmup_step
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes


_FILENAME_TEMPLATES = {
    "ANDREWS-SQUEEZER": "results_grouped_dt={dt}_P={P}_andrews.pkl",
    "LINEAR-TEST": "results_grouped_dt={dt}_P={P}_linear.pkl",
    "REACTION-DIFFUSION": "results_grouped_dt={dt}_P={P}_reaction_diffusion.pkl",
}


def build_filename(dt: float, problem_name: str, global_size: int) -> str:
    """Build filename for grouped scaling results."""
    template = _FILENAME_TEMPLATES.get(
        problem_name,
        "results_grouped_dt={dt}_P={P}_" + problem_name + ".pkl",
    )
    return template.format(dt=format_dt(dt), P=global_size)


def format_dt(dt: float) -> str:
    """Stable formatting for float time steps in filenames."""
    return f"{dt:.1e}"


def build_ranks_to_test(num_nodes: int, ranks_to_test: list = None) -> list:
    """
    Build a list of MPI ranks to test, starting from num_nodes and halving until 1.

    Parameters
    ----------
    num_nodes : int
        Maximum number of ranks (e.g. number of collocation nodes).
    ranks_to_test : list or None, optional
        If provided, this list is returned unchanged. If None, it is generated.

    Returns
    -------
    list
        List like [num_nodes, num_nodes//2, ..., 1]
    """
    if ranks_to_test is not None:
        return ranks_to_test

    P = num_nodes
    ranks_to_test = []

    while P >= 1:
        ranks_to_test.append(P)
        if P == 1:
            break
        P //= 2

    return ranks_to_test


def run_grouped_case(
    *,
    problem_name: str,
    t0: float,
    dt: float,
    Tend: float,
    global_comm: MPI.Comm,
    global_rank: int,
    num_nodes: int,
    num_ranks: int,
    QI: str,
    sweeper_type: str,
    use_mpi: bool,
    use_mpi_grouped: bool,
    **kwargs: Any,
):
    """
    Run one grouped MPI test with fixed M=num_nodes but varying P=num_ranks.

    Uses a sub-communicator with size=num_ranks and calls compute_solution with num_nodes=M.
    """

    if global_rank >= num_ranks:
        global_comm.Split(color=MPI.UNDEFINED, key=global_rank)
        return None, None

    sub_comm = global_comm.Split(color=1, key=global_rank)
    sub_rank = sub_comm.Get_rank()
    sub_size = sub_comm.Get_size()
    assert sub_size == num_ranks

    runtime, solution_stats = compute_solution(
        problem_name=problem_name,
        t0=t0,
        dt=dt,
        Tend=Tend,
        num_nodes=num_nodes,
        QI=QI,
        sweeper_type=sweeper_type,
        use_mpi=use_mpi,
        use_mpi_grouped=use_mpi_grouped,
        measure=True,
        comm=sub_comm,
        **kwargs,
    )

    t_wall = sub_comm.reduce(runtime, op=MPI.MAX, root=0)

    niter_mean = _mean_niter(solution_stats) if sub_rank == 0 else None

    sub_comm.Free()
    return (t_wall, niter_mean) if sub_rank == 0 else (None, None)


def run_mpi_grouped_breakeven_test(
    global_comm: MPI.Comm,
    problem_name: str,
    dt: float,
    sweepers: list,
    QI_parallel_methods: list,
    num_nodes: int,
    ranks_to_test: list = None,
    **kwargs,
) -> None:
    """Run grouped-MPI breakeven test for fixed M and varying P."""

    assert global_comm is not None
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    ranks_to_test = build_ranks_to_test(num_nodes, ranks_to_test)

    sanity_checks(global_size, ranks_to_test)

    # Prepare output
    results_path: Optional[Path] = None
    if global_rank == 0:
        # Use explicit filename by default, but keep helper around:
        results_filename = build_filename(dt, problem_name)
        # results_filename = "results_scaling_test.pkl"
        results_path = ensure_results_path(problem_name, results_filename)

    results = {} if global_rank == 0 else None

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    # Warm-up once on the full communicator.
    run_warmup_step(global_comm=global_comm, global_size=global_size, problem_name=problem_name, t0=t0)

    # ---- Actual tests ----
    for sweeper_type in sweepers:
        for QI_par in QI_parallel_methods:
            key = f"{sweeper_type}_{QI_par}"
            if global_rank == 0:
                results[key] = {}

            for P in ranks_to_test:
                global_comm.Barrier()

                t_wall, niter_mean = run_grouped_case(
                    problem_name=problem_name,
                    t0=t0,
                    dt=dt,
                    Tend=Tend,
                    global_comm=global_comm,
                    global_rank=global_rank,
                    num_nodes=num_nodes,
                    num_ranks=P,
                    QI=QI_par,
                    sweeper_type=sweeper_type,
                    **kwargs,
                )

                global_comm.Barrier()

                if global_rank == 0:
                    results[key][P] = {
                        "t_wall": t_wall,
                        "niter_mean": niter_mean,
                    }

                    with open(results_path, "wb") as f:
                        dill.dump(results, f)

    if global_rank == 0:
        with open(results_path, "wb") as f:
            dill.dump(results, f)


def sanity_checks(global_size: int, ranks_to_test: list) -> None:
    """Some checking."""
    if max(ranks_to_test) > global_size:
        raise ParameterError(
            f"Need at least max(ranks_to_test) MPI processes allocated. "
            f"Got {global_size=}, but need at least {max(ranks_to_test)}."
        )
    if max(ranks_to_test) > num_nodes:
        raise ParameterError(
            f"Need num_ranks <= num_nodes for grouped sweeper. " f"Got max ranks {max(ranks_to_test)} but {num_nodes=}."
        )


if __name__ == "__main__":
    global_comm = MPI.COMM_WORLD
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    num_nodes = global_comm.Get_size()
    config_linear["num_nodes"] = num_nodes
    run_mpi_grouped_breakeven_test(global_comm=global_comm, num_nodes=num_nodes, **config_linear)
