import numpy as np
import dill
from mpi4py import MPI
import os

from pySDC.projects.DAE import compute_solution

from pySDC.helpers.stats_helper import get_sorted

from pySDC.projects.DAE.misc.configurations import LinearTestScaling


def run_test_and_split_communicator(
        config, global_comm, global_rank, num_nodes, QI, sweeper_type, use_mpi, **kwargs
):
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

    if global_rank < num_nodes:
        # Split the communicator to create a new communicator for this test
        sub_comm = global_comm.Split(color=1, key=global_rank)
        sub_rank = sub_comm.Get_rank()

        sub_num_nodes = sub_comm.Get_size()

        # Perform the computation with the sub-communicator
        solution_stats = compute_solution(
            config.problem_name,
            config.t0,
            config.dt,
            config.Tend,
            sub_num_nodes,
            QI,
            sweeper_type,
            use_mpi,
            hook_class=config.hook_class,
            comm=sub_comm,
            **kwargs,
        )

        timing_run = get_sorted(solution_stats, type="timing_run")[0][1]
        timing_run_full = sub_comm.reduce(timing_run, op=MPI.MAX, root=0)

        sub_comm.Free()

        # Only the root of sub_comm returns the collected data
        if sub_rank == 0:
            return timing_run_full
        else:
            return None

    else:
        # Split the communicator to exclude this process
        global_comm.Split(color=MPI.UNDEFINED, key=global_rank)
        return None


def run_mpi_test(config):
    """Runs MPI test."""

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    if global_rank == 0:
        output_dir = "data" + "/" + f"{config.problem_name}" + "/" + "results"
        os.makedirs(output_dir, exist_ok=True)

        fname = f"results_scaling.pkl"
        path = os.path.join(output_dir, fname)

    config.set_num_processes(global_size)

    config.check_global_comm_size(global_size)

    results = {} if global_rank == 0 else None

    for sweeper_type in config.sweepers:
        key_ser = f"{sweeper_type}_{config.QI_ser}"
        if global_rank == 0:
            results[key_ser] = {}

        for num_nodes in config.num_processes:
            if global_rank == 0:
                results[key_ser][num_nodes] = 0
            
            if global_rank == 0:
                print(f"\nRunning {config.QI_ser} with {sweeper_type} using {num_nodes} nodes...\n")

                solution_stats = compute_solution(
                    config.problem_name,
                    config.t0,
                    config.dt,
                    config.Tend,
                    num_nodes,
                    config.QI_ser,
                    sweeper_type,
                    False,
                    hook_class=config.hook_class,
                )

                timing_run = np.array(get_sorted(solution_stats, type="timing_run", sortby="time"))

                results[key_ser][num_nodes] = timing_run[0][1]

                with open(path, "wb") as f:
                    dill.dump(results, f)

    for sweeper_type in config.sweepers:
        for QI_par in config.qDeltas_parallel:
            key_par = f"{sweeper_type}_{QI_par}"
            if global_rank == 0:
                results[key_par] = {}

            for num_nodes in config.num_processes:
                if global_rank == 0:
                    results[key_par][num_nodes] = 0

                    print(f"\nRunning {QI_par} with {sweeper_type} using {num_nodes} nodes...\n")

                global_comm.Barrier()

                timing_run_full = run_test_and_split_communicator(
                    config, global_comm, global_rank, num_nodes, QI_par, sweeper_type, True
                )

                global_comm.Barrier()

                if global_rank == 0:
                    results[key_par][num_nodes] = timing_run_full

                    with open(path, "wb") as f:
                        dill.dump(results, f)

    if global_rank == 0:
        with open(path, "wb") as f:
            dill.dump(results, f)


if __name__ == "__main__":
    config = LinearTestScaling()
    run_mpi_test(config)
    