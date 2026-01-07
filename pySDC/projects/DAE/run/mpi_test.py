import dill
from mpi4py import MPI
import os

from pySDC.core.errors import ParameterError
from pySDC.projects.DAE import compute_solution
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE.misc.methods_config import RADAU_METHODS, RK_METHODS


def build_filename(dt, problem_name):
    """Builds filename for specific problem to access correct stats."""

    if problem_name == "ANDREWS-SQUEEZER":
        return f"results_scaling_{dt=}_andrews.pkl"
    elif problem_name == "LINEAR-TEST":
        return f"results_scaling_{dt=}_linear.pkl"
    elif problem_name == "REACTION-DIFFUSION":
        return f"results_scaling_{dt=}_reaction_diffusion.pkl"


def run_test_and_split_communicator(
    problem_name,
    t0,
    dt,
    Tend,
    global_comm,
    global_rank,
    num_nodes,
    QI,
    sweeper_type,
    use_mpi,
    **kwargs,
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
        runtime, solution_stats = compute_solution(
            problem_name,
            t0,
            dt,
            Tend,
            sub_num_nodes,
            QI,
            sweeper_type,
            use_mpi,
            measure=True,
            comm=sub_comm,
            **kwargs,
        )

        timing_run_full = sub_comm.reduce(runtime, op=MPI.MAX, root=0)

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


def run_mpi_test(problem_name, dt, sweepers, QI_serial_methods, QI_parallel_methods, **kwargs):
    """Runs MPI test."""

    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    if global_rank == 0:
        output_dir = "data" + "/" + f"{problem_name}" + "/" + "results"
        os.makedirs(output_dir, exist_ok=True)

        filename = build_filename(dt, problem_name)
        path = os.path.join(output_dir, filename)

    num_processes = range(2, global_size + 1)

    if global_size > num_processes[-1]:
        raise ParameterError(
            f"Only maximum {num_processes[-1]} processes are allowed, but global size is {global_size}"
        )

    results = {} if global_rank == 0 else None

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    is_executed = {name: False for name in RADAU_METHODS + RK_METHODS} if global_rank == 0 else None

    for QI_ser in QI_serial_methods:
        for sweeper_type in sweepers:
            sweeper_type_eff = sweeper_type

            if global_rank == 0:
                if QI_ser in RADAU_METHODS + RK_METHODS:
                    if is_executed[QI_ser]:
                        continue

                    sweeper_type_eff = "fullyImplicitDAE" if QI_ser in RADAU_METHODS else "constrainedDAE"
                    is_executed[QI_ser] = True

                key_ser = f"{sweeper_type_eff}_{QI_ser}"

                if key_ser not in results:
                    results[key_ser] = {}

            if QI_ser in RADAU_METHODS + RK_METHODS:
                if global_rank == 0:
                    num_nodes_ref = num_processes[0]

                    runtime, solution_stats = compute_solution(
                        problem_name,
                        t0,
                        dt,
                        Tend,
                        num_nodes_ref,
                        QI_ser,
                        sweeper_type_eff,
                        use_mpi=False,
                        measure=True,
                    )

                    # Copy runtimes to all nodes in dict
                    for num_nodes in num_processes:
                        results[key_ser][num_nodes] = runtime

                    with open(path, "wb") as f:
                        dill.dump(results, f)

            else:
                for num_nodes in num_processes:
                    if global_rank == 0:
                        runtime, solution_stats = compute_solution(
                            problem_name,
                            t0,
                            dt,
                            Tend,
                            num_nodes,
                            QI_ser,
                            sweeper_type_eff,
                            use_mpi=False,
                            measure=True,
                        )

                        results[key_ser][num_nodes] = runtime

                        with open(path, "wb") as f:
                            dill.dump(results, f)

    # Dummy run to avoid overhead
    global_comm.Barrier()

    dt_dummy = 1e-4

    _ = compute_solution(
        problem_name,
        t0,
        dt=dt_dummy,
        Tend=t0 + dt_dummy,
        num_nodes=global_size,
        QI="MIN-SR-NS",
        sweeper_type="constrainedDAE",
        use_mpi=True,
        measure=False,
        comm=global_comm,
    )

    global_comm.Barrier()

    for sweeper_type in sweepers:
        for QI_par in QI_parallel_methods:
            key_par = f"{sweeper_type}_{QI_par}"
            if global_rank == 0:
                results[key_par] = {}

            for num_nodes in num_processes:
                if global_rank == 0:
                    results[key_par][num_nodes] = 0

                global_comm.Barrier()

                timing_run_full = run_test_and_split_communicator(
                    problem_name,
                    t0,
                    dt,
                    Tend,
                    global_comm,
                    global_rank,
                    num_nodes,
                    QI_par,
                    sweeper_type,
                    use_mpi=True,
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
    config_linear = get_configs(problem_name="LINEAR-TEST", config_type="scaling")
    run_mpi_test(**config_linear)
