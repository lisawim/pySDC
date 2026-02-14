import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize("num_nodes", [2, 3, 4])
@pytest.mark.parametrize("QI", ["IE", "LU"])
def test_versions(num_nodes, QI, launch=True):
    r"""
    Make a test if the result matches between the MPI and non-MPI versions of a sweeper.
    Tests solution at the right end point and the residual.

    Parameters
    ----------
    num_nodes : int
        Number of collocation nodes to use.
    residual_type : str
        Type of residual computation.
    semi_implicit : bool
        If True, semi-implicit sweeper is used.
    index_case : int
        Case of DAE index. Choose either between :math:`1` or :math:`2`.
    initial_guess : str
        Type of initial guess for simulation.
    launch : bool
        If yes, it will launch `mpirun` with the required number of processes
    """

    problem_name = "LINEAR-TEST"
    sweeper_type = "constrainedDAE"

    t0 = 0.0
    dt = 0.1
    Tend = t0 + dt

    use_mpi_grouped = False
    return_uend = True

    if launch:
        import os
        import subprocess

        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))

        my_env = os.environ.copy()
        my_env["PYTHONPATH"] = os.pathsep.join([repo_root, my_env.get("PYTHONPATH", "")])
        my_env["COVERAGE_PROCESS_START"] = "pyproject.toml"
        cmd = f"mpirun -np {num_nodes} python {__file__} --test_versions {num_nodes} {QI}".split()

        p = subprocess.Popen(cmd, env=my_env, cwd=".")

        p.wait()
        assert p.returncode == 0, "ERROR: did not get return code 0, got %s with %2i processes" % (
            p.returncode,
            num_nodes,
        )
    else:
        import numpy as np
        from pySDC.projects.DAE.run.utils import compute_solution

        _, uend_mpi, _ = compute_solution(
            problem_name=problem_name,
            t0=t0,
            dt=dt,
            Tend=Tend,
            num_nodes=int(num_nodes),
            QI=QI,
            sweeper_type=sweeper_type,
            use_mpi=True,
            use_mpi_grouped=use_mpi_grouped,
            return_uend=return_uend,
        )

        _, uend, _ = compute_solution(
            problem_name=problem_name,
            t0=t0,
            dt=dt,
            Tend=Tend,
            num_nodes=int(num_nodes),
            QI=QI,
            sweeper_type=sweeper_type,
            use_mpi=False,
            use_mpi_grouped=use_mpi_grouped,
            return_uend=return_uend,
        )

        assert np.allclose(uend_mpi, uend, atol=1e-14), "Got different solutions at end point!"


if __name__ == "__main__":
    import sys

    if "--test_versions" in sys.argv:
        test_versions(sys.argv[-2], sys.argv[-1], launch=False)
