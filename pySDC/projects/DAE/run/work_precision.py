import sys
import dill
import os
import subprocess

from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE.misc.configurations import get_configs
from pySDC.projects.DAE.misc.methods_config import QI_PARALLEL, RADAU_METHODS, RK_METHODS


def build_args_list(args, hook_class):
    """Build args list to pass to CLI."""

    args_list = []
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                args_list.append(f"--{k}")
        elif isinstance(v, list):
            args_list.append(f"--{k}")
            for item in v:
                args_list.append(str(item))
        else:
            args_list.append(f"--{k}={v}")

    for hook in hook_class:
        args_list.append(f"--hook_class={hook.__module__}.{hook.__name__}")

    return args_list


def run_all_simulations(hook_class, num_nodes, nsweeps, problem_name, sweepers, test_methods, **kwargs):
    python_exec = sys.executable

    output_dir = "data" + "/" + f"{problem_name}" + "/" + "results"
    os.makedirs(output_dir, exist_ok=True)

    t0 = 0.0
    dt_list, Tend = choose_time_step_sizes(problem_name)

    all_stats = {}

    fname = f"results_experiment_{num_nodes}.pkl"
    path = os.path.join(output_dir, fname)

    if not os.path.exists(path):
        with open(path, "wb") as f:
            dill.dump(all_stats, f)
            f.flush()
            os.fsync(f.fileno())

    assert os.path.getsize(path) > 0

    args = {"problem_name": problem_name}

    is_executed = {name: False for name in RADAU_METHODS + RK_METHODS}

    for sweeper_type in sweepers:
        for QI in test_methods:
            if QI in RADAU_METHODS + RK_METHODS:
                if is_executed[QI]:
                    continue

                sweeper_type = "fullyImplicitDAE" if QI in RADAU_METHODS else "constrainedDAE"
                is_executed[QI] = True

            key = f"{sweeper_type}_{QI}"
            all_stats[key] = {}

            use_mpi = True if QI in QI_PARALLEL else False

            args.update(
                {
                    "t0": t0,
                    "dt_list": dt_list,
                    "Tend": Tend,
                    "use_mpi": use_mpi,
                    "QI": QI,
                    "sweeper_type": sweeper_type,
                    "problem_name": problem_name,
                    "num_nodes": str(num_nodes),
                    "nsweeps": str(nsweeps),
                    "output_dir": output_dir,
                }
            )

            args_list = build_args_list(args, hook_class)

            cmd = (
                ["mpiexec", "-n", str(num_nodes), python_exec, "run_single_experiment.py"] + args_list
                if use_mpi
                else [python_exec, "run_single_experiment.py"] + args_list
            )

            env = os.environ.copy()
            # env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
            subprocess.run(cmd, check=True, env=env, close_fds=True)


if __name__ == "__main__":
    config_andrews = get_configs(problem_name="ANDREWS-SQUEEZER", config_type="work_precision")
    run_all_simulations(**config_andrews)
