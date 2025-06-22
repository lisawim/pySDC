import numpy as np
import dill
import os
import subprocess

from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep

from pySDC.projects.DAE import QI_PARALLEL, ExperimentConfig


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


def run_all_simulations(config=ExperimentConfig, problem_name="LINEAR-TEST"):
    output_dir = "data" + "/" + f"{problem_name}" + "/" + "results"
    os.makedirs(output_dir, exist_ok=True)

    t0, Tend = 0.0, 1.0
    n_steps_list = [5, 10, 20, 50, 100, 200, 500, 1000]
    dt_list = [Tend / n_steps for n_steps in n_steps_list]

    hook_class = [LogGlobalErrorPostStep]

    all_stats = {}

    args = {"problem_name": problem_name}

    for sweeper_type in config.sweeper_type_list:

        for QI in config.qDelta_list:
            key = f"{sweeper_type}_{QI}"
            all_stats[key] = {}

            if QI in ["RadauIIA5", "RadauIIA7"]:
                sweeper_type = "fullyImplicitDAE"

            use_mpi = True if QI in QI_PARALLEL else False

            args.update({
                "t0": t0,
                "dt_list": dt_list,
                "Tend": Tend,
                "use_mpi": use_mpi,
                "QI": QI,
                "sweeper_type": sweeper_type,
                "problem_name": problem_name,
                "num_nodes": str(config.num_nodes),
                "output_dir": output_dir,
            })

            args_list = build_args_list(args, hook_class)

            cmd = (
                ["mpiexec", "-n", str(config.num_nodes), "python3", "run_single_experiment.py"] + args_list
                if use_mpi
                else
                ["python3", "run_single_experiment.py"] + args_list
            )

            env = os.environ.copy()
            subprocess.run(cmd, check=True, env=env, close_fds=True)

            fname = f"result_{sweeper_type}_{QI}_{config.num_nodes}.pkl"
            path = os.path.join(output_dir, fname)
            with open(path, "rb") as f:
                res = dill.load(f)

            all_stats[key] = res

    fname = f"results_experiment_{config.num_nodes}.pkl"
    path = os.path.join(output_dir, fname)
    with open(path, "wb") as f:
        dill.dump(all_stats, f)


# Script is started with "python3 work_precision.py"
if __name__ == "__main__":
    run_all_simulations(problem_name="LINEAR-TEST")
