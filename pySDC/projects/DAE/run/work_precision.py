import sys
import dill
import os
import subprocess

from pySDC.projects.DAE.misc.configurations import LinearTestWorkPrecision


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


def run_all_simulations(config):
    python_exec = sys.executable

    output_dir = "data" + "/" + f"{config.problem_name}" + "/" + "results"
    os.makedirs(output_dir, exist_ok=True)

    n_steps_list = [2, 5, 10, 20, 50, 100, 200, 500]
    dt_list = [config.Tend / n_steps for n_steps in n_steps_list]

    all_stats = {}

    fname = f"results_experiment_{config.num_nodes}.pkl"
    path = os.path.join(output_dir, fname)

    if not os.path.exists(path):
        with open(path, "wb") as f:
            dill.dump(all_stats, f)
            f.flush()
            os.fsync(f.fileno())

    assert os.path.getsize(path) > 0

    args = {"problem_name": config.problem_name}

    for sweeper_type in config.sweepers:

        for QI in config.test_methods:
            key = f"{sweeper_type}_{QI}"
            all_stats[key] = {}

            if QI in config.radau_methods:
                sweeper_type = "fullyImplicitDAE"

            use_mpi = True if QI in config.qDeltas_parallel else False

            args.update({
                "t0": config.t0,
                "dt_list": dt_list,
                "Tend": config.Tend,
                "use_mpi": use_mpi,
                "QI": QI,
                "sweeper_type": sweeper_type,
                "problem_name": config.problem_name,
                "num_nodes": str(config.num_nodes),
                "output_dir": output_dir,
            })

            args_list = build_args_list(args, config.hook_class)

            cmd = (
                ["mpiexec", "-n", str(config.num_nodes), python_exec, "run_single_experiment.py"] + args_list
                if use_mpi
                else
                [python_exec, "run_single_experiment.py"] + args_list
            )

            env = os.environ.copy()
            #env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
            subprocess.run(cmd, check=True, env=env, close_fds=True)


if __name__ == "__main__":
    config = LinearTestWorkPrecision()
    run_all_simulations(config)
