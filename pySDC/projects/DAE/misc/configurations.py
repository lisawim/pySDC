from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostIter, LogGlobalErrorPostStep
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_work import LogWork
from pySDC.projects.DAE.misc.hooksDAE import LogExactError, MyCPUTimings, MyLogSolutionAfterIteration
from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import LogPositionErrorEndPostIteration
from pySDC.projects.DAE.problems.reactionDiffusionPDAE import LogAchievedNewtonTolerancePostStep
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes


def _ensure_known_problem(problem_name: str) -> None:
    if problem_name not in {"ANDREWS-SQUEEZER", "LINEAR-TEST", "REACTION-DIFFUSION"}:
        raise NotImplementedError(f"Unknown/unsupported problem_name={problem_name!r}")


def get_configs(problem_name: str, config_type: str, nsweeps: int = 1) -> dict:
    _ensure_known_problem(problem_name)

    if config_type == "work_precision":
        num_nodes = 6
        config = {
            "hook_class": [LogGlobalErrorPostStep],
            "num_nodes": 6,
            "nsweeps": nsweeps,
            "problem_name": problem_name,
            "sweepers": ["constrainedDAE", "semiImplicitDAE"],
            "test_methods": [
                "EE",
                "IE",
                "LU",
                "MIN-SR-NS",
                "MIN-SR-S",
                # "MIN-SR-FLEX",
                "Picard",
                "DOPRI5",
                "RadauIIA5",
                "RadauIIA7",
            ],
        }
        assert config["nsweeps"] <= num_nodes, "nsweeps must be <= num_nodes for work-precision config"

        # Only implicit methods since problem is stiff
        if problem_name == "REACTION-DIFFUSION":
            config["test_methods"] = ["IE", "LU", "MIN-SR-S", "RadauIIA5", "RadauIIA7"]

    elif config_type == "scaling":
        dt_list, _ = choose_time_step_sizes(problem_name=problem_name)

        config = {
            "hook_class": [
                MyCPUTimings,
                LogEmbeddedErrorEstimate,
                LogGlobalErrorPostIter,
                LogGlobalErrorPostStep,
                LogSolution,
                MyLogSolutionAfterIteration,
                LogWork,
            ],
            "problem_name": problem_name,
            "sweepers": ["constrainedDAE", "semiImplicitDAE"],
            "QI_serial_methods": ["LU", "DOPRI5", "RadauIIA5", "RadauIIA7"],
            "QI_parallel_methods": ["MIN-SR-NS", "MIN-SR-S"],
            "dt": None,
        }

        # problem-specific overrides
        if problem_name == "ANDREWS-SQUEEZER":
            config["hook_class"] += [LogPositionErrorEndPostIteration]

        if problem_name == "REACTION-DIFFUSION":
            config["hook_class"] += [LogAchievedNewtonTolerancePostStep]

        if problem_name == "REACTION-DIFFUSION":
            config["QI_serial_methods"] = ["LU", "RadauIIA5", "RadauIIA7"]
            config["QI_parallel_methods"] = ["MIN-SR-S"]

        dt_index_by_problem = {
            "ANDREWS-SQUEEZER": 0,
            "LINEAR-TEST": 3,
            "REACTION-DIFFUSION": 1,
        }
        config["dt"] = dt_list[dt_index_by_problem[problem_name]]

    elif config_type == "speedup_at_accuracy":
        config = get_configs(problem_name=problem_name, config_type="scaling")
        config["hook_class"] = [
            MyCPUTimings, LogEmbeddedErrorEstimate, LogExactError, LogGlobalErrorPostStep
        ]
        if problem_name == "REACTION-DIFFUSION":
            config["hook_class"] += [LogAchievedNewtonTolerancePostStep]

        config["stop_at_accuracy_for_speedup"] = True

    elif config_type == "breakeven":
        config = get_configs(problem_name=problem_name, config_type="scaling")
        config.pop("QI_serial_methods", None)

        if problem_name in ["ANDREWS-SQUEEZER", "LINEAR-TEST"]:
            config["QI_parallel_methods"] = ["MIN-SR-NS"]

    else:
        raise NotImplementedError(f"Unknown config_type={config_type!r}")

    return config
