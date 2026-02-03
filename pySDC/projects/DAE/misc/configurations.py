from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes


def get_configs(problem_name: str, config_type: str) -> dict:
    if config_type == "work_precision":
        hook_class = [LogGlobalErrorPostStep]
        num_nodes = 6
        sweepers = ["constrainedDAE", "semiImplicitDAE"]
        test_methods = ["EE", "IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard", "DOPRI5", "RadauIIA5", "RadauIIA7"]

        if problem_name in ["ANDREWS-SQUEEZER", "LINEAR-TEST"]:
            test_methods = ["EE", "IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard", "DOPRI5", "RadauIIA5", "RadauIIA7"]
        elif problem_name == "REACTION-DIFFUSION":
            # Only implicit methods since problem is stiff
            test_methods = ["IE", "LU", "MIN-SR-S", "RadauIIA5", "RadauIIA7"]
        else:
            raise NotImplementedError(f"Test methods for {problem_name} are not implemented!")

        config = {
            "hook_class": hook_class,
            "num_nodes": num_nodes,
            "problem_name": problem_name,
            "sweepers": sweepers,
            "test_methods": test_methods,
        }

    elif config_type == "scaling":
        QI_serial_methods = ["LU", "DOPRI5", "RadauIIA5", "RadauIIA7"]
        QI_parallel_methods = ["MIN-SR-NS", "MIN-SR-S"]

        sweepers = ["constrainedDAE", "semiImplicitDAE"]
        dt_list, _ = choose_time_step_sizes(problem_name=problem_name)

        hook_class = [LogEmbeddedErrorEstimate, LogGlobalErrorPostStep]

        if problem_name == "REACTION-DIFFUSION":
            QI_serial_methods = ["LU", "RadauIIA5", "RadauIIA7"]
            QI_parallel_methods = ["MIN-SR-S"]

        if problem_name == "ANDREWS-SQUEEZER":
            dt = dt_list[0]
        elif problem_name == "LINEAR-TEST":
            dt = dt_list[3]
        elif problem_name == "REACTION-DIFFUSION":
            dt = dt_list[2]

        config = {
            "hook_class": hook_class,
            "problem_name": problem_name,
            "dt": dt,
            "sweepers": sweepers,
            "QI_serial_methods": QI_serial_methods,
            "QI_parallel_methods": QI_parallel_methods,
        }

    elif config_type == "breakeven":
        config = get_configs(problem_name=problem_name, config_type="scaling")
        config.pop("hook_class", None)
        config.pop("QI_serial_methods", None)

    return config
