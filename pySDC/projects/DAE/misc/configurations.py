from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep


def get_configs(problem_name, config_type):
    if config_type == "work_precision":
        sweepers = ["constrainedDAE", "semiImplicitDAE"]
        test_methods = ["EE", "IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard", "DOPRI5", "RadauIIA5", "RadauIIA7"]

        if problem_name == "ANDREWS-SQUEEZER":
            config = {
                "hook_class": [LogGlobalErrorPostStep],
                "num_nodes": 6,
                "problem_name": problem_name,
                "sweepers": sweepers,
                "test_methods": test_methods,
            }

        elif problem_name == "LINEAR-TEST":
            config = {
                "hook_class": [LogGlobalErrorPostStep],
                "num_nodes": 6,
                "problem_name": problem_name,
                "sweepers": sweepers,
                "test_methods": test_methods,
            }

        elif problem_name == "REACTION-DIFFUSION":
            # Only implicit methods since problem is stiff
            test_methods = ["IE", "LU", "MIN-SR-S", "RadauIIA5", "RadauIIA7"]

            config = {
                "hook_class": [LogGlobalErrorPostStep],
                "num_nodes": 6,
                "problem_name": problem_name,
                "sweepers": sweepers,
                "test_methods": test_methods,
            }
    elif config_type == "scaling":
        QI_serial_methods = ["LU", "DOPRI5", "RadauIIA5", "RadauIIA7"]
        QI_parallel_methods = ["MIN-SR-NS", "MIN-SR-S"]
        sweepers = ["constrainedDAE", "semiImplicitDAE"]

        if problem_name == "LINEAR-TEST":
            config = {
                "problem_name": problem_name,
                "dt": 0.1,
                "sweepers": sweepers,
                "QI_serial_methods": QI_serial_methods,
                "QI_parallel_methods": QI_parallel_methods,
            }

        elif problem_name == "ANDREWS-SQUEEZER":
            config = {
                "problem_name": problem_name,
                "dt": 1e-3,
                "sweepers": sweepers,
                "QI_serial_methods": QI_serial_methods,
                "QI_parallel_methods": QI_parallel_methods,
            }
        elif problem_name == "REACTION-DIFFUSION":
            QI_serial_methods = ["LU", "RadauIIA5", "RadauIIA7"]
            QI_parallel_methods = ["MIN-SR-S"]

            config = {
                "problem_name": problem_name,
                "dt": 0.1,
                "sweepers": sweepers,
                "QI_serial_methods": QI_serial_methods,
                "QI_parallel_methods": QI_parallel_methods,
            }

    return config
