from pySDC.core.errors import ParameterError
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep


def get_configs(problem_name, config_type):
    if config_type == "work_precision":
        test_methods = ["IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard", "RadauIIA5", "RadauIIA7"]

        if problem_name == "ANDREWS-SQUEEZER":
            sweepers = ["constrainedDAE", "semiImplicitDAE"]

            config = {
                "hook_class": [LogGlobalErrorPostStep],
                "num_nodes": 6,
                "problem_name": problem_name,
                "sweepers": sweepers,
                "test_methods": test_methods,
            }

        elif problem_name == "LINEAR-TEST":
            sweepers = ["constrainedDAE", "fullyImplicitDAE", "semiImplicitDAE"]

            config = {
                "hook_class": [LogGlobalErrorPostStep],
                "num_nodes": 6,
                "problem_name": problem_name,
                "sweepers": sweepers,
                "test_methods": test_methods,
            }

    return config


class BaseConfig:
    def __init__(self):
        self._qDeltas_parallel = ["MIN-SR-NS", "MIN-SR-S"]
        self._qDeltas_serial = ["IE", "LU", "Picard"]

        self._test_methods = None

        self._sweepers = [
            "constrainedDAE", "embeddedDAE", "fullyImplicitDAE", "semiImplicitDAE"
        ]

        self._radau_methods = ["RadauIIA5", "RadauIIA7", "RadauIIA9"]

        self.num_nodes = None

    @property
    def qDeltas(self):
        return self.qDeltas_serial + self.qDeltas_parallel

    # TODO: It is maybe unnecessary to have both qDeltas an qDeltas_experiment..
    @property
    def test_methods(self):
        return self._test_methods

    @property
    def qDeltas_parallel(self):
        return self._qDeltas_parallel
    
    @property
    def qDeltas_serial(self):
        return self._qDeltas_serial
    
    @property
    def sweepers(self):
        return self._sweepers

    @property
    def radau_methods(self):
        return self._radau_methods


class LinearTestBaseConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self._test_methods = ["IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard", "RadauIIA5", "RadauIIA7"]

        self.t0 = 0.0
        self.Tend = 1.0
        self.problem_name = "LINEAR-TEST"

class AndrewsBaseConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self._test_methods = ["IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard", "RadauIIA5", "RadauIIA7"]

        self.t0 = 0.0
        self.Tend = 0.03
        self.problem_name = "ANDREWS-SQUEEZER"


class LinearTestScaling(LinearTestBaseConfig):
    def __init__(self):
        super().__init__()

        self.dt = 1e-1

        self.num_processes = None
        self.hook_class = []

        self.QI_ser = "LU"

    def set_num_processes(self, global_size):
        self.num_processes = range(2, global_size + 1)

    def check_global_comm_size(self, global_size):
        if global_size > self.num_processes[-1]:
            raise ParameterError(
                f"{self.__class__.__name__} allows maximum {self.num_processes[-1]}, but global size is {global_size}"
            )
        
class LinearTestScaling_semi_implicit(LinearTestScaling):
    def __init__(self):
        super().__init__()

        self._sweepers = ["constrainedDAE", "semiImplicitDAE"]

    def check_global_comm_size(self, global_size):
        pass
