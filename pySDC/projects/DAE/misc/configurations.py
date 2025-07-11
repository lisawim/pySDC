from pySDC.core.errors import ParameterError
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
from pySDC.projects.DAE.misc.hooksDAE import (
    LogGlobalErrorPostStepDifferentialVariable,
    LogGlobalErrorPostStepAlgebraicVariable,
)


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

    
class LinearTestWorkPrecision(LinearTestBaseConfig):
    def __init__(self):
        super().__init__()

        self.num_nodes = 6
        self.hook_class = [
            LogGlobalErrorPostStep,
            LogGlobalErrorPostStepDifferentialVariable,
            LogGlobalErrorPostStepAlgebraicVariable,
        ]

        self._sweepers = [
            "constrainedDAE", "fullyImplicitDAE", "semiImplicitDAE"
        ]


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
