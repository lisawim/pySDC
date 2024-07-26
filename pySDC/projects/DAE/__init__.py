from .misc import (
    DAEMesh,
    LogGlobalErrorPostStepDifferentialVariable,
    LogGlobalErrorPostStepAlgebraicVariable,
    ptype_dae,
)
from .problems import (
    DiscontinuousTestDAE,
    pendulum_2d,
    problematic_f,
    simple_dae_1,
    synchronous_machine_infinite_bus,
    one_transistor_amplifier,
    two_transistor_amplifier,
    WSCC9BusSystem,
    get_initial_Ybus,
    get_event_Ybus,
)
from .run import run, check_order, runSimulation, setup, mainFullyImplicitDAE, mainSyncMachine
from .sweepers import (
    fully_implicit_DAE,
    fully_implicit_DAE_MPI,
    RungeKuttaDAE,
    BackwardEulerDAE,
    TrapezoidalRuleDAE,
    EDIRK4DAE,
    DIRK43_2DAE,
    SemiImplicitDAE,
    SemiImplicitDAEMPI,
    SweeperDAEMPI,
)

misc_all = [
    'DAEMesh',
    'LogGlobalErrorPostStepDifferentialVariable',
    'LogGlobalErrorPostStepAlgebraicVariable',
    'ptype_dae',
]

problems_all = [
    'DiscontinuousTestDAE',
    'pendulum_2d',
    'problematic_f',
    'simple_dae_1',
    'synchronous_machine_infinite_bus',
    'one_transistor_amplifier',
    'two_transistor_amplifier',
    'WSCC9BusSystem',
    'get_initial_Ybus',
    'get_event_Ybus',
]

run_all = ['run', 'check_order', 'runSimulation', 'setup', 'mainFullyImplicitDAE', 'mainSyncMachine']

sweeper_all = [
    'fully_implicit_DAE',
    'fully_implicit_DAE_MPI',
    'RungeKuttaDAE',
    'BackwardEulerDAE',
    'TrapezoidalRuleDAE',
    'EDIRK4DAE',
    'DIRK43_2DAE',
    'SemiImplicitDAE',
    'SemiImplicitDAEMPI',
    'SweeperDAEMPI',
]

__all__ = []
__all__ += misc_all + problems_all + run_all + sweeper_all
