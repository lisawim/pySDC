import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import (
    LinearTestSPP,
    LinearTestSPPMinion,
    DiscontinuousTestSPP,
)

from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded
from pySDC.projects.DAE.problems.LinearTestDAEMinion import (
    LinearTestDAEMinionEmbedded,
    LinearTestDAEMinionConstrained,
)
from pySDC.projects.DAE.problems.LinearTestDAE import (
    LinearTestDAEEmbedded,
    LinearTestDAEConstrained,
)
from pySDC.projects.DAE.problems.chatGPTDAE import (
    chatGPTDAEEmbedded,
    chatGPTDAEConstrained,
)
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAEEmbedded

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation
from pySDC.projects.DAE.misc.HookClass_DAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable

from pySDC.helpers.stats_helper import get_sorted


def main():
    problems = [
        LinearTestSPP,
        # LinearTestDAEEmbedded,
        # DiscontinuousTestSPP,
        # DiscontinuousTestDAEEmbedded
    ]
    sweepers = [generic_implicit, genericImplicitEmbedded]

    if sweepers[1].__name__.startswith("genericImplicit"):
        startsWith = True

    assert startsWith, "To store files correctly, set one of the DAE sweeper into list at index 1!"

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    conv_type = 'increment'

    # parameters for convergence
    maxiter = 6

    # hook class to be used
    hook_class = {
        'generic_implicit': [LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation],
        'genericImplicitEmbedded': [
            LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable
        ],
    }

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-12

    eps_list = [10 ** (-m) for m in range(2, 12)]
    epsValues = {
        'generic_implicit': eps_list,
        'genericImplicitEmbedded': [0.0],
    }

    t0 = 0.0
    Tend = 1.0
    # nSteps = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])
    # dtValues = (Tend - t0) / nSteps
    dtValues = np.logspace(-2.5, 0.0, num=40)

    colors = [
        'lightsalmon',
        'lightcoral',
        'indianred',
        'firebrick',
        'brown',
        'maroon',
        'lightgray',
        'darkgray',
        'gray',
        'dimgray',
    ]
    colors = list(reversed(colors))