from .utils import (
    checkLogDistribution,
    computeNormalityDeviation,
    computeSolution,
    getErrors,
    getCollocationMatrix,
    getEndTime,
    getIterationMatrix,
    getJacobianMatrix,
    getRHS,
    getMaxVal,
    getMinVal,
    getWork,
    roundDownToPreviousBase10,
    roundUpToNextBase10,
    roundUpToNextX,
)
from .utilsPlot import getColor, getColorQI, getLabel, get_linestyle, get_linestyle_QI, getMarker, getMarkerQI, Plotter

QI_SERIAL = ["IE", "LU", "MIN-SR-S", "MIN-SR-NS", "MIN-SR-FLEX"]

QI_PARALLEL = ["MIN-SR-S", "MIN-SR-NS", "MIN-SR-FLEX"]

__all__ = [
    "checkLogDistribution",
    "computeNormalityDeviation",
    "computeSolution",
    "getErrors",
    "getCollocationMatrix",
    "getColor",
    "getColorQI",
    "getEndTime",
    "getIterationMatrix",
    "getJacobianMatrix",
    "getRHS",
    "getWork",
    "getLabel",
    "get_linestyle",
    "get_linestyle_QI",
    "getMarker",
    "getMarkerQI",
    "getMaxVal",
    "getMinVal",
    "Plotter",
    # "QI_SERIAL",
    # "QI_PARALLEL",
    "roundDownToPreviousBase10",
    "roundUpToNextX",
    "roundUpToNextBase10",
]
