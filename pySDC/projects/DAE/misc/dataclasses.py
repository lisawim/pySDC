import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QEndErrorResult:
    """
    Data container for the final error of qend, used in Andrews' test case.
    """
    ind: int
    t_ref: float
    qend: np.ndarray
    qend_ref: np.ndarray
    qend_error: float


@dataclass(frozen=True)
class ScalingRunStats:
    """
    Result container for MPI scaling / work-precision tests.
    """

    t_wall: float
    e_global_steps: list
    e_embedded_steps: Optional[float] = None
    t_cpu_steps: Optional[np.ndarray] = None
    qend_error: Optional[float] = None
    niter_steps: Optional[float] = None
    niter_mean: Optional[float] = None
    t_cpu_one_step: Optional[np.ndarray] = None
    e_global_one_step: Optional[np.ndarray] = None
    work_newton_steps: Optional[np.ndarray] = None
    newton_tol_achieved_steps: Optional[np.ndarray] = None


@dataclass(frozen=True)
class SpeedupAccuracyRunStats:
    """
    Result container for speedup tests where it is stopped if certain accuracy is achieved.
    """

    t_wall_stop_at_acc: float
    e_global_steps: list
    e_embedded_steps: Optional[float] = None
    e_exact_steps: Optional[float] = None
    t_cpu_steps: Optional[np.ndarray] = None
    qend_error: Optional[float] = None
    niter_mean: Optional[float] = None
    niter_steps: Optional[float] = None
    work_newton_steps: Optional[np.ndarray] = None
    newton_tol_achieved_steps: Optional[np.ndarray] = None


@dataclass(frozen=True)
class WorkPrecisionResult:  # TODO: Embed dataclass in run_single_experiment.py
    """
    Result container for work-precision tests.
    """
    dt_list: list[float]
    wc_times: list[float]
    all_max_global_error: list[float]
    q_max_final_error: Optional[list[float]] = None
