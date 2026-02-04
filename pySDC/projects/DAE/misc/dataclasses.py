import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QEndErrorResult:
    ind: int
    t_ref: float
    qend: np.ndarray
    qend_ref: np.ndarray
    qend_max_final_err: float


@dataclass(frozen=True)
class ScalingRunStats:
    """
    Result container for MPI scaling / work-precision tests.
    """

    t_wall: float
    e_global_post_step: list
    niter_mean: Optional[float] = None
    e_emb_post_step: Optional[float] = None
    qend_max_final_error: Optional[float] = None  # optional, for Andrews'


@dataclass(frozen=True)
class WorkPrecisionResult:  # TODO: Embed dataclass in run_single_experiment.py
    dt_list: list[float]
    wc_times: list[float]
    all_max_global_error: list[float]
    q_max_final_error: Optional[list[float]] = None
