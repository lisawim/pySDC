import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class QEndErrorResult:
    ind: int
    t_ref: float
    qend: np.ndarray
    qend_ref: np.ndarray
    qend_max_final_err: float


@dataclass(frozen=True)
class WorkPrecisionResult:  # TODO: Embed dataclass in run_single_experiment.py
    dt_list: List[float]
    wc_times: List[float]
    all_max_global_error: List[float]
    q_max_final_error: Optional[List[float]] = None
