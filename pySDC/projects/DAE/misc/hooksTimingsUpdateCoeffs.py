import numpy as np
from pySDC.core.hooks import Hooks


class UpdateCoeffsTimingsRun(Hooks):
    def post_run(self, step, level_number):
        """
        Default routine called after each run

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_run(step, level_number)

        L = step.levels[level_number]

        elapsed_time_update_coeffs = L.sweep.elapsed_time_update_coeffs

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="timing_update_coeffs_run",
            value=elapsed_time_update_coeffs,
        )


class UpdateCoeffsTimingsIter(Hooks):
    def post_iteration(self, step, level_number):
        """
        Default routine called after each iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_iteration(step, level_number)

        L = step.levels[level_number]

        elapsed_time_update_coeffs = L.sweep.elapsed_time_update_coeffs

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="timing_update_coeffs_iteration",
            value=elapsed_time_update_coeffs,
        )
