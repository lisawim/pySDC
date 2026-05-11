import time
from pySDC.core.hooks import Hooks


class MyCPUTimings(Hooks):
    """Logs iteration timing at L.time + L.dt instead of L.time."""

    def __init__(self):
        super().__init__()

        self.__t0_sweep = None
        self.__t0_iteration = None
        self.__t0_step = None
        self.__t1_sweep = None
        self.__t1_iteration = None
        self.__t1_step = None

    def _compute_time_elapsed(self, event_after, event_before):
        return event_after - event_before

    def _get_event(self):
        return time.perf_counter()

    def pre_step(self, step, level_number):
        """
        Hook called before each step

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_step(step, level_number)
        self.__t0_step = self._get_event()

    def pre_iteration(self, step, level_number):
        """
        Default routine called before iteration starts

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().pre_iteration(step, level_number)
        self.__t0_iteration = self._get_event()

    def pre_sweep(self, step, level_number):
        super().pre_sweep(step, level_number)
        self.__t0_sweep = self._get_event()

    def post_sweep(self, step, level_number):
        super().post_sweep(step, level_number)
        self.__t1_sweep = self._get_event()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f"timing_post_sweep",
            value=self._compute_time_elapsed(self.__t1_sweep, self.__t0_sweep),
        )

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)
        self.__t1_iteration = self._get_event()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f"timing_post_iteration",
            value=self._compute_time_elapsed(self.__t1_iteration, self.__t0_iteration),
        )

    def post_step(self, step, level_number):
        """
        Default routine called after each step or block

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number
        """
        super().post_step(step, level_number)
        self.__t1_step = self._get_event()

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f"timing_post_step",
            value=self._compute_time_elapsed(self.__t1_step, self.__t0_step),
        )
