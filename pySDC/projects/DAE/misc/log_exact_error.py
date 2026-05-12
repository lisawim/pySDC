from pySDC.core.hooks import Hooks


class LogExactError(Hooks):
    """
    Store the exact error from convergence controller at the end of each step as "exact_error".
    """

    def post_step(self, step, level_number):
        """
        Record exact error.

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_step(step, level_number)

        L = step.levels[level_number]

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time + L.dt,
            level=L.level_index,
            iter=iter,
            sweep=L.status.sweep,
            type="exact_error",
            value=L.status.exact_error,
        )
