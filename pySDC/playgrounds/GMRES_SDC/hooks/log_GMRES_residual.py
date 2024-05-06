from pySDC.core.Hooks import hooks


class LogGMRESResidualPostStep(hooks):
    """
    Hooks class to log the relative residual from GMRES
    (which is implemented via a SciPy routine) after each step.
    """

    def post_step(self, step, level_number):
        r"""
        Default routine called after each step.

        Parameters
        ----------
        step : pySDC.core.Step
            Current step.
        level_number : pySDC.core.level
            Current level number.
        """

        super().post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        pr_norm = L.sweep.pr_norm

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='relative_residual_gmres_post_step',
            value=pr_norm,
        )


class LogGMRESResidualPostIter(hooks):
    """
    Hooks class to log the relative residual from GMRES
    (which is implemented via a SciPy routine) after each iteration.
    """

    def post_iteration(self, step, level_number):
        r"""
        Default routine called after each step.

        Parameters
        ----------
        step : pySDC.core.Step
            Current step.
        level_number : pySDC.core.level
            Current level number.
        """

        super().post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        pr_norm = L.sweep.pr_norm

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='relative_residual_gmres_post_iter',
            value=pr_norm,
        )