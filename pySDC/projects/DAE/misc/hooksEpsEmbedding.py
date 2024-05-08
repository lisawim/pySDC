from pySDC.core.Hooks import hooks


class LogGlobalErrorPostStep(hooks):
    r"""
    Hook class to log the error to the output generated by the sweeper after
    each time step.

    Here, the error is logged for all components **except** for the last one,
    which can be logged by using ``LogGlobalErrorPostStepAlgebraicVariable``.
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
        P = L.prob

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        L.sweep.compute_end_point()

        # compute and save errors
        # Note that the component from which the error is measured is specified here
        upde = P.u_exact(step.time + step.dt)
        e_global = abs(upde[:-1] - L.uend[:-1])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_global_post_step',
            value=e_global,
        )


class LogGlobalErrorPostStepPerturbation(hooks):
    """
    Logs the global error in the variable with perturbation parameter :math:`\varepsilon`
    that represents the algebraic variable.

    Here, the error is logged **only** for the last component.
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

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        e_global_algebraic = abs(upde[-1] - L.uend[-1])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_global_algebraic_post_step',
            value=e_global_algebraic,
        )


class LogGlobalErrorPostIter(hooks):
    r"""
    Hook class to log the error to the output generated by the sweeper after
    each time step.

    Here, the error is logged for all components **except** for the last one,
    which can be logged by using ``LogGlobalErrorPostStepAlgebraicVariable``.
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
        P = L.prob

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        L.sweep.compute_end_point()

        # compute and save errors
        # Note that the component from which the error is measured is specified here
        upde = P.u_exact(step.time + step.dt)
        e_global = abs(upde[:-1] - L.uend[:-1])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_global_post_iter',
            value=e_global,
        )


class LogGlobalErrorPostIterPerturbation(hooks):
    """
    Logs the global error in the variable with perturbation parameter :math:`\varepsilon`
    that represents the algebraic variable.

    Here, the error is logged **only** for the last component.
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

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        e_global_algebraic = abs(upde[-1] - L.uend[-1])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='e_global_algebraic_post_iter',
            value=e_global_algebraic,
        )