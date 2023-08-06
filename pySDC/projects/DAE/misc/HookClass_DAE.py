from pySDC.core.Hooks import hooks


class approx_solution_hook(hooks):
    """
    Hook class to add the approximate solution to the output generated by the sweeper after each time step
    """

    def __init__(self):
        """
        Initialization routine for the custom hook
        """
        super(approx_solution_hook, self).__init__()

    def post_step(self, step, level_number):
        """
        Default routine called after each step
        Args:
            step: the current step
            level_number: the current level number
        """

        super(approx_solution_hook, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='approx_solution',
            value=L.uend,
        )


class error_hook(hooks):
    """
    Hook class to add the approximate solution to the output generated by the sweeper after each time step
    """

    def __init__(self):
        """
        Initialization routine for the custom hook
        """
        super(error_hook, self).__init__()

    def post_step(self, step, level_number):
        """
        Default routine called after each step
        Args:
            step: the current step
            level_number: the current level number
        """

        super(error_hook, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        L.sweep.compute_end_point()

        # compute and save errors
        # Note that the component from which the error is measured is specified here
        upde = P.u_exact(step.time + step.dt)
        err = abs(upde[0] - L.uend[0])
        # err = abs(upde[4] - L.uend[4])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='error_post_step',
            value=err,
        )
        err_z = abs(upde[1] - L.uend[1])
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='error_post_step_alg',
            value=err_z,
        )
