import time
from pySDC.core.hooks import Hooks


class LogExactError(Hooks):
    """
    Store the exact error from convergence controller at the end of each step as "exact_error".
    """

    def post_step(self, step, level_number, appendix=''):
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


class MyLogSolutionAfterIteration(Hooks):
    """
    Store the solution at the end of each iteration as "u_post_iteration".
    """

    def post_iteration(self, step, level_number):
        """
        Record solution at the end of the iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="u_post_iteration",
            value=L.uend,
        )


class LogAbsValuePreIterAlgebraicConstraints(Hooks):
    """
    Hook class to log the absolute value of algebraic constraints of
    a semi-explicit DAE after prediction. Requires the implementation of
    a algebraic_constraints() method.
    """

    def pre_iteration(self, step, level_number):
        r"""
        Default routine called before each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().pre_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        # L.sweep.compute_end_point()

        # Compute value of algebraic equation(s)
        g = P.algebraic_constraints(L.u[-1], step.time + step.dt)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="abs_g_pre_iteration",
            value=abs(g),
        )


class LogAbsValuePostIterAlgebraicConstraints(Hooks):
    """
    Hook class to log the absolute value of algebraic constraints of
    a semi-explicit DAE after each iteration. Requires the implementation of
    a algebraic_constraints() method.
    """

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        # Compute value of algebraic equation(s)
        g = P.algebraic_constraints(L.uend, step.time + step.dt)

        g_val_newton = P.g_val_newton

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="abs_g_post_iteration",
            value=abs(g),
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="abs_g_newton_post_iteration",
            value=g_val_newton,
        )


class LogAbsValuePostStepAlgebraicConstraints(Hooks):
    """
    Hook class to log the absolute value of algebraic constraints of
    a semi-explicit DAE after each step. Requires the implementation of
    a algebraic_constraints() method.
    """

    def post_step(self, step, level_number):
        r"""
        Default routine called after each step.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        # Compute value of algebraic equation(s)
        g = P.algebraic_constraints(L.uend, step.time + step.dt)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="abs_g_post_step",
            value=abs(g),
        )


class LogIntegrationErrorPreIter(Hooks):
    """
    Hook class to log the integration error using the Qd matrix of
    a semi-explicit DAE after prediction. Requires the implementation of
    a algebraic_constraints() method.
    """

    def pre_iteration(self, step, level_number):
        r"""
        Default routine called before each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().pre_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        M = L.sweep.coll.num_nodes

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        # L.sweep.compute_end_point()

        integral = P.dtype_u(P.init, val=0.0)
        for j in range(1, M + 1):
            integral[:] += L.dt * L.sweep.QI[M, j] * L.f[j][:]
        err_int = L.u[0][:] + integral[:] - L.u[-1][:]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="err_int_pre_iteration",
            value=abs(err_int),
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="err_int_diff_pre_iteration",
            value=abs(err_int.diff[:]),
        )


class LogIntegrationErrorPostIter(Hooks):
    """
    Hook class to log the integration error using the Qd matrix of
    a semi-explicit DAE after each iteration. Requires the implementation of
    a algebraic_constraints() method.
    """

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        M = L.sweep.coll.num_nodes

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        # L.sweep.compute_end_point()

        integral = P.dtype_u(P.init, val=0.0)
        for j in range(1, M + 1):
            integral[:] += L.dt * L.sweep.QI[M, j] * L.f[j][:]
        err_int = L.u[0][:] + integral[:] - L.u[-1][:]

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="err_int_post_iteration",
            value=abs(err_int),
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="err_int_diff_post_iteration",
            value=abs(err_int.diff[:]),
        )


class LogGlobalError(Hooks):
    def log_global_error(self, step, level_number, attr, variable, suffix=''):
        """
        Function to add the global error to the stats

        Args:
            step (pySDC.Step.step): The current step
            level_number (int): The index of the level
            suffix (str): Suffix for naming the variable in stats

        Returns:
            None
        """
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        u_ex = L.prob.u_exact(t=L.time + L.dt)

        u_ex_part = getattr(u_ex, attr)
        u_num_part = getattr(L.uend, attr)

        e_global = abs(u_ex_part - u_num_part)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f"e_global_{variable}{suffix}",
            value=e_global,
        )


class LogGlobalErrorDiffVar(LogGlobalError):
    def pre_iteration(self, step, level_number):
        super().pre_iteration(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_pre_iteration")

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_post_iteration")

    def pre_sweep(self, step, level_number):
        super().pre_sweep(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_pre_sweep")

    def post_sweep(self, step, level_number):
        super().post_sweep(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_post_sweep")

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self.log_global_error(step, level_number, attr="diff", variable="differential", suffix="_post_step")


class LogGlobalErrorAlgVar(LogGlobalError):
    def pre_iteration(self, step, level_number):
        super().pre_iteration(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_pre_iteration")

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_post_iteration")

    def pre_sweep(self, step, level_number):
        super().pre_sweep(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_pre_sweep")

    def post_sweep(self, step, level_number):
        super().post_sweep(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_post_sweep")

    def post_step(self, step, level_number):
        super().post_step(step, level_number)
        self.log_global_error(step, level_number, attr="alg", variable="algebraic", suffix="_post_step")
