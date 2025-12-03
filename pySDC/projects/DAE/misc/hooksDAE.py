from pySDC.core.hooks import Hooks


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


class LogGlobalErrorPreIterDifferentialVariable(Hooks):
    """
    Hook class to log the error to the output generated by the sweeper after
    prediction.
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

        # compute and save errors
        # Note that the component from which the error is measured is specified here
        upde = P.u_exact(step.time + step.dt)
        e_global_differential = abs(upde.diff - L.u[-1].diff)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_differential_pre_iteration",
            value=e_global_differential,
        )


class LogGlobalErrorPreIterationAlgebraicVariable(Hooks):
    """
    Logs the global error in the algebraic variable before each iteration.
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

        L = step.levels[level_number]
        P = L.prob

        # L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        e_global_algebraic = abs(upde[:].alg - L.u[-1][:].alg)
        # print(L.time + L.dt, e_global_algebraic)
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_algebraic_pre_iteration",
            value=e_global_algebraic,
        )


class LogGlobalErrorPostIterDiff(Hooks):
    """
    Logs the global error in the differential variable and its derivative after each iteration.
    """

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        e_global = abs(upde.diff - L.uend.diff)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_differential_post_iteration",
            value=e_global,
        )


class LogGlobalErrorPostIterAlg(Hooks):
    """
    Logs the global error in the algebraic variable and its derivative after each iteration.
    """

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        e_global_algebraic = abs(upde[:].alg - L.uend[:].alg)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_algebraic_post_iteration",
            value=e_global_algebraic,
        )


class LogGlobalErrorPostStepDifferentialVariable(Hooks):
    """
    Hook class to log the error to the output generated by the sweeper after
    each time step.
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

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        L.sweep.compute_end_point()

        # compute and save errors
        # Note that the component from which the error is measured is specified here
        upde = P.u_exact(step.time + step.dt)
        e_global_differential = abs(upde.diff - L.uend.diff)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_differential_post_step",
            value=e_global_differential,
        )


class LogGlobalErrorPostStepAlgebraicVariable(Hooks):
    """
    Logs the global error in the algebraic variable after each step.
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

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        e_global_algebraic = abs(upde.alg - L.uend.alg)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_algebraic_post_step",
            value=e_global_algebraic,
        )
