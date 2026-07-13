import numpy as np

from pySDC.core.errors import ParameterError, ProblemError
from pySDC.core.hooks import Hooks
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.core.problem import WorkCounter


class LogGlobalErrorSimpleDAE(Hooks):
    def log_global_error(self, step, level_number, suffix=''):
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

        u_ex_flat = u_ex.flatten()
        uend_flat = L.uend.flatten()

        components = {
            "u1": 0,
            "u2": 1,
            "z": 3,
        }

        for name, idx in components.items():
            e_global = abs(u_ex_flat[idx] - uend_flat[idx])

            self.add_to_stats(
                process=step.status.slot,
                time=L.time + L.dt,
                level=L.level_index,
                iter=step.status.iter,
                sweep=L.status.sweep,
                type=f"e_global_{name}{suffix}",
                value=e_global,
            )

    def pre_iteration(self, step, level_number):
        super().pre_iteration(step, level_number)
        self.log_global_error(step, level_number, suffix="_pre_iteration")

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)
        self.log_global_error(step, level_number, suffix="_post_iteration")

    def pre_sweep(self, step, level_number):
        super().pre_sweep(step, level_number)
        self.log_global_error(step, level_number, suffix="_pre_sweep")

    def post_sweep(self, step, level_number):
        super().post_sweep(step, level_number)
        self.log_global_error(step, level_number, suffix="_post_sweep")


class SimpleDAE(ProblemDAE):
    r"""
    Example implementing a smooth linear index-2 differential-algebraic equation (DAE) with known analytical solution.
    The DAE system is given by

    .. math::
        \frac{d u_1 (t)}{dt} = (\alpha - \frac{1}{2 - t}) u_1 (t) + (2-t) \alpha z (t) + \frac{3 - t}{2 - t},

    .. math::
        \frac{d u_2 (t)}{dt} = \frac{1 - \alpha}{t - 2} u_1 (t) - u_2 (t) + (\alpha - 1) z (t) + 2 e^{t},

    .. math::
        0 = (t + 2) u_1 (t) + (t^{2} - 4) u_2 (t) - (t^{2} + t - 2) e^{t}.

    The exact solution of this system is

    .. math::
        u_1 (t) = u_2 (t) = e^{t},

    .. math::
        z (t) = -\frac{e^{t}}{2 - t}.

    This example is commonly used to test that numerical implementations are functioning correctly. See, for example,
    page 267 of [1]_.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    References
    ----------
    .. [1] U. Ascher, L. R. Petzold. Computer method for ordinary differential equations and differential-algebraic
        equations. Society for Industrial and Applied Mathematics (1998).
    """

    def __init__(
        self,
        newton_tol=1e-14,
        newton_maxiter=10,
        stop_at_maxiter=False,
        stop_at_nan=False,
        verbose=False,
    ):
        """Initialization routine"""
        super().__init__(nvars=3, newton_tol=newton_tol)

        self._makeAttributeAndRegister(
            "newton_tol",
            "newton_maxiter",
            "stop_at_maxiter",
            "stop_at_nan",
            "verbose",
            localVars=locals(),
        )

        self.a = 10.0

        self.work_counters["rhs"] = WorkCounter()
        self.work_counters["newton"] = WorkCounter()

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            Current value of the right-hand side of f (which includes three components).
        """
        # Smooth index-2 DAE pg. 267 Ascher and Petzold (also the first example in KDC Minion paper)
        f = self.dtype_f(self.init)

        f.diff[:2] = (
            -du.diff[0] + (self.a - 1 / (2 - t)) * u.diff[0] + (2 - t) * self.a * u.alg[0] + (3 - t) / (2 - t) * np.exp(t),
            -du.diff[1] + (1 - self.a) / (t - 2) * u.diff[0] - u.diff[1] + (self.a - 1) * u.alg[0] + 2 * np.exp(t),
        )
        f.alg[0] = (t + 2) * u.diff[0] + (t**2 - 4) * u.diff[1] - (t**2 + t - 2) * np.exp(t)
        self.work_counters['rhs']()
        return f

    def u_exact(self, t):
        """
        Routine for the exact solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object containing three components.
        """
        me = self.dtype_u(self.init)
        me.diff[0] = np.exp(t)
        me.diff[1] = np.exp(t)
        me.alg[0] = -np.exp(t) / (2 - t)
        return me

    def du_exact(self, t):
        """
        Routine for the derivative of the exact solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object containing three components.
        """

        me = self.dtype_u(self.init)
        me.diff[0] = np.exp(t)
        me.diff[1] = np.exp(t)
        me.alg[0] = (np.exp(t) * (t - 3)) / ((2 - t) ** 2)
        return me


class SimpleDAEConstrained(SimpleDAE):

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the numerical solution at time t.
        du : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            The right-hand side of f (contains two components).
        """

        # Shortcuts
        u1, u2, z = u.diff[0], u.diff[1], u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = (self.a - 1 / (2 - t)) * u1 + (2 - t) * self.a * z + ((3 - t) / (2 - t)) * np.exp(t)
        f.diff[1] = ((self.a - 1) / (2 - t)) * u1 - u2 + (self.a - 1) * z + 2 * np.exp(t)

        f.alg[0] = self.algebraic_constraints(u, t)
        self.work_counters["rhs"]()
        return f

    def algebraic_constraints(self, u, t):
        """
        Here, the algebraic equations are the hidden constraints which are obtained by
        differentiating the "real" algebraic equations of the index-two problem.
        """
        u1, u2 = u.diff[0], u.diff[1]

        g = (t + 2) * u1 + (t**2 - 4) * u2 + (2 - t - t**2) * np.exp(t)
        return g
    
    def g(self, factor, u, t, rhs):
        r"""
        Function of implicit system to be solved arising in ``fullyImplicitDAE``.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.1darray
            Function :math:`g`.
        """

        u1, u2, z = u.diff[0], u.diff[1], u.alg[0]
        rhs_u1, rhs_u2 = rhs.diff[0], rhs.diff[1]

        f_u1 = (self.a - 1 / (2 - t)) * u1 + (2 - t) * self.a * z + ((3 - t) / (2 - t)) * np.exp(t)
        f_u2 = ((self.a - 1) / (2 - t)) * u1 - u2 + (self.a - 1) * z + 2 * np.exp(t)

        g1 = u1 - factor * f_u1 - rhs_u1
        g2 = u2 - factor * f_u2 - rhs_u2
        g3 = self.algebraic_constraints(u, t)
        return np.array([g1, g2, g3])

    def dg(self, factor, t):
        r"""
        Jacobian of function :math:`g`.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).

        Returns
        -------
        np.2darray
            Jacobian matrix.
        """

        return np.array(
            [
                [1 - factor * (self.a - 1 / (2 - t)), 0.0, -factor * (2 - t) * self.a],
                [-factor * ((self.a - 1) / (2 - t)), 1 + factor, -factor * (self.a - 1)],
                [t + 2, t**2 - 4, 0.0],
            ]
        )
    
    def solve_system(self, rhs, factor, u0, t):
        r"""
        Newton's method to solve the linear system.

        Parameters
        ----------
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        u = self.dtype_u(u0)

        # Start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = self.g(factor=factor, u=u, t=t, rhs=rhs)

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Inverse of dg
            dg = self.dg(factor=factor, t=t)

            # Newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.alg[0] -= dx[2]

            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        solution = self.dtype_u(self.init)
        solution[:] = u[:]
        return solution
    
    # def solve_system(self, rhs, factor, u0, t):
    #     """
    #     Wrapper for the base class solver interface with omitted implicit system.

    #     Parameters
    #     ----------
    #     rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
    #         Right-hand side of the nonlinear system to be solved.
    #     factor : float
    #         Step size-related factor (e.g., node-to-node step size).
    #     u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
    #         Initial guess for the solution.
    #     t : float
    #         Current time point.

    #     Returns
    #     -------
    #     me : pySDC.projects.DAE.misc.meshDAE.MeshDAE
    #         Numerical solution of the nonlinear system.
    #     """

    #     b = np.array([rhs.diff[0], rhs.diff[1], 0.0])

    #     dg = self.dg(factor, t)
    #     u = np.linalg.solve(dg, b)

    #     # res = dg @ u - b
    #     # print(np.abs(res))

    #     solution = self.dtype_u(self.init)
    #     solution.diff[0] = u[0]
    #     solution.alg[0] = u[1]

    #     return solution


class SimpleDAEConstrainedIndexOne(SimpleDAEConstrained):

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the numerical solution at time t.
        du : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            The right-hand side of f (contains two components).
        """

        # Shortcuts
        u1, u2, z = u.diff[0], u.diff[1], u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[:2] = (
            (self.a - 1 / (2 - t)) * u1 + (2 - t) * self.a * z + (3 - t) / (2 - t) * np.exp(t),
            (1 - self.a) / (t - 2) * u1 - u2 + (self.a - 1) * z + 2 * np.exp(t),
        )
        f.alg[0] = self.algebraic_constraints(u, t)
        self.work_counters["rhs"]()
        return f

    def algebraic_constraints(self, u, t):
        """
        Here, the algebraic equations are the hidden constraints which are obtained by
        differentiating the "real" algebraic equations of the index-two problem.
        """
        u1, u2 = u.diff[0], u.diff[1]
        g = (t + 2) * u1 + (t**2 - 4) * u2 - (t**2 + t - 2) * np.exp(t)
        return g

    def g(self, t, rhs):
        r"""
        Function of implicit system to be solved arising in ``fullyImplicitDAE``.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.1darray
            Function :math:`g`.
        """

        g = self.algebraic_constraints(u=rhs, t=t)
        return np.array([g])
    
    def dg(self, t):
        r"""
        Jacobian of function :math:`g`.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).

        Returns
        -------
        np.2darray
            Jacobian matrix.
        """

        return np.array(
            [
                [(t + 2), t**2 - 4, 0],
            ]
        )

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Newton's method to solve the linear system.

        Parameters
        ----------
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        u = self.dtype_u(u0)

        # Start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = self.g(t=t, rhs=rhs)

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Inverse of dg
            dg = self.dg(t=t)

            # Newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.alg[0] -= dx[2]

            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        solution = self.dtype_u(self.init)
        solution[:] = u[:]
        return solution


class SimpleDAEConstrainedHalfExplicit(SimpleDAEConstrained):

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the numerical solution at time t.
        du : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            The right-hand side of f (contains two components).
        """

        # Shortcuts
        u1, u2, z = u.diff[0], u.diff[1], u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[:2] = (
            (self.a - 1 / (2 - t)) * u1 + (2 - t) * self.a * z + (3 - t) / (2 - t) * np.exp(t),
            (1 - self.a) / (t - 2) * u1 - u2 + (self.a - 1) * z + 2 * np.exp(t),
        )
        f.alg[0] = self.algebraic_constraints(u, t)
        self.work_counters["rhs"]()
        return f

    def eval_f_diff(self, y, z, t):
        """
        Evaluate only the differential RHS f(y,z).
        """
        u = self.dtype_u(self.init)
        u.diff[:] = y
        u.alg[:] = z

        f = self.eval_f(u, t)
        return f.diff[:]
    
    def eval_g(self, y, t):
        g = (t + 2) * y[0] + (t**2 - 4) * y[1] - (t**2 + t - 2) * np.exp(t)
        return g

    def dg(self, t, factor):
        dg = (t + 2) * factor * (2 - t) * self.a + (t ** 2 - 4) * factor * (self.a - 1)
        return dg
    
    def solve_system(self, rhs, factor, u0, t, t_c):
        r"""
        Newton's method to solve the linear system.

        Parameters
        ----------
        h : callable
            Function with respect to algebraic variable z.
        z_guess : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess.
        t : float
            Current time.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        u = self.dtype_u(u0)

        # Start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Form the function h(u), such that the solution to the nonlinear problem is a root of g
            f = self.eval_f(u, t_c)
            g = self.eval_g(rhs.diff[:] + factor * f.diff[:], t)

            # If g is close to 0, then we are done
            res = abs(g)
            if res < self.newton_tol:
                break

            # Inverse of dg
            dg = self.dg(t=t, factor=factor)

            # Newton update: u1 = u0 - g/dg
            dx = dg ** (-1) * g

            u.alg[0] -= dx

            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)
        solution = self.dtype_u(self.init)
        solution[:].alg[:] = u[:].alg[:]
        return solution[:].alg[:]