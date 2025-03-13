import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres

from pySDC.core.errors import ProblemError
from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class LinearTestDAE(ProblemDAE):
    """
    Semi-explicit linear DAE of index one. It reads

    .. math::
        \dfrac{d}{dt}u_d = \lambda_d u_d + \lambda_a u_a,

    .. math::
        0 = \lambda_d u_d - \lambda_a u_a,

    where :math:`u_d` is the differential variable and :math:`u_a` is denoted as the algebraic
    variable. :math:`\lambda_d` and :math:`\lambda_a` are non-zero fixed parameters.

    Parameters
    ----------
    lamb_diff : float
        Parameter :math:`\lambda_d`.
    lamb_alg : float
        Parameter :math:`\lambda_a`.
    newton_tol : float
        Tolerance for inner solver to terminate.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts work, here, the number of right-hand side evaluations and work in inner solver
        are counted.
    """

    def __init__(self, newton_tol=1e-12, newton_maxiter=20, solver_type="newton", stop_at_maxiter=False, stop_at_nan=False):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister("newton_tol", "newton_maxiter", "solver_type", "stop_at_maxiter", "stop_at_nan", localVars=locals())
        self.work_counters["rhs"] = WorkCounter()
        self.work_counters["newton"] = WorkCounter()

        self.lamb_diff = -2.0
        self.lamb_alg = 1.0

        self.A = np.zeros((2, 2))
        self.A[0, :] = [self.lamb_diff, self.lamb_alg]
        self.A[1, :] = [self.lamb_diff, -self.lamb_alg]

        self.Aalg = np.zeros((2, 2))
        self.Aalg[1, :] = self.A[1, :]

        self.Adiff = np.zeros((2, 2))
        self.Adiff[0, :] = self.A[0, :]

        self.Id0 = sp.diags_array([1, 0], offsets=0)

    def g(self, factor, u, t, rhs):
        y, z = u.diff[0], u.alg[0]
        rhs0, rhs1 = rhs.diff[0], rhs.alg[0]

        g1 = y - factor * (self.lamb_diff * y + self.lamb_alg * z) - (self.lamb_diff * rhs0 + self.lamb_alg * rhs1)
        g2 = -factor * (self.lamb_diff * y - self.lamb_alg * z) - (self.lamb_diff * rhs0 - self.lamb_alg * rhs1)

        return np.array([g1, g2])

    def dg(self, factor):
        return np.array([[1 - factor * self.lamb_diff, -factor * self.lamb_alg], [-factor * self.lamb_diff, factor * self.lamb_alg]])
    
    def dg_inv(self, factor):
        det_dg = factor * self.lamb_alg - 2 * factor ** 2 * self.lamb_diff * self.lamb_alg
        dg_inv = np.array(
            [
                [factor * self.lamb_alg, factor * self.lamb_alg],
                [factor * self.lamb_diff, 1 - factor * self.lamb_diff],
            ]
        )
        return 1 / det_dg * dg_inv

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
            The right-hand side of f (contains two components).
        """

        # Shortcuts
        u_diff, u_alg = u.diff[0], u.alg[0]
        du_diff = du.diff[0]

        f = self.dtype_f(self.init)
        f.diff[0] = du_diff - self.lamb_diff * u_diff - self.lamb_alg * u_alg
        f.alg[:] = self.algebraicConstraints(u, t)
        return f

    def algebraicConstraints(self, u, t):
        r"""
        Returns the algebraic constraints of the semi-explicit DAE system.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains two components).
        """

        f = self.dtype_f(self.init)
        f.alg[0] = self.lamb_diff * u.diff[0] - self.lamb_alg * u.alg[0]
        return f.alg

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Solver for nonlinear implicit system (defined in sweeper).

        Parameters
        ----------
        impl_sys : callable
            Implicit system to be solved.
        u_approx : dtype_u
            Approximation of solution :math:`u` which is needed to solve
            the implicit system.
        factor : float
            Abbrev. for the node-to-node stepsize.
        u0 : dtype_u
            Initial guess for solver.
        t : float
            Current time :math:`t`.

        Returns
        -------
        me : dtype_u
            Numerical solution.
        """

        me = self.dtype_u(self.init)

        if self.solver_type == "direct":
            b = np.array([u_approx.diff[0], u_approx.alg[0]])
            u = np.linalg.solve(self.Id0 - factor * self.A, self.A.dot(b))

            me.diff[0] = u[0]
            me.alg[0] = u[1]

            return me

        elif self.solver_type == "newton":
            u = self.dtype_u(u0)

            # Start newton iteration
            n = 0
            res = 99
            while n < self.newton_maxiter:
                # Form the function g(u), such that the solution to the nonlinear problem is a root of g
                g = self.g(factor, u, t, u_approx)

                # If g is close to 0, then we are done
                res = np.linalg.norm(g, np.inf)
                if res < self.newton_tol:
                    break

                # Inverse of dg
                dg_inv = self.dg_inv(factor)

                # Newton update: u1 = u0 - g/dg
                dx = dg_inv @ g

                u.diff[0] -= dx[0]
                u.alg[0] -= dx[1]

                n += 1
                self.work_counters["newton"]()

            if np.isnan(res) and self.stop_at_nan:
                raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
            elif np.isnan(res):
                self.logger.warning("Newton got nan after %i iterations..." % n)
            # if n == self.newton_maxiter:
                # msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
                # if self.stop_at_maxiter:
                #     raise ProblemError(msg)
                # else:
                #     self.logger.warning(msg)

            me[:] = u[:]
            # me = super().solve_system(impl_sys, u_approx, factor, u0, t)
            return me

        raise NotImplementedError()

    def u_exact(self, t, **kwargs):
        r"""
        Routine for the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[0] = np.exp(2 * self.lamb_diff * t)
        me.alg[0] = self.lamb_diff / self.lamb_alg * np.exp(2 * self.lamb_diff * t)
        return me

    def du_exact(self, t):
        me = self.dtype_u(self.init)
        me.diff[0] = 2 * self.lamb_diff * np.exp(2 * self.lamb_diff * t)
        me.alg[0] = (2 * self.lamb_diff ** 2) / self.lamb_alg * np.exp(2 * self.lamb_diff * t)
        return me


class SemiImplicitLinearTestDAE(LinearTestDAE):
    """
    Semi-explicit linear DAE of index one. It reads

    .. math::
        \dfrac{d}{dt}u_d = \lambda_d u_d + \lambda_a u_a,

    .. math::
        0 = \lambda_d u_d - \lambda_a u_a,

    where :math:`u_d` is the differential variable and :math:`u_a` is denoted as the algebraic
    variable. :math:`\lambda_d` and :math:`\lambda_a` are non-zero fixed parameters.

    Parameters
    ----------
    lamb_diff : float
        Parameter :math:`\lambda_d`.
    lamb_alg : float
        Parameter :math:`\lambda_a`.
    newton_tol : float
        Tolerance for inner solver to terminate.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts work, here, the number of right-hand side evaluations and work in inner solver
        are counted.
    """

    def __init__(self, newton_tol=1e-12, newton_maxiter=20, solver_type="newton", stop_at_maxiter=False, stop_at_nan=False):
        """Initialization routine"""
        super().__init__(newton_tol=newton_tol, newton_maxiter=newton_maxiter, solver_type=solver_type, stop_at_maxiter=stop_at_maxiter, stop_at_nan=stop_at_nan)
        self._makeAttributeAndRegister("newton_tol", "newton_maxiter", "solver_type", "stop_at_maxiter", "stop_at_nan", localVars=locals())

        self.Adiff2 = np.zeros((2, 2))
        self.Adiff2[:, 0] = self.A[:, 0].copy()

        self.Aalg2 = np.zeros((2, 2))
        self.Aalg2[:, 1] = self.A[:, 1].copy()

    def g(self, factor, u, t, rhs):
        y, z = u.diff[0], u.alg[0]
        rhs0 = rhs.diff[0]

        g1 = y - factor * self.lamb_diff * y - self.lamb_alg * z - self.lamb_diff * rhs0
        g2 = -factor * self.lamb_diff * y + self.lamb_alg * z - self.lamb_diff * rhs0

        return np.array([g1, g2])

    def dg(self, factor):
        return np.array([[1 - factor * self.lamb_diff, -self.lamb_alg], [-factor * self.lamb_diff, self.lamb_alg]])
    
    def dg_inv(self, factor):
        det_dg = self.lamb_alg - 2 * factor * self.lamb_diff * self.lamb_alg
        dg_inv = np.array(
            [
                [self.lamb_alg, self.lamb_alg],
                [factor * self.lamb_diff, 1 - factor * self.lamb_diff],
            ]
        )
        return 1 / det_dg * dg_inv

    def solve_system(self, impl_sys, u_approx, factor, u0, t):
        r"""
        Solver for nonlinear implicit system (defined in sweeper).

        Parameters
        ----------
        impl_sys : callable
            Implicit system to be solved.
        u_approx : dtype_u
            Approximation of solution :math:`u` which is needed to solve
            the implicit system.
        factor : float
            Abbrev. for the node-to-node stepsize.
        u0 : dtype_u
            Initial guess for solver.
        t : float
            Current time :math:`t`.

        Returns
        -------
        me : dtype_u
            Numerical solution.
        """

        me = self.dtype_u(self.init)

        if self.solver_type == "direct":
            b = np.array([u_approx.diff[0], u_approx.alg[0]])
            u = np.linalg.solve(self.Id0 - factor * self.Adiff2 - self.Aalg2, self.Adiff2.dot(b))

            me.diff[0] = u[0]
            me.alg[0] = u[1]

            return me
        
        elif self.solver_type == "newton":
            u = self.dtype_u(u0)

            # Start newton iteration
            n = 0
            res = 99
            while n < self.newton_maxiter:
                # Form the function g(u), such that the solution to the nonlinear problem is a root of g
                g = self.g(factor, u, t, u_approx)

                # If g is close to 0, then we are done
                res = np.linalg.norm(g, np.inf)
                if res < self.newton_tol:
                    break

                # Inverse of dg
                dg_inv = self.dg_inv(factor)

                # Newton update: u1 = u0 - g/dg
                dx = dg_inv @ g

                u.diff[0] -= dx[0]
                u.alg[0] -= dx[1]

                n += 1
                self.work_counters["newton"]()

            if np.isnan(res) and self.stop_at_nan:
                raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
            elif np.isnan(res):
                self.logger.warning("Newton got nan after %i iterations..." % n)
            # if n == self.newton_maxiter:
                # msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
                # if self.stop_at_maxiter:
                #     raise ProblemError(msg)
                # else:
                #     self.logger.warning(msg)

            me[:] = u[:]
            return me

        raise NotImplementedError()


class LinearTestDAEConstrained(LinearTestDAE):
    r"""
    For this class no quadrature is used for the algebraic constraints, i.e., system for algebraic constraints is solved directly.
    """

    def __init__(self, nvars=2, newton_tol=1e-12, newton_maxiter=50, solver_type="newton", stop_at_maxiter=False, stop_at_nan=False):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister(
            "newton_tol", "newton_maxiter", "solver_type", "stop_at_maxiter", "stop_at_nan", localVars=locals()
        )
        self.work_counters[self.solver_type] = WorkCounter()
        self.work_counters[self.solver_type + "_failure"] = WorkCounter()
        self.work_counters["rhs"] = WorkCounter()


        self.rhs = []
        self.jac = []

    def f(self, u, t):
        y, z = u.diff[0], u.alg[0]

        f1 = self.lamb_diff * y + self.lamb_alg * z
        f2 = self.lamb_diff * y - self.lamb_alg * z

        return np.array([f1, f2])

    def g(self, factor, u, t, rhs):
        y, z = u.diff[0], u.alg[0]

        g1 = y - factor * (self.lamb_diff * y + self.lamb_alg * z) - rhs.diff[0]
        g2 = self.lamb_diff * y - self.lamb_alg * z

        return np.array([g1, g2])

    def dg(self, factor):
        return np.array([[1 - factor * self.lamb_diff, -factor * self.lamb_alg], [self.lamb_diff, -self.lamb_alg]])
    
    def dg_inv(self, factor):
        det_dg = -self.lamb_alg + 2 * factor * self.lamb_diff * self.lamb_alg
        dg_inv = np.array(
            [
                [-self.lamb_alg, factor * self.lamb_alg],
                [-self.lamb_diff, 1 - factor * self.lamb_diff],
            ]
        )
        return 1 / det_dg * dg_inv

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains two components).
        """

        f_comps = self.f(u, t)

        f = self.dtype_f(self.init)
        f.diff[0] = f_comps[0]
        f.alg[0] = f_comps[1]
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).
        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        self.set_rhs(rhs)

        # Note that Jacobian is constant here
        jac = self.dg(factor)
        self.set_jac(jac)

        me = self.dtype_u(self.init)

        if self.solver_type == 'newton':
            u = self.dtype_u(u0)

            # Start newton iteration
            n = 0
            res = 99
            while n < self.newton_maxiter:
                # Form the function g(u), such that the solution to the nonlinear problem is a root of g
                g = self.g(factor, u, t, rhs)

                # If g is close to 0, then we are done
                res = np.linalg.norm(g, np.inf)
                if res < self.newton_tol:
                    break

                # Inverse of dg
                dg_inv = self.dg_inv(factor)

                # Newton update: u1 = u0 - g/dg
                dx = dg_inv @ g

                u.diff[0] -= dx[0]
                u.alg[0] -= dx[1]

                n += 1
                self.work_counters["newton"]()

            if np.isnan(res) and self.stop_at_nan:
                raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
            elif np.isnan(res):
                self.logger.warning("Newton got nan after %i iterations..." % n)
            if n == self.newton_maxiter:
                # msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
                # if self.stop_at_maxiter:
                #     raise ProblemError(msg)
                # else:
                #     self.logger.warning(msg)

                self.work_counters["newton_failure"]()

            me[:] = u[:]

        elif self.solver_type == "direct":
            b = np.array([rhs.diff[0], rhs.alg[0]])
            u = np.linalg.solve(self.Id0 - factor * self.Adiff + self.Aalg, b)
            me.diff[0] = u[0]
            me.alg[0] = u[1]

        elif self.solver_type == "gmres":
            b = np.array([rhs.diff[0], rhs.alg[0]])
            x0 = np.array([u0.diff[0], u0.alg[0]])
            u = gmres(
                A=self.Id0 - factor * self.Adiff - self.Aalg,
                b=b,
                x0=x0,
                rtol=1e-14,
                maxiter=100,
                atol=0,
                callback=self.work_counters[self.solver_type],
                callback_type='legacy',
            )[0]

            me.diff[0] = u[0]
            me.alg[0] = u[1]

        else:
            raise NotImplementedError(f"{self.solver_type} is not implemented. Use 'direct', 'newton' or 'gmres' instead.")

        return me

    def set_rhs(self, rhs):
        self.rhs.append([rhs.diff[0], rhs.alg[0]])

    def set_jac(self, jac):
        self.jac.append(jac)

    def clear_rhs(self):
        self.rhs = []

    def clear_jac(self):
        self.jac = []


class LinearTestDAEEmbedded(LinearTestDAEConstrained):
    r"""
    For this class the naively approach of embedded SDC is used.
    """

    def g(self, factor, u, t, rhs):
        y, z = u.diff[0], u.alg[0]

        g1 = y - factor * (self.lamb_diff * y + self.lamb_alg * z) - rhs.diff[0]
        g2 = -factor * (self.lamb_diff * y - self.lamb_alg * z) - rhs.alg[0]

        return np.array([g1, g2])

    def dg(self, factor):
        return np.array([[1 - factor * self.lamb_diff, -factor * self.lamb_alg], [-factor * self.lamb_diff, factor * self.lamb_alg]])
    
    def dg_inv(self, factor):
        det_dg = factor * self.lamb_alg - 2 * factor ** 2 * self.lamb_diff * self.lamb_alg
        dg_inv = np.array(
            [
                [factor * self.lamb_alg, factor * self.lamb_alg],
                [factor * self.lamb_diff, 1 - factor * self.lamb_diff],
            ]
        )
        return 1 / det_dg * dg_inv

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).
        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        self.set_rhs(rhs)

        # Note that Jacobian is constant here
        jac = self.dg(factor)
        self.set_jac(jac)

        me = self.dtype_u(self.init)

        if self.solver_type == 'newton':
            u = self.dtype_u(u0)

            # Start newton iteration
            n = 0
            res = 99
            while n < self.newton_maxiter:
                # Form the function g(u), such that the solution to the nonlinear problem is a root of g
                g = self.g(factor, u, t, rhs)

                # If g is close to 0, then we are done
                res = np.linalg.norm(g, np.inf)
                if res < self.newton_tol:
                    break

                # assemble dg
                dg_inv = self.dg_inv(factor)

                # Newton update: u1 = u0 - g/dg
                dx = dg_inv @ g

                u.diff[0] -= dx[0]
                u.alg[0] -= dx[1]

                n += 1
                self.work_counters['newton']()

            if np.isnan(res) and self.stop_at_nan:
                raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
            elif np.isnan(res):
                self.logger.warning('Newton got nan after %i iterations...' % n)
            if n == self.newton_maxiter:
                # msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
                # if self.stop_at_maxiter:
                #     raise ProblemError(msg)
                # else:
                #     self.logger.warning(msg)

                self.work_counters['newton_failure']()

            me[:] = u[:]

        elif self.solver_type == 'direct':
            b = np.array([rhs.diff[0], rhs.alg[0]])
            u = np.linalg.solve(self.Id0 - factor * self.A, b)
            me.diff[0] = u[0]
            me.alg[0] = u[1]

        elif self.solver_type == 'gmres':
            b = np.array([rhs.diff[0], rhs.alg[0]])
            x0 = np.array([u0.diff[0], u0.alg[0]])

            u = gmres(
                A=self.Id0 - factor * self.A,
                b=b,
                x0=x0,
                rtol=1e-15,
                maxiter=100,
                atol=0,
                callback=self.work_counters[self.solver_type],
                callback_type='legacy',
            )[0]

            me.diff[0] = u[0]
            me.alg[0] = u[1]

        else:
            raise NotImplementedError(f"{self.solver_type} is not implemented. Use 'direct', 'newton' or 'gmres' instead.")

        return me

    def u_exact(self, t, **kwargs):
        r"""
        Routine for the exact solution at time :math:`t`. Note that for the embedded SDC scheme the initial condition for
        the algebraic variable is multiplied by zero since in the limit we have :math:`\varepsilon = 0`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[0] = np.exp(2 * self.lamb_diff * t)
        me.alg[0] = self.lamb_diff / self.lamb_alg * np.exp(2 * self.lamb_diff * t)
        return me
