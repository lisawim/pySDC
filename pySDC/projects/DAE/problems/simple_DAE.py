import warnings
import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Problem import WorkCounter
from pySDC.core.Errors import ProblemError


class Jacobian:
    def __init__(self, u, func, rdiff=1e-9):
        self.n = len(u)
        self.m = len(func(u))
        self.func = func
        self.rdiff = rdiff

        self.jacobian = np.zeros((self.n, self.m))

    def evalJacobian(self, u):
        e = np.zeros(self.n)
        e[0] = 1
        for k in range(self.n):
            self.jacobian[:, k] = 1 / self.rdiff * (self.func(u + self.rdiff * e) - self.func(u))
            e = np.roll(e, 1)


class pendulum_2d(ptype_dae):
    r"""
    Example implementing the well known 2D pendulum as a first order differential-algebraic equation (DAE) of index 3.
    The DAE system is given by the equations

    .. math::
        \frac{dp}{dt} = u,

    .. math::
        \frac{dq}{dt} = v,

    .. math::
        m\frac{du}{dt} = -p \lambda,

    .. math::
        m\frac{dv}{dt} = -q \lambda - g,

    .. math::
        0 = p^2 + q^2 - l^2

    for :math:`l=1` and :math:`m=1`. The pendulum is used in most introductory literature on DAEs, for example on page 8
    of [1]_.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    t_end: float
        The end time at which the reference solution is determined.

    References
    ----------
    .. [1] E. Hairer, C. Lubich, M. Roche. The numerical solution of differential-algebraic systems by Runge-Kutta methods.
        Lect. Notes Math. (1989).
    """

    def __init__(self, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=5, newton_tol=newton_tol)
        # load reference solution
        # data file must be generated and stored under misc/data and self.t_end = t[-1]
        # data = np.load(r'pySDC/projects/DAE/misc/data/pendulum.npy')
        # t = data[:, 0]
        # solution = data[:, 1:]
        # self.u_ref = interp1d(t, solution, kind='cubic', axis=0, fill_value='extrapolate')
        self.t_end = 0.0
        self.g = 9.8

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
            Current value of the right-hand side of f (which includes five components).
        """

        # The last element of u is a Lagrange multiplier. Not sure if this needs to be time dependent, but must model the
        # weight somehow
        f = self.dtype_f(self.init)
        f.diff[:4] = (
            du.diff[0] - u.diff[2],
            du.diff[1] - u.diff[3],
            du.diff[2] + u.alg[0] * u.diff[0],
            du.diff[3] + u.alg[0] * u.diff[1] + self.g,
        )
        f.alg[0] = u.diff[0] ** 2 + u.diff[1] ** 2 - 1
        self.work_counters['rhs']()
        return f

    def u_exact(self, t):
        """
        Approximation of the exact solution generated by spline interpolation of an extremely accurate numerical reference solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object. It contains fixed initial conditions at initial time.
        """
        me = self.dtype_u(self.init)
        if t == 0:
            # me.diff[:4] = (-1, 0, 0, 0)
            me.diff[0] = -1
            me.diff[1] = 0
            me.diff[2] = 0
            me.diff[3] = 0
            me.alg[0] = 0
        elif t < self.t_end:
            u_ref = self.u_ref(t)
            me.diff[:4] = u_ref[:4]
            me.alg[0] = u_ref[5]
        else:
            # self.logger.warning("Requested time exceeds domain of the reference solution. Returning zero.")
            # me.diff[:4] = (0, 0, 0, 0)
            me.diff[0] = 0
            me.diff[1] = 0
            me.diff[2] = 0
            me.diff[3] = 0
            me.alg[0] = 0
        return me


class pendulum_2dIntegralFormulation(pendulum_2d):
    def __init__(self, nvars=5, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

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

        q1, q2, v1, v2 = u.diff[0], u.diff[1], u.diff[2], u.diff[3]
        lamb = u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = v1
        f.diff[1] = v2
        f.diff[2] = -lamb * q1
        f.diff[3] = -lamb * q2 - self.g
        f.alg[0] = q1 ** 2 + q2 ** 2 - 1
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            q1, q2, v1, v2 = u.diff[0], u.diff[1], u.diff[2], u.diff[3]
            lamb = u.alg[0]
            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    q1 - factor * f.diff[0] - rhs.diff[0],
                    q2 - factor * f.diff[1] - rhs.diff[1],
                    v1 - factor * f.diff[2] - rhs.diff[2],
                    v2 - factor * f.diff[3] - rhs.diff[3],
                    f.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1, 0, -factor, 0, 0],
                    [0, 1, 0, -factor, 0],
                    [lamb * factor, 0, 1, 0, q1 * factor],
                    [0, lamb * factor, 0, 1, q2 * factor],
                    [2 * q1, 2 * q2, 0, 0, 0],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.diff[2] -= dx[2]
            u.diff[3] -= dx[3]
            u.alg[0] -= dx[4]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class pendulum_2dIntegralFormulation2(pendulum_2dIntegralFormulation):

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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            q1, q2, v1, v2 = u.diff[0], u.diff[1], u.diff[2], u.diff[3]
            lamb = u.alg[0]
            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    q1 - factor * f.diff[0] - rhs.diff[0],
                    q2 - factor * f.diff[1] - rhs.diff[1],
                    v1 - factor * f.diff[2] - rhs.diff[2],
                    v2 - factor * f.diff[3] - rhs.diff[3],
                    -factor * (f.alg[0]) - rhs.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1, 0, -factor, 0, 0],
                    [0, 1, 0, -factor, 0],
                    [lamb * factor, 0, 1, 0, q1 * factor],
                    [0, lamb * factor, 0, 1, q2 * factor],
                    [-2 * factor * q1, -2 * factor * q2, 0, 0, 0],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.diff[2] -= dx[2]
            u.diff[3] -= dx[3]
            u.alg[0] -= dx[4]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class LinearIndexTwoDAE(ptype_dae):
    def __init__(self, newton_tol=1e-10):
        """Initialization routine"""
        super().__init__(nvars=3, newton_tol=newton_tol)

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

        x1, x2 = u.diff[0], u.diff[1]
        dx1, dx2 = du.diff[0], du.diff[1]
        z = u.alg[0]

        # taken from Minion et al. (2011), SI-SDC-DAE paper
        f = self.dtype_f(self.init)
        f.diff[0] = dx1 - x1,
        f.diff[1] = dx2 - 2 * x1 + 1e5 * x2 - z - (1e5 + 1) * np.exp(t)
        f.alg[0] = x1 + x2
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
        me.diff[1] = -np.exp(t)
        me.alg[0] = -(4 + 2e5) * np.exp(t)
        return me


class LinearIndexTwoDAEIntegralFormulation(LinearIndexTwoDAE):
    def __init__(self, nvars=3, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

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

        x1, x2 = u.diff[0], u.diff[1]
        z = u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = x1
        f.diff[1] = 2 * x1 - 1e5 * x2 + z + (1e5 + 1) * np.exp(t)
        f.alg[0] = x1 + x2
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            x1, x2 = u.diff[0], u.diff[1]
            z = u.alg[0]
            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    x1 - factor * f.diff[0] - rhs.diff[0],
                    x2 - factor * f.diff[1] - rhs.diff[1],
                    f.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor, 0, 0],
                    [-2 * factor, 1 + 1e5 * factor, -factor],
                    [1, 1, 0],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.alg[0] -= dx[2]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            # else:
            #     self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class LinearIndexTwoDAEIntegralFormulation2(LinearIndexTwoDAEIntegralFormulation):
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            x1, x2 = u.diff[0], u.diff[1]
            z = u.alg[0]
            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    x1 - factor * f.diff[0] - rhs.diff[0],
                    x2 - factor * f.diff[1] - rhs.diff[1],
                    -factor * f.alg[0] - rhs.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor, 0, 0],
                    [-2 * factor, 1 + 1e5 * factor, -factor],
                    [-factor, -factor, 0],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.alg[0] -= dx[2]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            # else:
            #     self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class simple_dae_1(ptype_dae):
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

    def __init__(self, newton_tol=1e-10):
        """Initialization routine"""
        super().__init__(nvars=3, newton_tol=newton_tol)

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
        a = 10.0
        f = self.dtype_f(self.init)

        f.diff[:2] = (
            -du.diff[0] + (a - 1 / (2 - t)) * u.diff[0] + (2 - t) * a * u.alg[0] + (3 - t) / (2 - t) * np.exp(t),
            -du.diff[1] + (1 - a) / (t - 2) * u.diff[0] - u.diff[1] + (a - 1) * u.alg[0] + 2 * np.exp(t),
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
        me.diff[:2] = (np.exp(t), np.exp(t))
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
        me.diff[:2] = (np.exp(t), np.exp(t))
        me.alg[0] = (np.exp(t) * (t - 3)) / ((2 - t) ** 2)
        return me


class simple_dae_1IntegralFormulation(simple_dae_1):
    def __init__(self, nvars=3, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.a = 10.0

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

        u1, u2 = u.diff[0], u.diff[1]
        z = u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = (self.a - 1 / (2 - t)) * u1 + (2 - t) * self.a * z + (3 - t) / (2 - t) * np.exp(t)
        f.diff[1] = (1 - self.a) / (t - 2) * u1 - u2 + (self.a - 1) * z + 2 * np.exp(t)
        f.alg[0] = (t + 2) * u1 + (t**2 - 4) * u2 - (t**2 + t - 2) * np.exp(t)
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            u1, u2 = u.diff[0], u.diff[1]
            z = u.alg[0]

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    u1 - factor * ((self.a - 1 / (2 - t)) * u1 + (2 - t) * self.a * z + (3 - t) / (2 - t) * np.exp(t)) - rhs.diff[0],
                    u2 - factor * ((1 - self.a) / (t - 2) * u1 - u2 + (self.a - 1) * z + 2 * np.exp(t)) - rhs.diff[1],
                    (t + 2) * u1 + (t**2 - 4) * u2 - (t**2 + t - 2) * np.exp(t),
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor * (self.a - 1 / (2 - t)), 0, -factor * self.a * (2 - t)],
                    [-factor * (1 - self.a) / (t - 2), 1 + factor, -factor * (self.a - 1)],
                    [t + 2, t**2 - 4, 0],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.alg[0] -= dx[2]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class simple_dae_1IntegralFormulation2(simple_dae_1IntegralFormulation):

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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            u1, u2 = u.diff[0], u.diff[1]
            z = u.alg[0]

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    u1 - factor * ((self.a - 1 / (2 - t)) * u1 + (2 - t) * self.a * z + (3 - t) / (2 - t) * np.exp(t)) - rhs.diff[0],
                    u2 - factor * ((1 - self.a) / (t - 2) * u1 - u2 + (self.a - 1) * z + 2 * np.exp(t)) - rhs.diff[1],
                    -factor * ((t + 2) * u1 + (t**2 - 4) * u2 - (t**2 + t - 2) * np.exp(t)) - rhs.alg[0],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1 - factor * (self.a - 1 / (2 - t)), 0, -factor * self.a * (2 - t)],
                    [-factor * (1 - self.a) / (t - 2), 1 + factor, -factor * (self.a - 1)],
                    [-factor * (t + 2), -factor * (t**2 - 4), 0],
                ]
            )

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.alg[0] -= dx[2]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            # else:
            #     self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me



class problematic_f(ptype_dae):
    r"""
    Standard example of a very simple fully implicit index-2 differential algebraic equation (DAE) that is not
    numerically solvable for certain choices of the parameter :math:`\eta`. The DAE system is given by

    .. math::
        \frac{d y(t)}{dt} + \eta t \frac{d z(t)}{dt} + (1 + \eta) z (t) = g (t).

    .. math::
        y (t) + \eta t z (t) = f(t),

    See, for example, page 264 of [1]_.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    eta : float
        Specific parameter of the problem.

    References
    ----------
    .. [1] U. Ascher, L. R. Petzold. Computer method for ordinary differential equations and differential-algebraic
        equations. Society for Industrial and Applied Mathematics (1998).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_tol, eta=1):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister('eta', localVars=locals())

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
            Current value of the right-hand side of f (which includes two components).
        """
        f = self.dtype_f(self.init)
        f[:] = (
            u[0] + self.eta * t * u[1] - np.sin(t),
            du[0] + self.eta * t * du[1] + (1 + self.eta) * u[1] - np.cos(t),
        )
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
            The reference solution as mesh object containing two components.
        """
        me = self.dtype_u(self.init)
        me[:] = (np.sin(t), 0)
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
            The reference solution as mesh object containing two components.
        """

        me = self.dtype_u(self.init)
        me[:] = (np.cos(t), 0)
        return me
