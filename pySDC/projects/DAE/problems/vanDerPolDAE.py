import numpy as np

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.core.errors import ProblemError


class VanDerPolDAE(ProblemDAE):
    r"""
    This class implements the nonlinear Van der Pol equation of the form

    .. math::
        \frac{d}{dt} y = -z,

    .. math::
        0 = y - (\frac{z^3}{3} - z)

    with initial conditions :math:`y(0) = 2`, :math:`z(0) = 0`.
    """

    def __init__(self, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=(1, 1), newton_tol=newton_tol)
        self._makeAttributeAndRegister('newton_tol', localVars=locals())
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['newton'] = WorkCounter()

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
        y, z = u.diff[0], u.alg[0]
        dy = du.diff[0]

        f = self.dtype_f(self.init)
        f.diff[0] = dy - z
        f.alg[0] = z - y**2 * z - y
        return f

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
        # print(type(sp.special.lambertw(np.exp(2 * t))))
        if t > 0.0:
            me.diff[0] = 0.0  # 1j * sp.special.lambertw(np.exp(2 * t))
            me.alg[0] = 0.0  # sp.integrate.quad(1j * sp.special.lambertw(np.exp(2 * t)), 0, t)[0]
        elif t == 0.0:
            me.diff[0] = 2.0  # -2/3#-1j * sp.special.lambertw(np.exp(2 * t))
            me.alg[0] = -2 / 3  # 1.0#sp.integrate.quad(-1j * sp.special.lambertw(np.exp(2 * t)), -t, 0)[0]
        return me


class VanDerPolConstrained(VanDerPolDAE):
    r"""
    For this class no quadrature is used for the algebraic constraints, i.e., system for algebraic constraints is solved directly.
    """

    def __init__(self, nvars=(1, 1), newton_tol=1e-12, newton_maxiter=100):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the right-hand side of the problem.

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

        y, z = u.diff[0], u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = z
        f.alg[0] = z - y**2 * z - y
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

        rhs_diff = rhs.diff[0]

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            y, z = u.diff[0], u.alg[0]

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array([y - factor * z - rhs_diff, z - y**2 * z - y])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            dg = np.array([[1, -factor], [-2 * y * z - 1, 1 - y**2]])

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.alg[0] -= dx[1]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class VanDerPolEmbedded(VanDerPolConstrained):

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

        rhs_diff, rhs_alg = rhs.diff[0], rhs.alg[0]

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            y, z = u.diff[0], u.alg[0]

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array([y - factor * z - rhs_diff, -factor * (z - y**2 * z - y) - rhs_alg])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            dg = np.array([[1, -factor], [factor * (2 * y * z + 1), factor * (y**2 - 1)]])

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)  # .reshape(u.shape).view(type(u))

            u.diff[0] -= dx[0]
            u.alg[0] -= dx[1]

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me
