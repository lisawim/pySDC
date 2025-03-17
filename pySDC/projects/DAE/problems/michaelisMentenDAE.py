import numpy as np
import dill
from pathlib import Path

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.core.errors import ProblemError


class MichaelisMentenDAE(ProblemDAE):
    r"""
    This class implements the nonlinear Michaelis-Menten kinetics problem of the form

    .. math::
        \frac{d}{dt} s_0 = -s_0 + (s_0 + \kappa - \lambda) c_0,

    .. math::
        0 = s_0 - (s_0 + \kappa) c_0

    with initial conditions :math:`s_0(0) = 1`, :math:`c_0(0) = 1`.
    """

    def __init__(self, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=False):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister("newton_tol", "newton_maxiter", "stop_at_maxiter", "stop_at_nan", localVars=locals())
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['newton'] = WorkCounter()

        path_to_data = Path("/Users/lisa/Projects/Python/pySDC/pySDC/projects/DAE/data/")
        if not path_to_data.exists():
            path_to_data = Path("/beegfs/wimmer/Python/pySDC/lib/python3.9/site-packages/pySDC/projects/DAE/problems/")

        fname = path_to_data / f"refSol_SciPy_michaelisMentenDAE.dat"
        with open(fname, "rb") as f:
            self.u_ref = dill.load(f)

        self.kappa = 1
        self.lamb = 0.375

    def g(self, factor, u, t, rhs):
        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff, u_approx_alg = rhs.diff[0], rhs.alg[0]

        g1 = s0 + factor * s0 + u_approx_diff - u_approx_diff * u_approx_alg - u_approx_diff * factor * c0 - u_approx_alg * factor * s0 - factor ** 2 * s0 * c0 - self.kappa * u_approx_alg - self.kappa * factor * c0 + self.lamb * u_approx_alg + self.lamb * factor * c0
        g2 = factor * s0 + u_approx_diff - u_approx_diff * u_approx_alg - u_approx_diff * factor * c0 - factor * s0 * u_approx_alg - factor ** 2 * s0 * c0 - self.kappa * u_approx_alg - self.kappa * factor * c0

        return np.array([g1, g2])

    def dg(self, factor, u, rhs):
        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff, u_approx_alg = rhs.diff[0], rhs.alg[0]

        dg = np.array(
            [
                [1 + factor - factor * u_approx_alg - factor ** 2 * c0, -u_approx_diff * factor - factor ** 2 * s0 - self.kappa * factor + self.lamb * factor],
                [factor - factor * u_approx_alg - factor ** 2 * c0, -u_approx_diff * factor - factor ** 2 * s0 - self.kappa * factor],
            ]
        )
        return dg
    
    def dg_inv(self, factor, u, rhs):
        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff, u_approx_alg = rhs.diff[0], rhs.alg[0]

        det_dg = (1 + factor - factor * u_approx_alg - factor ** 2 * c0) * (-u_approx_diff * factor - factor ** 2 * s0 - self.kappa * factor) - (factor - factor * u_approx_alg - factor ** 2 * c0) * (-u_approx_diff * factor - factor ** 2 * s0 - self.kappa * factor + self.lamb * factor)
        dg_inv = np.array(
            [
                [-u_approx_diff * factor - factor ** 2 * s0 - self.kappa * factor, -u_approx_diff * factor + factor ** 2 * s0 + self.kappa * factor - self.lamb * factor],
                [-factor + factor * u_approx_alg + factor ** 2 * c0, 1 + factor - factor * u_approx_alg - factor ** 2 * c0],
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
        s0, c0 = u.diff[0], u.alg[0]
        ds0 = du.diff[0]

        f = self.dtype_f(self.init)
        f.diff[0] = ds0 + s0 - (s0 + self.kappa - self.lamb) * c0
        f.alg[0] = s0 - (s0 + self.kappa) * c0
        return f
    
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
            dg_inv = self.dg_inv(factor, u, u_approx)

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
        if 0.0 < t <= 2.0:
            try:
                me.diff[0] = self.u_ref(t)[0]
                me.alg[0] = self.u_ref(t)[1]
            except ValueError:  # for cluster
                me.diff[0] = self.u_ref(t)[0][0]
                me.alg[0] = self.u_ref(t)[0][1]
        elif t == 0.0:
            me.diff[0] = 1.0
            me.alg[0] = 1.0
        return me


class SemiImplicitMichaelisMentenDAE(MichaelisMentenDAE):

    def __init__(self, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=False):
        """Initialization routine"""
        super().__init__(newton_tol=newton_tol, newton_maxiter=newton_maxiter, stop_at_maxiter=stop_at_maxiter, stop_at_nan=stop_at_nan)
        self._makeAttributeAndRegister("newton_tol", "newton_maxiter", "stop_at_maxiter", "stop_at_nan", localVars=locals())

    def g(self, factor, u, t, rhs):
        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff = rhs.diff[0]

        g1 = s0 + factor * s0 + u_approx_diff - u_approx_diff * c0 - factor * s0 * c0 - self.kappa * c0 + self.lamb * c0
        g2 = factor * s0 + u_approx_diff - u_approx_diff * c0 - factor * s0 * c0 - self.kappa * c0

        return np.array([g1, g2])

    def dg(self, factor, u, rhs):
        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff, u_approx_alg = rhs.diff[0], rhs.alg[0]

        dg = np.array(
            [
                [1 + factor - factor * c0, -u_approx_diff - factor * s0 - self.kappa + self.lamb],
                [factor - factor * c0, -u_approx_diff - factor * s0 - self.kappa],
            ]
        )
        return dg
    
    def dg_inv(self, factor, u, rhs):
        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff = rhs.diff[0]

        det_dg = (1 + factor - factor * c0) * (-u_approx_diff - factor * s0 - self.kappa) - (factor - factor * c0) * (-u_approx_diff - factor * s0 - self.kappa + self.lamb)
        dg_inv = np.array(
            [
                [-u_approx_diff - factor * s0 - self.kappa, u_approx_diff + factor * s0 + self.kappa - self.lamb],
                [-factor + factor * c0, 1 + factor - factor * c0],
            ]
        )
        return 1 / det_dg * dg_inv


class MichaelisMentenConstrained(MichaelisMentenDAE):
    r"""
    For this class no quadrature is used for the algebraic constraints, i.e., system for algebraic constraints is solved directly.
    """

    def __init__(self, nvars=2, newton_tol=1e-12, newton_maxiter=100):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def g(self, factor, u, t, rhs):
        s0, c0 = u.diff[0], u.alg[0]

        g1 = s0 - factor * (-s0 + (s0 + self.kappa - self.lamb) * c0) - rhs.diff[0]
        g2 = s0 - (s0 + self.kappa) * c0

        return np.array([g1, g2])

    def dg(self, factor, u):
        s0, c0 = u.diff[0], u.alg[0]

        dg = np.array(
            [
                [1 - factor * (-1 + c0), -factor * (s0 + self.kappa - self.lamb)],
                [1 - c0, -(s0 + self.kappa)],
            ]
        )

        return dg
    
    def dg_inv(self, factor, u):
        s0, c0 = u.diff[0], u.alg[0]

        det_dg = -s0 - self.kappa - factor * self.lamb + factor * c0 * self.lamb

        dg_inv = np.array(
            [
                [-(s0 + self.kappa), factor * (s0 + self.kappa - self.lamb)],
                [c0 - 1, 1 - factor * (-1 + c0)],
            ]
        )
        return 1 / det_dg * dg_inv

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

        s0, c0 = u.diff[0], u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = -s0 + (s0 + self.kappa - self.lamb) * c0
        f.alg[0] = s0 - (s0 + self.kappa) * c0
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
            # Form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = self.g(factor, u, t, rhs)

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Inverse of dg
            dg_inv = self.dg_inv(factor, u)

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

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class MichaelisMentenEmbedded(MichaelisMentenConstrained):
    def g(self, factor, u, t, rhs):
        s0, c0 = u.diff[0], u.alg[0]

        g1 = s0 - factor * (-s0 + (s0 + self.kappa - self.lamb) * c0) - rhs.diff[0]
        g2 = -factor * (s0 - (s0 + self.kappa) * c0) - rhs.alg[0]

        return np.array([g1, g2])

    def dg(self, factor, u):
        s0, c0 = u.diff[0], u.alg[0]

        dg = np.array(
            [
                [1 - factor * (-1 + c0), -factor * (s0 + self.kappa - self.lamb)],
                [-factor * (1 - c0), -factor * (-s0 - self.kappa)],
            ]
        )

        return dg
    
    def dg_inv(self, factor, u):
        s0, c0 = u.diff[0], u.alg[0]

        det_dg = factor * (s0 + self.kappa) + factor ** 2 * self.lamb * (1 - c0)

        dg_inv = np.array(
            [
                [-factor * (-s0 - self.kappa), factor * (s0 + self.kappa - self.lamb)],
                [factor * (1 - c0), 1 - factor * (-1 + c0)],
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = self.g(factor, u, t, rhs)

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                print(res)
                break

            # Inverse of dg
            dg_inv = self.dg_inv(factor, u)

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
        # print(n, res)
        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me
