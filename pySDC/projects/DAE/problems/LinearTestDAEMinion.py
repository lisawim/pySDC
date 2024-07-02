import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import gmres

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.core.errors import ProblemError


class LinearTestDAEMinionConstrained(ptype_dae):
    r"""
    For this class no quadrature is used for the algebraic constraints, i.e., system for algebraic constraints is solved directly.
    """

    def __init__(self, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=False):
        super().__init__(nvars=(3, 1), newton_tol=newton_tol)
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        self.A = np.zeros((4, 4))
        self.A[0, :] = [1, 0, -1, 1]
        self.A[1, :] = [0, -1e4, 0, 0]
        self.A[2, :] = [1, 0, 0, 0]
        self.A[3, :] = [1, 1, 0, 1]

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

        u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]
        u4 = u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = u1 - u3 + u4
        f.diff[1] = -1e4 * u2 + (1 + 1e4) * np.exp(t)
        f.diff[2] = u1
        f.alg[0] = self.algebraicConstraints(u, t)[0]
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
        f.alg[0] = u.diff[0] + u.diff[1] + u.alg[0] - np.exp(t)
        return f.alg

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
            u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]
            u4 = u.alg[0]

            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    u1 - factor * f.diff[0] - rhs.diff[0],
                    u2 - factor * f.diff[1] - rhs.diff[1],
                    u3 - factor * f.diff[2] - rhs.diff[2],
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
                    [1 - factor, 0, factor, -factor],
                    [0, 1 + factor * 1e4, 0, 0],
                    [-factor, 0, 1, 0],
                    [1, 1, 0, 1],
                ]
            )
            
            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.diff[2] -= dx[2]
            u.alg[0] -= dx[3]

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

    def u_exact(self, t, u_init=None, t_init=None):
        r"""
        Routine to approximate the exact solution at time t by ``SciPy`` or give initial conditions when called at :math:`t=0`.

        Parameters
        ----------
        t : float
            Current time.
        u_init : pySDC.problem.vanderpol.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            Approximate exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[0] = np.cos(t)
        me.diff[1] = np.exp(t)
        me.diff[2] = np.sin(t)
        me.alg[0] = -np.cos(t)
        return me


class LinearTestDAEMinionEmbedded(LinearTestDAEMinionConstrained):
    r"""
    For this class the naively approach of embedded SDC is used.
    """

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
            u1, u2, u3 = u.diff[0], u.diff[1], u.diff[2]
            u4 = u.alg[0]

            f = self.eval_f(u, t)

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array(
                [
                    u1 - factor * f.diff[0] - rhs.diff[0],
                    u2 - factor * f.diff[1] - rhs.diff[1],
                    u3 - factor * f.diff[2] - rhs.diff[2],
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
                    [1 - factor, 0, factor, -factor],
                    [0, 1 + factor * 1e4, 0, 0],
                    [-factor, 0, 1, 0],
                    [-factor, -factor, 0, -factor],
                ]
            )
            
            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g)

            u.diff[0] -= dx[0]
            u.diff[1] -= dx[1]
            u.diff[2] -= dx[2]
            u.alg[0] -= dx[3]

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