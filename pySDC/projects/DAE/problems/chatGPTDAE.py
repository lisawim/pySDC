import numpy as np

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.core.errors import ProblemError


class chatGPTDAE(ptype_dae):
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

        u_diff, u_alg = u.diff, u.alg
        du_diff = du.diff

        f = self.dtype_f(self.init)
        f.diff[0] = du_diff[0] + u_diff[0] - t - 1
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
        f.alg[:] = u.diff[:] - u.alg[:]
        return f.alg

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
        me.diff[0] = t
        me.alg[0] = t
        return me


class chatGPTDAEConstrained(chatGPTDAE):
    r"""
    For this class no quadrature is used for the algebraic constraints, i.e., system for algebraic constraints is solved directly.
    """

    def __init__(self, nvars=(1, 1), newton_tol=1e-12, newton_maxiter=50, stop_at_maxiter=False, stop_at_nan=False):
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

        u_diff, u_alg = u.diff, u.alg

        f = self.dtype_f(self.init)
        f.diff[0] = -u_diff[0] + t + 1
        f.alg[:] = self.algebraicConstraints(u, t)
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
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            y, z = u.diff[0], u.alg[0]
            f_diff = -y + t + 1
            f_alg = self.algebraicConstraints(u, t)[0]

            g = np.array([y - factor * f_diff - rhs.diff[0], f_alg]).flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            # print(n, res)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array([[1 - factor * (-1), 0], [1, -1]])

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g).reshape(u.shape).view(type(u))
            u.diff[0] -= dx[0]
            u.alg[0] -= dx[1]

            n += 1
            self.work_counters['newton']()
        # print()
        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        # elif np.isnan(res):
        #     self.logger.warning('Newton got nan after %i iterations...' % n)

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class chatGPTDAEEmbedded(chatGPTDAEConstrained):
    r"""
    For this class no quadrature is used for the algebraic constraints, i.e., system for algebraic constraints is solved directly.
    """

    def __init__(self, nvars=(1, 1), newton_tol=1e-12, newton_maxiter=50, stop_at_maxiter=False, stop_at_nan=False):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

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
            y, z = u.diff[0], u.alg[0]
            f_diff = -y + t + 1
            f_alg = self.algebraicConstraints(u, t)[0]

            g = np.array([y - factor * f_diff - rhs.diff[0], -factor * f_alg - rhs.alg[0]]).flatten()

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            # print(n, res)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array([[1 - factor * (-1), 0], [-factor * 1, -factor * (-1)]])

            # newton update: u1 = u0 - g/dg
            dx = np.linalg.solve(dg, g).reshape(u.shape).view(type(u))
            u.diff[0] -= dx[0]
            u.alg[0] -= dx[1]

            n += 1
            self.work_counters['newton']()
        # print()
        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        # elif np.isnan(res):
        #     self.logger.warning('Newton got nan after %i iterations...' % n)

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me