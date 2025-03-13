import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class ProtheroRobinsonDAE(ProblemDAE):
    r"""
    Example implementing the cosine problem also known as the Prothero-Robinson example [1]_.
    For :math:`\varepsilon = 0` we have the algebraic equation

    .. math::
        0 = -u + p

    for :math:`p(t) = cos(t)`. The exact solution is given by :math:`u(t) = p(t)`.

    Parameters
    ----------
    newton_tol : float
        Tolerance for inner solver to terminate.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts work, here, the number of right-hand side evaluations and work in inner solver
        are counted.
    """

    def __init__(self, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister("newton_tol", localVars=locals())
        self.work_counters["rhs"] = WorkCounter()
        self.work_counters["newton"] = WorkCounter()

    def p(self, t):
        return np.cos(t)

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

        f = self.dtype_f(self.init)
        f.diff[0] = 0
        f.alg[0] = -u.alg[0] + self.p(t)
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
        me.diff[0] = 0
        me.alg[0] = self.p(t)
        return me


class ProtheroRobinsonDAEConstrained(ProtheroRobinsonDAE):
    def __init__(self, nvars=2, newton_tol=1e-12, newton_maxiter=20):
        """Initialization routine"""
        super().__init__(newton_tol=newton_tol)

        self._makeAttributeAndRegister("newton_maxiter", localVars=locals())

        self.work_counters["rhs"] = WorkCounter()

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

        f = self.dtype_f(self.init)
        f.diff[0] = 0
        f.alg[0] = -u.alg[0] + self.p(t)
        self.work_counters["rhs"]()
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

        me = self.dtype_u(self.init)

        u = self.dtype_u(u0)

        # Start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array([-u.alg[0] + self.p(t)])

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Inverse of dg
            dg_inv = np.array([-1])

            # Newton update: u1 = u0 - g/dg
            dx = dg_inv @ g

            u.diff[0] = 0
            u.alg[0] -= dx

            n += 1
            self.work_counters["newton"]()

        me[:] = u[:]
        return me


class ProtheroRobinsonDAEEmbedded(ProtheroRobinsonDAEConstrained):
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

        me = self.dtype_u(self.init)

        u = self.dtype_u(u0)

        # Start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array([-factor * (-u.alg[0] + self.p(t)) - rhs.alg[0]])

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Inverse of dg
            dg_inv = np.array([1 / factor])

            # Newton update: u1 = u0 - g/dg
            dx = dg_inv @ g

            u.diff[0] -= 0
            u.alg[0] -= dx

            n += 1
            self.work_counters["newton"]()

        me[:] = u[:]
        return me
