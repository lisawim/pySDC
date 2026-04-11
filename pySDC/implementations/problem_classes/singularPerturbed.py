import numpy as np
from scipy.sparse.linalg import gmres
import scipy.linalg as la

from pySDC.core.problem import Problem, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.errors import ProblemError


def compute_solution(A, u0):
    """
    Computes the solution of the differential equation y' = Ay.

    Parameters:
        A (numpy.ndarray): Coefficient matrix (2x2).
        initial_conditions (numpy.ndarray): Initial conditions [x0, y0].

    Returns:
        x_t (function): Function for x(t).
        y_t (function): Function for y(t).
    """
    # Compute eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = la.eig(A)

    # Solve for constants C and D using initial conditions
    constants = la.solve(eigenvectors, u0)

    return constants, eigenvalues, eigenvectors


class LinearTestSPP(Problem):
    r"""
    Example implementing the singular perturbation problem of the form

    .. math::
        \frac{d}{dt} u_d = \lambda_d u_d + \lambda_a u_a,

    .. math::
        \varepsilon \frac{d}{dt} u_a = \lambda_d u_d - \lambda_a u_a

    for :math:`0 < \varepsilon \ll 1` and :math:`\lambda_d = \lambda_a = 1`. The linear system at each node is solved
    by Newton's method. Note that the system can also be solved directly by a linear solver or, more
    directly: by a direct solver, i.e., the system is solved to get the exact solution.

    Parameters
    ----------
    nvars : int
        Number of unknowns in the problem.
    newton_tol : float
        Tolerance for Newton's method to terminate.
    newton_maxiter : int
        Maximum number of iterations for Newton's method.
    stop_at_maxiter : bool, optional
        Indicates that the Newton solver should stop if maximum number of iterations are executed.
    stop_at_nan : bool, optional
        Indicates that the Newton solver should stop if ``nan`` values arise.
    eps : float, optional
        Perturbation parameter :math:`\varepsilon`.

    Attributes
    ----------
    A : np.2darray
        2-by-2 coefficient matrix of the right-hand side of the linear system.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
            self,
            nvars=2,
            newton_tol=1e-12,
            newton_maxiter=20,
            solver_type="direct",
            stop_at_maxiter=False,
            stop_at_nan=False,
            eps=0.001,
            **kwargs,
    ):
        """Initialization routine"""
        Problem.__init__(self, init=(nvars, None, np.dtype('float64')), **kwargs)
        self._makeAttributeAndRegister(
            "newton_tol", "newton_maxiter", "solver_type", "stop_at_maxiter", "stop_at_nan", "eps", localVars=locals()
        )
        self.nvars = nvars
        self.work_counters[self.solver_type] = WorkCounter()
        self.work_counters[self.solver_type + "_failure"] = WorkCounter()
        self.work_counters["rhs"] = WorkCounter()

        self.lamb_diff = -2.0
        self.lamb_alg = 1.0

        self.Id = np.identity(2)

        self.A = np.zeros((2, 2))
        self.A[0, :] = [self.lamb_diff, self.lamb_alg]
        self.A[1, :] = [self.lamb_diff / self.eps, -self.lamb_alg / self.eps]

        self.u0 = np.array([1, -2])

        self.constants, self.eigenvalues, self.eigenvectors = compute_solution(self.A, self.u0)

        self.rhs = []
        self.jac = []

    def f(self, u, t):
        f1 = self.lamb_diff * u[0] + self.lamb_alg * u[1]
        f2 = 1 / self.eps * (self.lamb_diff * u[0] - self.lamb_alg * u[1])

        return np.array([f1, f2])
    
    def eval_nonhomogeneous_part(self, t):
        non_f = self.dtype_f(self.init, val=0.0)
        return non_f


    def g(self, factor, u, t, rhs):
        g1 = u[0] - factor * (self.lamb_diff * u[0] + self.lamb_alg * u[1]) - rhs[0]
        g2 = u[1] - factor * (self.lamb_diff * u[0] - self.lamb_alg * u[1]) / self.eps - rhs[1]

        return np.array([g1, g2])

    def dg(self, factor):
        dg = np.array(
            [
                [1 - factor * self.lamb_diff, -factor * self.lamb_alg],
                [(-factor * self.lamb_diff) / self.eps, 1 + (factor * self.lamb_alg) / self.eps],
            ]
        )
        return dg
    
    def dg_inv(self, factor):
        det_dg = 1 + (factor * self.lamb_alg) / self.eps - factor * self.lamb_diff - (2 * factor ** 2 * self.lamb_diff * self.lamb_alg) / self.eps

        dg_inv = np.array(
            [
                [1 + (factor * self.lamb_alg) / self.eps, factor * self.lamb_alg],
                [(factor * self.lamb_diff) / self.eps, 1 - factor * self.lamb_diff],
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

        f_rhs = self.f(u, t)

        f = self.dtype_f(self.init)
        f[:] = f_rhs[:]
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver.

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

        if self.solver_type == "newton":
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
                u -= dg_inv @ g

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
            me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.A, rhs)  # direct solve

        elif self.solver_type == "gmres":
            me[:] = gmres(
                A=self.Id - factor * self.A,
                b=rhs,
                x0=u0,
                rtol=1e-13,
                maxiter=1000,
                atol=0,
                callback=self.work_counters[self.solver_type],
                callback_type="legacy",
            )[0]

        else:
            raise NotImplementedError

        return me
    
    def set_rhs(self, rhs):
        self.rhs.append([rhs[0], rhs[1]])

    def set_jac(self, jac):
        self.jac.append(jac)

    def clear_rhs(self):
        self.rhs = []

    def clear_jac(self):
        self.jac = []

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

        result_y = 0
        for i in range(len(self.eigenvalues)):
            result_y += self.constants[i] * self.eigenvectors[0, i] * np.exp(self.eigenvalues[i] * t)

        result_z = 0
        for i in range(len(self.eigenvalues)):
            result_z += self.constants[i] * self.eigenvectors[1, i] * np.exp(self.eigenvalues[i] * t)

        me[:] = (np.real(result_y), np.real(result_z))
        return me
