import numpy as np
import dill
from pathlib import Path
from scipy.sparse.linalg import gmres
import scipy.linalg as la

from pySDC.core.problem import Problem, WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.projects.DAE.misc.meshDAE import MeshDAE
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh
from pySDC.core.errors import ProblemError


class chatGPTSPP(Problem):
    r"""
    Example implementing the singular perturbation problem of the form

    .. math::
        \frac{d}{dt} y = -y + t + 1,

    .. math::
        \varepsilon \frac{d}{dt} z = -z + y

    for :math:`0 < \varepsilon \ll 1`. The linear system at each node is solved
    by Newton's method.

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
            self, nvars=2, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=False, eps=0.001
    ):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals()
        )
        self.nvars = nvars
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        self.A = np.zeros((2, 2))
        self.A[0, :] = [-1, 0]
        self.A[1, :] = [1 / self.eps, -1 / self.eps]

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

        b = np.array([t + 1, 0])
        f = self.dtype_f(self.init)
        f[:] = self.A.dot(u) + b
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

        u = self.dtype_u(u0)
        Id = np.identity(2)

        b = np.array([t + 1, 0])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = u - factor * (self.A.dot(u) + b) - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * self.A

            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

            n += 1
            self.work_counters['newton']()
        # print(n, res)
        # print()
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
        me[:] = (t, t - self.eps + self.eps * np.exp(-t / self.eps))
        return me


class ProtheroRobinsonSPP(Problem):
    r"""
    Example implementing the cosine problem also known as the Prothero-Robinson example [1]_ of the form

    .. math::
        \frac{d}{dt} u = \frac{d}{dt} p - \frac{1}{\varepsilon} (u - p)

    for :math:`0 < \varepsilon \ll 1` and :math:`p(t) = cos(t)`. The exact solution is given by
    :math:`u(t) = p(t)`.

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

    References
    ----------
    .. [1] A. Prothero, A. Robinson. On the stability and accuracy of one-step methods for solving stiff systems
        of ordinary differential equations. Mathematics of Computation, pp. 145-162, 28 (1974).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
            self, nvars=2, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True, eps=0.001
    ):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals()
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.A = np.array([-1 / self.eps])

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
            The right-hand side of f (contains four components).
        """

        f = self.dtype_f(self.init)
        f[0] = 0
        f[1] = self.A.dot(u[1]) - self.A.dot(np.cos(t)) - np.sin(t)
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            f = self.eval_f(u, t)
            g = np.array([u[1] - factor * f[1] - rhs[1]])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array([1 - factor * self.A])

            dg_inv = dg[0] ** (-1)

            # newton update: u1 = u0 - g/dg
            u[1] -= dg_inv * g

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
        me[1] = u[1]

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
        me[:] = np.cos(t)
        return me


class DiscontinuousTestSPP(Problem):
    dtype_u = mesh
    dtype_f = mesh
    def __init__(
            self, nvars=2, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=False, eps=0.001
    ):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals()
        )
        self.nvars = nvars
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        self.t_switch_exact = np.arccosh(50)
        self.t_switch = None
        self.nswitches = 0

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

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = 2 * u[0] - 100
        f = self.dtype_f(self.init)
        f[0] = 0 if h >= 0 or t >= t_switch else u[1]
        f[1] = (u[0] ** 2 - u[1] ** 2 - 1) / self.eps
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

        u = self.dtype_u(u0)
        Id = np.identity(2)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            f = self.eval_f(u, t)
            g = u - factor * f - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            t_switch = np.inf if self.t_switch is None else self.t_switch
            h = 2 * u[0] - 100
            if h >= 0 or t >= t_switch:
                dg = np.array([[1, 0], [-2 * factor * (1 / self.eps) * u[0], 2 * factor * (1 / self.eps) * u[1]]])
            else:
                dg = np.array([[1, -factor], [-2 * factor * (1 / self.eps) * u[0], 2 * factor * (1 / self.eps) * u[1]]])

            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

            n += 1
            self.work_counters['newton']()

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

        assert t >= 1.0, 'ERROR: u_exact only available for t>=1'

        me = self.dtype_u(self.init)

        if t > 1.0:

            def eval_rhs(t, u):
                return self.eval_f(u, t)

            me[:] = self.generate_scipy_reference_solution(
                eval_rhs, t, u_init, t_init, method='Radau'
            )  # , max_step=1e-5)
        else:
            me[:] = (np.cosh(1.0), np.sinh(1.0))
        return me

    def generate_scipy_reference_solution(self, eval_rhs, t, u_init=None, t_init=None, **kwargs):
        """
        Compute a reference solution using `scipy.solve_ivp` with very small tolerances.
        Keep in mind that scipy needs the solution to be a one dimensional array. If you are solving something higher
        dimensional, you need to make sure the function `eval_rhs` takes a flattened one-dimensional version as an input
        and output, but reshapes to whatever the problem needs for evaluation.

        The keyword arguments will be passed to `scipy.solve_ivp`. You should consider passing `method='BDF'` for stiff
        problems and to accelerate that you can pass a function that evaluates the Jacobian with arguments `jac(t, u)`
        as `jac=jac`.

        Args:
            eval_rhs (function): Function evaluate the full right hand side. Must have signature `eval_rhs(float: t, numpy.1darray: u)`
            t (float): current time
            u_init (pySDC.implementations.problem_classes.Lorenz.dtype_u): initial conditions for getting the exact solution
            t_init (float): the starting time

        Returns:
            numpy.ndarray: Reference solution
        """
        import numpy as np
        from scipy.integrate import solve_ivp

        kwargs = {
            'atol': 100 * np.finfo(float).eps,
            'rtol': 100 * np.finfo(float).eps,
            **kwargs,
        }
        u_init = self.u_exact(t=1.0) if u_init is None else u_init * 1.0
        t_init = 1.00 if t_init is None else t_init

        u_shape = u_init.shape
        return solve_ivp(eval_rhs, (t_init, t), u_init.flatten(), **kwargs).y[:, -1].reshape(u_shape)


class LinearTestSPPMinion(Problem):
    r"""
    Example implementing the singular perturbation problem of the form

    .. math::
        \frac{d}{dt} u_1 = u_1 - u_3 + u_4,

    .. math::
        \frac{d}{dt} u_2 = -10^4 u_2 + (1 + 10^4) e^t,

    .. math::
        \frac{d}{dt} u_3 = u_1,

    .. math::
        \varepsilon \frac{d}{dt} u_4 = u_1 + u_2 + u_4 - e^t

    for :math:`0 < \varepsilon \ll 1`. The linear system at each node is solved by Newton's method. Note
    that the system can also be solved directly by a linear solver.

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
        4-by-4 coefficient matrix of the right-hand side of the linear system.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
            self, nvars=4, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=False, eps=0.001
    ):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals()
        )
        self.nvars = nvars

        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        self.A = np.zeros((4, 4))
        self.A[0, :] = [1, 0, -1, 1]
        self.A[1, :] = [0, -1e4, 0, 0]
        self.A[2, :] = [1, 0, 0, 0]
        self.A[3, :] = [1 / self.eps, 1 / self.eps, 0, 1 / self.eps]
        lambdas = np.linalg.eigvals(self.A)

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
            The right-hand side of f (contains four components).
        """

        f = self.dtype_f(self.init)
        b = np.array([0, (1 + 1e4) * np.exp(t), 0, -np.exp(t) / self.eps])
        f[:] = self.A.dot(u) + b
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

        u = self.dtype_u(u0)
        Id = np.identity(4)
        b = np.array([0, (1 + 1e4) * np.exp(t), 0, -np.exp(t) / self.eps])

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = u - factor * (self.A.dot(u) + b) - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.newton_tol:
                break

            # assemble dg
            dg = Id - factor * self.A

            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            print(f"Nan at time {t} for eps={self.eps}")
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            print(f"Nan at time {t} for eps={self.eps}")
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
        if t > 0.0:
            if self.eps >= 1e-3:

                def eval_rhs(t, u):
                    return self.eval_f(u, t)

                me[:] = self.generate_scipy_reference_solution(
                    eval_rhs, t, u_init, t_init, method='Radau', max_step=1e-8
                )
            else:
                me[:] = (np.cos(t), np.exp(t), np.sin(t), -np.cos(t))
        else:
            me[:] = (np.cos(t), np.exp(t), np.sin(t), -np.cos(t))
        return me


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
            self, nvars=2, newton_tol=1e-12, newton_maxiter=20, solver_type="direct", stop_at_maxiter=False, stop_at_nan=False, eps=0.001, **kwargs
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
        # if t > 0.0:
        #     def eval_rhs(t, u):
        #         return self.eval_f(u, t)

        #     me[:] = self.generate_scipy_reference_solution(
        #         eval_rhs, t, u_init, t_init, method='Radau'
        #     )

        # elif t == 0.0:
        #     me[:] = (np.exp(2 * self.lamb_diff * t), (self.lamb_diff / self.lamb_alg) * np.exp(2 * self.lamb_diff * t))

        result_y = 0
        for i in range(len(self.eigenvalues)):
            result_y += self.constants[i] * self.eigenvectors[0, i] * np.exp(self.eigenvalues[i] * t)

        result_z = 0
        for i in range(len(self.eigenvalues)):
            result_z += self.constants[i] * self.eigenvectors[1, i] * np.exp(self.eigenvalues[i] * t)

        me[:] = (np.real(result_y), np.real(result_z))
        return me


class LinearTestSPPIMEX(LinearTestSPP):
    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(
            self, nvars=2, newton_tol=1e-12, newton_maxiter=20, solver_type="direct", stop_at_maxiter=False, stop_at_nan=False, eps=0.001, **kwargs
    ):
        super().__init__(nvars, newton_tol, newton_maxiter, solver_type, stop_at_maxiter, stop_at_nan, eps, **kwargs)

        self.Adiff = np.zeros((2, 2))
        self.Adiff[0, :] = self.A[0, :]

        self.Aalg = np.zeros((2, 2))
        self.Aalg[1, :] = self.A[1, :]

    def __eval_fexpl(self, u, t):
        """
        Helper routine to evaluate the explicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        fexpl : dtype_f
            Explicit part of the right-hand side.
        """

        fexpl = self.dtype_u(self.init)
        fexpl = self.Adiff @ u

        return fexpl

    def __eval_fimpl(self, u, t):
        """
        Helper routine to evaluate the implicit part of the right-hand side.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed (not used here).

        Returns
        -------
        fimpl : dtype_f
            Implicit part of the right-hand side.
        """

        fimpl = self.dtype_u(self.init, val=0.0)
        fimpl = self.Aalg @ u

        return fimpl
    
    def eval_f(self, u, t):
        """
        Routine to evaluate both parts of the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side divided into two parts.
        """

        f = self.dtype_f(self.init)
        f.impl = self.__eval_fimpl(u, t)
        f.expl = self.__eval_fexpl(u, t)

        self.work_counters["rhs"]()
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

        assert self.solver_type == "direct", "Only direct solve is implemented!"

        me = self.dtype_u(self.init)

        factor_tilde = self.eps * factor
        me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.Aalg, rhs)  # direct solve
        # me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.Aalg, rhs)  # direct solve uses scaled factor ``factor_tilde``

        return me


class LinearTestSPP_YP(ProblemDAE):
    """
    Example implementing the singular perturbation problem of the form

    .. math::
        \frac{d}{dt} u_d = \lambda_d u_d + \lambda_a u_a,

    .. math::
        \varepsilon \frac{d}{dt} u_a = \lambda_d u_d - \lambda_a u_a

    for :math:`0 < \varepsilon \ll 1` and :math:`\lambda_d = \lambda_a = 1`. The linear system at each node is solved
    by a Newton-Krylov method defined in ProblemDAE.

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

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_tol=1e-12, solver_type="newton", eps=0.01):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister("eps", "newton_tol", "solver_type", localVars=locals())
        self.work_counters["rhs"] = WorkCounter()
        self.work_counters["newton"] = WorkCounter()

        self.lamb_diff = -2.0
        self.lamb_alg = 1.0

        self.Id = np.identity(2)

        self.A = np.zeros((2, 2))
        self.A[0, :] = [self.lamb_diff, self.lamb_alg]
        self.A[1, :] = [self.lamb_diff / self.eps, -self.lamb_alg / self.eps]

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
        y, z = u[0], u[1]
        du_diff, du_alg = du[0], du[1]

        f = self.dtype_f(self.init)
        f[0] = du_diff - self.lamb_diff * y - self.lamb_alg * z
        f[1] = du_alg - 1 / self.eps * (self.lamb_diff * y - self.lamb_alg * z)
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

        if self.solver_type == "direct":
            me = self.dtype_u(self.init)

            b = np.array([u_approx[0], u_approx[1]])
            u = np.linalg.solve(self.Id - factor * self.A, self.A.dot(b))
            me[0] = u[0]
            me[1] = u[1]

            return me

        elif self.solver_type == "newton":
            me = super().solve_system(impl_sys, u_approx, factor, u0, t)
            return me

        raise NotImplementedError()

    def u_exact(self, t, u_init=None, t_init=None):
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
        if t > 0.0:
            def eval_rhs(t, u):
                f0 = self.lamb_diff * u[0] + self.lamb_alg * u[1]
                f1 = 1 / self.eps * (self.lamb_diff * u[0] - self.lamb_alg * u[1])
                return np.array([f0, f1])

            me[:] = self.generate_scipy_reference_solution(
                eval_rhs, t, u_init, t_init, method='Radau'
            )

        elif t == 0.0:
            me[0] = np.exp(2 * self.lamb_diff * t)
            me[1] = self.lamb_diff / self.lamb_alg * np.exp(2 * self.lamb_diff * t)

        return me

    def du_exact(self, t):
        r"""
        Routine to approximate the derivative of the exact solution at time t.

        Parameters
        ----------
        t : float
            Current time.
        u_init : pySDC.problem.dtype_u
            Initial conditions for getting the exact solution.
        t_init : float
            The starting time.

        Returns
        -------
        me : dtype_u
            Approximate exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[0] = 2 * self.lamb_diff * np.exp(2 * self.lamb_diff * t)
        me.alg[0] = (2 * self.lamb_diff ** 2) / self.lamb_alg * np.exp(2 * self.lamb_diff * t)
        return me


class MichaelisMentenSPP(Problem):
    dtype_u = mesh
    dtype_f = mesh
    def __init__(
            self, nvars=2, newton_tol=1e-12, newton_maxiter=20, stop_at_maxiter=False, stop_at_nan=False, eps=0.001
    ):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals()
        )
        self.nvars = nvars
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        path_to_data = Path("/Users/lisa/Projects/Python/pySDC/pySDC/projects/DAE/data/")
        if not path_to_data.exists():
            path_to_data = Path("/beegfs/wimmer/Python/pySDC/lib/python3.9/site-packages/pySDC/implementations/problem_classes/")

        if eps in [10 ** (-m) for m in range(1, 12)]:            
            fname = path_to_data / f"refSol_SciPy_michaelisMenten_{eps=}.dat"

            with open(fname, "rb") as f:
                self.u_ref = dill.load(f)

        else:
            self.u_ref = 0

        self.kappa = 1
        self.lamb = 0.375

    def g(self, factor, u, t, rhs):
        s0, c0 = u[0], u[1]

        g1 = s0 - factor * (-s0 + s0 * c0 + self.kappa * c0 - self.lamb * c0) - rhs[0]
        g2 = c0 - factor * (s0 - (s0 + self.kappa) * c0) / self.eps - rhs[1]

        return np.array([g1, g2])

    def dg(self, factor, u):
        s0, c0 = u[0], u[1]

        dg = np.array(
            [
                [1 - factor * (-1 + c0), -factor * (s0 + self.kappa - self.lamb)],
                [-factor * (1 - c0) / self.eps, 1 - factor * (-s0 - self.kappa) / self.eps]
            ]
        )
        return dg
    
    def dg_inv(self, factor, u):
        s0, c0 = u[0], u[1]

        det_dg = 1 + factor - factor * c0 + factor * (s0 + self.kappa) / self.eps + factor ** 2 * self.lamb * (1 - c0) / self.eps

        dg_inv = np.array(
            [
                [1 - factor * (-s0 - self.kappa) / self.eps, factor * (s0 + self.kappa - self.lamb)],
                [factor * (1 - c0) / self.eps, 1 - factor * (-1 + c0)],
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

        s0, c0 = u[0], u[1]

        f = self.dtype_f(self.init)
        f[:] = (-s0 + (s0 + self.kappa - self.lamb) * c0, (s0 - (s0 + self.kappa) * c0) / self.eps)
        self.work_counters["rhs"]()
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
            u -= dg_inv @ g

            n += 1
            self.work_counters["newton"]()

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

        if 0.0 < t <= 2.0:
            me[0] = self.u_ref(t)[0]
            me[1] = self.u_ref(t)[1]
        elif t == 0.0:
            me[:] = (1.0, 1.0)
        return me


class MichaelisMentenSPP_YP(ProblemDAE):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, newton_tol=1e-12, eps=0.01):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister('eps', 'newton_tol', localVars=locals())
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['newton'] = WorkCounter()

        if eps in [10 ** (-m) for m in range(1, 12)]:
            path_to_data = '/home/lisa/Buw/Programme/Python/Libraries/pySDC/pySDC/projects/DAE/data/'
            fname = path_to_data + f'refSol_SciPy_michaelisMenten_{eps=}.dat'
            f = open(fname, 'rb')
            self.u_ref = dill.load(f)
            f.close()
        else:
            self.u_ref = 0

        self.kappa = 1
        self.lamb = 0.375

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
        s0, c0 = u[0], u[1]
        ds0, dc0 = du[0], du[1]

        f = self.dtype_f(self.init)
        f[0] = ds0 + s0 - (s0 + self.kappa - self.lamb) * c0
        f[1] = dc0 - (s0 - (s0 + self.kappa) * c0) / self.eps
        return f

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

        if 0.0 < t <= 2.0:
            me[0] = self.u_ref(t)[0]
            me[1] = self.u_ref(t)[1]
        elif t == 0.0:
            me[:] = (1.0, 1.0)
        return me


class StiffPendulum(Problem):
    r"""
    Example implementing the stiff mechanical system of a pendulum of the form

    .. math::
        \frac{d}{dt} q_1 = v_1,

    .. math::
        \frac{d}{dt} q_2 = v_2,

    .. math::
        \frac{d}{dt} v_1 = - \frac{1}{\varepsilon} \frac{r - 1}{r} q_1,

    .. math::
        \frac{d}{dt} v_2 = - \frac{1}{\varepsilon} \frac{r - 1}{r} q_2 - g,

    with :math:`r = \sqrt{q_1^2 + q_2^2}` the length of the spring and the scaled constant of gravity :math:`g`
    here set to :math:`9.8`. Note that the reduced system in the case :math:`\varepsilon = 0` is a semi-explicit DAE
    of index :math:`3`, see [1]_.

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

    References
    ----------
    .. [1] U. Ascher, L. R. Petzold. Computer method for ordinary differential equations and differential-algebraic
        equations. Society for Industrial and Applied Mathematics (1998).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
            self, nvars=4, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True, eps=0.001, g=9.8
    ):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', 'g', localVars=locals()
        )
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
            The right-hand side of f (contains four components).
        """

        q1, q2, v1, v2 = u[0], u[1], u[2], u[3]
        r = np.sqrt(q1**2 + q2**2)

        f = self.dtype_f(self.init)
        f[:] = (
            v1,
            v2,
            (-1 / self.eps) * ((r - 1) / r) * q1,
            (-1 / self.eps) * ((r - 1) / r) * q2 - self.g,
        )
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            q1, q2, v1, v2 = u[0], u[1], u[2], u[3]
            # # form the function g(u), such that the solution to the nonlinear problem is a root of g
            r = np.sqrt(q1**2 + q2**2)
            g = np.array(
                [
                    q1 - factor * v1 - rhs[0],
                    q2 - factor * v2 - rhs[1],
                    v1 - factor * ((-1 / self.eps) * ((r - 1) / r) * q1) - rhs[2],
                    v2 - factor * ((-1 / self.eps) * ((r - 1) / r) * q2 - self.g) - rhs[3],
                ]
            )

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array(
                [
                    [1, 0, -factor, 0],
                    [0, 1, 0, -factor],
                    [
                        factor * 1 / self.eps * (q1**2 * r + q2**2 * (r - 1)) / ((q1**2 + q2**2) ** (3 / 2)),
                        factor * 1 / self.eps * (q1 * q2) / ((q1**2 + q2**2) ** (3 / 2)),
                        1,
                        0,
                    ],
                    [
                        factor * 1 / self.eps * (q1 * q2) / ((q1**2 + q2**2) ** (3 / 2)),
                        factor * 1 / self.eps * (q1**2 * (r - 1) + q2**2 * r) / ((q1**2 + q2**2) ** (3 / 2)),
                        0,
                        1,
                    ],
                ]
            )

            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

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

        if t > 0.0:

            def eval_rhs(t, u):
                return self.eval_f(u, t)

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init, method='BDF', max_step=1e-6)
        else:
            me[:] = (1 - self.eps ** (1 / 4), 0.0, 0.0, 0.0)
        return me


class VanDerPol(Problem):
    r"""
    Example implementing the nonlinear Van der Pol oscillator as a first-order system

    .. math::
        \frac{d}{dt} y = -z,

    .. math::
        \varepsilon \frac{d}{dt} z = y - \left(\frac{z^3}{3} - z\right)

    for :math:`0 < \varepsilon \ll 1`.

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
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(
            self, nvars=2, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True, eps=0.001
    ):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals()
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        if eps in [10 ** (-m) for m in range(1, 12)]:
            path_to_data = '/home/lisa/Buw/Programme/Python/Libraries/pySDC/pySDC/projects/DAE/data/'
            fname = path_to_data + f'refSol_SciPy_VanDerPol_{eps=}.dat'
            f = open(fname, 'rb')
            self.u_ref = dill.load(f)
            f.close()
        else:
            self.u_ref = 0

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
            The right-hand side of f (contains four components).
        """

        y, z = u[0], u[1]
        f = self.dtype_f(self.init)
        f[:] = (
            z,
            ((1 - y**2) * z - y) / self.eps,
        )
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

        u = self.dtype_u(u0)

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            y, z = u[0], u[1]

            # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = np.array([y - factor * (z) - rhs[0], z - factor * ((1 - y**2) * z - y) / self.eps - rhs[1]])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.array([[1, -factor], [factor * (2 * y * z + 1) / self.eps, 1 + factor * (y**2 - 1) / self.eps]])

            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

            y, z = u[0], u[1]

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
        if 0.0 < t <= 3.0:
            me[0] = self.u_ref(t)[0]
            me[1] = self.u_ref(t)[1]
        elif t == 0.0:
            me[0] = 2.0
            me[1] = -2 / 3 + self.eps * 10 / 81 - self.eps**2 * 292 / 2187 - self.eps**3  * 1814 / 19683
        else:
            raise NotImplementedError("No exact solution available for t > 3.0!")
        return me
