import numpy as np

from pySDC.core.Problem import ptype, WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import ProblemError


class EmbeddedLinearTestDAE(ptype):
    r"""
    Example implementing the singular perturbation problem of the form

    .. math::
        \frac{d}{dt} u_d = \lambda_d u_d + \lambda_a u_a,

    .. math::
        \varepsilon \frac{d}{dt} u_a = \lambda_d u_d - \lambda_a u_a

    for :math:`0 < \varepsilon \ll 1` and :math:`\lambda_d = \lambda_a = 1`. The linear system at each node is solved
    by Newton's method. Note that the system can also be solved directly by a linear solver.

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
    def __init__(self, nvars=2, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True, eps=0.001):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals())
        self.nvars = nvars
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        self.A = np.zeros((2, 2))
        self.A[0, :] = [1, 1]
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

        f = self.dtype_f(self.init)
        f[:] = self.A.dot(u)
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
            # # form the function g(u), such that the solution to the nonlinear problem is a root of g
            g = u - factor * self.A.dot(u) - rhs

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

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init, method='Radau')#, max_step=1e-5)
        else:
            me[:] = (np.exp(2 * t), np.exp(2 * t))
        return me


class EmbeddedDiscontinuousTestDAE(ptype):
    dtype_u = mesh
    dtype_f = mesh
    def __init__(self, nvars=2, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True, eps=0.001):
        """Initialization routine"""
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals())
        self.nvars = nvars
        self.t_switch_exact = np.arccosh(50)
        self.t_switch = None
        self.nswitches = 0
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

        y, z = u[0], u[1]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = 2 * y - 100
        f = self.dtype_f(self.init)
        if h >= 0 or t >= t_switch:
            f[:] = (
                0,
                (y**2 - z**2 - 1) / self.eps,
            )
        else:
            f[:] = (
                z,
                (y**2 - z**2 - 1) / self.eps,
            )
        self.work_counters['rhs']()
        return f

    def eval_jac(self, u, t):
        y, z = u[0], u[1]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = 2 * y - 100
        if h >= 0 or t >= t_switch:
            jac = np.array([[0, 0], [(2 * y) / self.eps, (- 2 * z) / self.eps]])
        else:
            jac = np.array([[0, 1], [(2 * y) / self.eps, (- 2 * z) / self.eps]])
        return jac

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
            # # form the function g(u), such that the solution to the nonlinear problem is a root of g

            g = u - factor * self.eval_f(u, t) - rhs

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            # print(n, res)
            if res < self.newton_tol:
                break

            # assemble dg
            dg = np.eye(2) - factor * self.eval_jac(u, t)

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

        assert t >= 1.0, 'ERROR: u_exact only available for t>=1'

        me = self.dtype_u(self.init)
        if t > 1.0:
            def eval_rhs(t, u):
                return self.eval_f(u, t)

            me[:] = self.generate_scipy_reference_solution(eval_rhs, t, u_init, t_init, method='Radau')#, max_step=1e-5)
        else:
            if t <= self.t_switch_exact:
                me[:] = (np.cosh(t), np.sinh(t))
            else:
                me[:] = (np.cosh(self.t_switch_exact), np.sinh(self.t_switch_exact))
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
        t_init = 1.0 if t_init is None else t_init

        u_shape = u_init.shape
        return solve_ivp(eval_rhs, (t_init, t), u_init.flatten(), **kwargs).y[:, -1].reshape(u_shape)
