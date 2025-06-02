import numpy as np
from pathlib import Path
from scipy.optimize import root

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

    def __init__(
            self,
            newton_tol=1e-12,
            newton_maxiter=100,
            stop_at_maxiter=False,
            stop_at_nan=False,
            solver_type="hybr",
            kappa=1.0,
            lamb=0.375,
        ):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)
        self._makeAttributeAndRegister(
            "newton_tol",
            "newton_maxiter",
            "stop_at_maxiter",
            "stop_at_nan",
            "solver_type",
            "kappa",
            "lamb",
            localVars=locals(),
        )
        self.work_counters["rhs"] = WorkCounter()
        self.work_counters[self.solver_type] = WorkCounter()

        for path in [
            Path("/Users/lisa/Projects/Python/pySDC/pySDC/projects/DAE/data/"),
            Path("/beegfs/wimmer/pySDC/projects/DAE/data/")
        ]:
            if path.exists():
                path_to_data = path
                break
        else:
            raise FileNotFoundError("Could not locate data directory.")

        self.t_ref = np.load(path_to_data / "t_solve_michaelis_menten.npy")
        self.u_ref = np.load(path_to_data / "u_solve_michaelis_menten.npy")

    def g(self, factor, u, t, rhs):
        r"""
        Function of implicit system to be solved arising in ``fullyImplicitDAE``.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.1darray
            Function :math:`g`.
        """

        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff, u_approx_alg = rhs.diff[0], rhs.alg[0]

        fs0 = factor * s0
        fc0 = factor * c0
        f2s0c0 = factor ** 2 * s0 * c0

        g1 = (
            s0 + fs0 + u_approx_diff - u_approx_diff * u_approx_alg - u_approx_diff * fc0
            - u_approx_alg * factor * s0 - f2s0c0 - self.kappa * u_approx_alg
            - self.kappa * factor * c0 + self.lamb * u_approx_alg + self.lamb * factor * c0
        )
        g2 = (
            fs0 + u_approx_diff - u_approx_diff * u_approx_alg - u_approx_diff * fc0
            - factor * s0 * u_approx_alg - f2s0c0 - self.kappa * u_approx_alg
            - self.kappa * factor * c0
        )

        return np.array([g1, g2])

    def dg(self, factor, u, t, rhs):
        r"""
        Jacobian of function :math:`g`.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.2darray
            Jacobian matrix.
        """

        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff, u_approx_alg = rhs.diff[0], rhs.alg[0]

        dg1_ds = 1 + factor - factor * u_approx_alg - factor ** 2 * c0
        dg1_dc = -u_approx_diff * factor - factor ** 2 * s0 - self.kappa * factor + self.lamb * factor

        dg2_ds = factor - factor * u_approx_alg - factor ** 2 * c0
        dg2_dc = -u_approx_diff * factor - factor ** 2 * s0 - self.kappa * factor

        return np.array([[dg1_ds, dg1_dc], [dg2_ds, dg2_dc]])

    def dg_inv(self, factor, u, t, rhs):
        """
        Inverse of the Jacobian matrix.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.2darray
            Inverse of Jacobian.
        """

        jac = self.dg(factor, u, t, rhs)
        det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
        if abs(det) < 1e-14:
            raise np.linalg.LinAlgError("Jacobian determinant close to zero.")

        return np.array([[jac[1, 1], -jac[0, 1]], [-jac[1, 0], jac[0, 0]]]) / det

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the numerical solution at time t.
        du : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            The right-hand side of f (contains two components).
        """

        s0, c0 = u.diff[0], u.alg[0]
        ds0 = du.diff[0]

        f = self.dtype_f(self.init)
        f.diff[0] = ds0 + s0 - (s0 + self.kappa - self.lamb) * c0
        f.alg[0] = s0 - (s0 + self.kappa) * c0
        return f

    def solve_system(self, impl_sys, rhs, factor, u0, t):
        r"""
        Dispatcher that selects the appropriate solver backend based on ``self.solver_type``.
        Possible solvers are:

        - "hybr": solves the system using SciPy's root finder with hybrid methods
        - "newton": solves the system via a custom Newton-Raphson method

        Parameters
        ----------
        impl_sys : callable
            The function representing the fully implicit system (required for 'hybr').
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.

        Raises
        ------
        ProblemError
            If ``self.solver_type`` is not recognized.
        """

        if self.solver_type == "hybr":
            return self.solve_with_hybr(rhs, factor, u0, t, impl_sys)
        elif self.solver_type == "newton":
            return self.solve_with_newton(rhs, factor, u0, t)
        else:
            raise ProblemError(f"Unknown solver_type: {self.solver_type}")

    def solve_with_hybr(self, u_approx, factor, u0, t, impl_sys=None):
        r"""
        Root solver for the linear system using SciPy's ``optimize.root`` with
        'hybr' method.

        Parameters
        ----------
        u_approx : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Approximation of u in the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.
        impl_sys : callable, optional
            Implicit system needed for the routine; is defined in
            ``fullyImplicitDAE`` or ``semiImplicitDAE`` sweeper.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        solution = self.dtype_u(self.init)

        def implSysFlatten(unknowns, **kwargs):
            sys = impl_sys(unknowns.reshape(solution.shape).view(type(u0)), self, factor, u_approx, t, **kwargs)
            return sys.flatten()

        opt = root(
            implSysFlatten,
            u0.flatten(),
            method=self.solver_type,
            tol=self.newton_tol,
        )

        solution = self.dtype_u(self.init)
        solution[:] = opt.x.reshape(solution.shape)
        self.work_counters["hybr"].niter += opt.nfev
        return solution
    
    def solve_with_newton(self, rhs, factor, u0, t):
        r"""
        Newton's method to solve the linear system.

        Parameters
        ----------
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

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
            dg_inv = self.dg_inv(factor, u, t, rhs)

            # Newton update: u1 = u0 - g/dg
            dx = dg_inv @ g

            u.diff[0] -= dx[0]
            u.alg[0] -= dx[1]

            n += 1
            self.work_counters[self.solver_type]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]
        return me

    def u_exact(self, t, u_init=None, t_init=None, **kwargs):
        r"""
        Routine for the exact solution at time :math:`t`. For any time a
        reference solution is used where the index is searched matching with
        the required time.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        u_ex : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Exact solution.
        """

        u_ex = self.dtype_u(self.init)
        if t > 0.0:
            i = np.searchsorted(self.t_ref, t)

            if i < len(self.t_ref) and np.isclose(self.t_ref[i], t, atol=1e-14):
                ind = i
            elif i > 0 and np.isclose(self.t_ref[i-1], t, atol=1e-14):
                ind = i - 1
            else:
                print("No suitable entry found.")

            u_ref = self.u_ref[ind, :]

            u_ex.diff[0] = u_ref[0]
            u_ex.alg[0] = u_ref[1]

        else:
            u_ex.diff[0] = 1.0
            u_ex.alg[0] = 1.0

        return u_ex

    def du_exact(self, t):
        r"""
        Routine for the initial condition of derivative of exact solution
        at time :math:`t`. Required for Runge-Kutta methods.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        du_ex : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Derivative of exact solution.
        """

        assert t == 0.0, f"ERROR: Only initial condition at time 0.0 available!"

        u0 = self.u_exact(t)
        s0, c0 = u0.diff[0], u0.alg[0]

        du_ex = self.dtype_u(self.init)
        du_ex.diff[0] = -s0 + (s0 + self.kappa - self.lamb) * c0
        du_ex.alg[0] = s0 - (s0 + self.kappa) * c0
        return du_ex


class SemiImplicitMichaelisMentenDAE(MichaelisMentenDAE):
    """Semi-implicit variant with only integration of differential variable."""

    def g(self, factor, u, t, rhs):
        r"""
        Function of implicit system to be solved arising in ``semiImplicitDAE``.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.1darray
            Function :math:`g`.
        """

        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff = rhs.diff[0]

        fs0 = factor * s0

        g1 = s0 + fs0 + u_approx_diff - u_approx_diff * c0 - fs0 * c0 - self.kappa * c0 + self.lamb * c0
        g2 = fs0 + u_approx_diff - u_approx_diff * c0 - fs0 * c0 - self.kappa * c0

        return np.array([g1, g2])

    def dg(self, factor, u, t, rhs):
        r"""
        Jacobian of function :math:`g`.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.2darray
            Jacobian matrix.
        """

        s0, c0 = u.diff[0], u.alg[0]
        u_approx_diff = rhs.diff[0]

        dg1_ds = 1 + factor - factor * c0
        dg1_dc = -u_approx_diff - factor * s0 - self.kappa + self.lamb

        dg2_ds = factor - factor * c0
        dg2_dc = -u_approx_diff - factor * s0 - self.kappa

        return np.array([[dg1_ds, dg1_dc], [dg2_ds, dg2_dc]])


class MichaelisMentenConstrained(MichaelisMentenDAE):
    """Constrained formulation where only the differential equation is integrated numerically"""

    def g(self, factor, u, t, rhs):
        r"""
        Function of implicit system to be solved arising in ``genericImplicitConstrained``.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.1darray
            Function :math:`g`.
        """
        s0, c0 = u.diff[0], u.alg[0]

        g1 = s0 + factor * s0 - factor * (s0 + self.kappa - self.lamb) * c0 - rhs.diff[0]
        g2 = s0 - (s0 + self.kappa) * c0

        return np.array([g1, g2])

    def dg(self, factor, u, t, rhs):
        r"""
        Jacobian of function :math:`g`.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.2darray
            Jacobian matrix.
        """

        s0, c0 = u.diff[0], u.alg[0]

        dg1_ds = 1 - factor * (-1 + c0)
        dg1_dc = -factor * (s0 + self.kappa - self.lamb)

        dg2_ds = 1 - c0
        dg2_dc = -(s0 + self.kappa)

        return np.array([[dg1_ds, dg1_dc], [dg2_ds, dg2_dc]])

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            The right-hand side of f (contains two components).
        """

        s0, c0 = u.diff[0], u.alg[0]

        f = self.dtype_f(self.init)
        f.diff[0] = -s0 + (s0 + self.kappa - self.lamb) * c0
        f.alg[0] = s0 - (s0 + self.kappa) * c0
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Wrapper for the base class solver interface with omitted implicit system.

        Parameters
        ----------
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the nonlinear system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time point.

        Returns
        -------
        me : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the nonlinear system.
        """

        return super().solve_system(None, rhs, factor, u0, t)

    def solve_with_hybr(self, rhs, factor, u0, t, impl_sys=None):
        r"""
        Root solver for the linear system using SciPy's ``optimize.root`` with
        'hybr' method. The needed function and their Jacobian is defined locally.

        Parameters
        ----------
        u_approx : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Approximation of u in the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.
        impl_sys : callable, optional
            Implicit system needed for the routine; only required when
            using ``fullyImplicitDAE`` or ``semiImplicitDAE`` sweeper.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        rhs_vec = np.array([rhs.diff[0], rhs.alg[0]])
        u0_vec = np.array([u0.diff[0], u0.alg[0]])

        def fun(u):
            s0, c0 = u[0], u[1]
            f1 = s0 - factor * (-s0 + (s0 + self.kappa - self.lamb) * c0) - rhs_vec[0]
            f2 = s0 - (s0 + self.kappa) * c0
            return np.array([f1, f2])

        def jac(u):
            s0, c0 = u[0], u[1]
            return np.array([
                [1 - factor * (-1 + c0), -factor * (s0 + self.kappa - self.lamb)],
                [1 - c0, -(s0 + self.kappa)],
            ])

        opt = root(fun, u0_vec, method="hybr", jac=jac, tol=self.newton_tol)

        solution = self.dtype_u(self.init)
        solution.diff[0], solution.alg[0] = opt.x[0], opt.x[1]
        self.work_counters["hybr"].niter += opt.nfev
        return solution


class MichaelisMentenEmbedded(MichaelisMentenConstrained):
    """Problem class for an embedded method where only the algebraic constraint is enforced"""
    def g(self, factor, u, t, rhs):
        r"""
        Function of implicit system to be solved arising in ``genericImplicitEmbedded``.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.1darray
            Function :math:`g`.
        """

        s0, c0 = u.diff[0], u.alg[0]

        g1 = s0 + factor * s0 - factor * (s0 + self.kappa - self.lamb) * c0 - rhs.diff[0]
        g2 = -factor * s0 + factor * (s0 + self.kappa) * c0 - rhs.alg[0]
        return np.array([g1, g2])

    def dg(self, factor, u, t, rhs):
        r"""
        Jacobian of function :math:`g`.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Current numerical solution.
        t : float
            Current time.
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side for the implicit system.

        Returns
        -------
        np.2darray
            Jacobian matrix.
        """

        s0, c0 = u.diff[0], u.alg[0]

        dg1_ds = 1 - factor * (-1 + c0)
        dg1_dc = -factor * (s0 + self.kappa - self.lamb)

        dg2_ds = -factor * (1 - c0)
        dg2_dc = -factor * (-s0 - self.kappa)

        return np.array([[dg1_ds, dg1_dc], [dg2_ds, dg2_dc]])

    def solve_with_hybr(self, rhs, factor, u0, t, impl_sys=None):
        r"""
        Root solver for the linear system using SciPy's ``optimize.root`` with
        'hybr' method. The needed function and their Jacobian is defined locally.

        Parameters
        ----------
        u_approx : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Approximation of u in the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.
        impl_sys : callable, optional
            Implicit system needed for the routine; only required when
            using ``fullyImplicitDAE`` or ``semiImplicitDAE`` sweeper.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        rhs_vec = np.array([rhs.diff[0], rhs.alg[0]])
        u0_vec = np.array([u0.diff[0], u0.alg[0]])

        def fun(u):
            s0, c0 = u[0], u[1]
            f1 = s0 - factor * (-s0 + (s0 + self.kappa - self.lamb) * c0) - rhs_vec[0]
            f2 = -factor * (s0 - (s0 + self.kappa) * c0) - rhs_vec[1]
            return np.array([f1, f2])

        def jac(u):
            s0, c0 = u[0], u[1]
            return np.array([
                [1 - factor * (-1 + c0), -factor * (s0 + self.kappa - self.lamb)],
                [-factor * (1 - c0), -factor * (-s0 - self.kappa)],
            ])

        opt = root(fun, u0_vec, method="hybr", jac=jac, tol=self.newton_tol)

        solution = self.dtype_u(self.init)
        solution.diff[0], solution.alg[0] = opt.x[0], opt.x[1]
        self.work_counters["hybr"].niter += opt.nfev
        return solution
