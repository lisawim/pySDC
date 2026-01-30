import numpy as np
from numpy.fft import rfft, irfft

from pySDC.core.errors import ProblemError
from pySDC.core.hooks import Hooks
from pySDC.core.problem import WorkCounter
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.projects.DAE.problems.spectralTester import SpectralTester
from pySDC.helpers import problem_helper


# Problem specific hooks
class LogSolutionPreIter(Hooks):
    """
    Store the solution before each iteration as "u".
    """

    def post_iteration(self, step, level_number):
        """
        Record solution at the end of the iteration

        Args:
            step (pySDC.Step.step): the current step
            level_number (int): the current level number

        Returns:
            None
        """
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )


class LogGlobalErrorPreIter(Hooks):
    """Logs global error of concentrations in reaction-diffusion after prediction."""

    def pre_iteration(self, step, level_number):
        r"""
        Default routine called before each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().pre_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        # Compute end point to use L.uend instead of L.u[-1]
        L.sweep.compute_end_point()

        u_ref = L.prob.u_exact(t=L.time + L.dt)

        self.add_to_stats(
            process=step.status.slot,
            process_sweeper=L.sweep.rank,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type=f'e_global_pre_iteration',
            value=abs(u_ref - L.uend),
        )


class LogGlobalErrorPreIterConcentrations(Hooks):
    """Logs global error of concentrations in reaction-diffusion after prediction."""

    def pre_iteration(self, step, level_number):
        r"""
        Default routine called before each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().pre_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        n = P.nvars

        # Compute end point to use L.uend instead of L.u[-1]
        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        du_ex = P.du_exact(step.time + step.dt)

        if type(P).__name__ == "ReactionDiffusionPDAE_FFT_Radau":
            uend_ex = P.dtype_u(P.physical_init)
            uend_ex[: P.N] = np.fft.self.itransform(upde[: P.Nr], n=P.N)
            uend_ex[P.N : 2 * P.N] = np.fft.self.itransform(upde[P.Nr : 2 * P.Nr], n=P.N)
            uend_ex[2 * P.N : 3 * P.N] = np.fft.self.itransform(upde[2 * P.Nr : 3 * P.Nr], n=P.N)

            uend = P.dtype_u(P.physical_init)
            uend[: P.N] = np.fft.self.itransform(L.uend[: P.Nr], n=P.N)
            uend[P.N : 2 * P.N] = np.fft.self.itransform(L.uend[P.Nr : 2 * P.Nr], n=P.N)
            uend[2 * P.N : 3 * P.N] = np.fft.self.itransform(L.uend[2 * P.Nr : 3 * P.Nr], n=P.N)

            uend_ex = uend_ex.flatten()
            uend = uend.flatten()

        else:
            uend_ex = upde.flatten()
            uend = L.uend.flatten()

            duend_ex = du_ex.flatten()
            duend = L.f[-1].flatten()  # Note that L.f[-1] corresponds to L.u[-1] AND L.uend!

        e_global_concentration_u = abs(uend_ex[:n] - uend[:n])
        e_global_concentration_v = abs(uend_ex[n : 2 * n] - uend[n : 2 * n])
        e_global_concentration_w = abs(uend_ex[2 * n : 3 * n] - uend[2 * n : 3 * n])

        e_global_concentration_gradient_u = abs(duend_ex[:n] - duend[:n])
        e_global_concentration_gradient_v = abs(duend_ex[n : 2 * n] - duend[n : 2 * n])
        e_global_concentration_gradient_w = abs(duend_ex[2 * n : 3 * n] - duend[2 * n : 3 * n])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_u_pre_iteration",
            value=e_global_concentration_u,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_v_pre_iteration",
            value=e_global_concentration_v,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_w_pre_iteration",
            value=e_global_concentration_w,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_du_pre_iteration",
            value=e_global_concentration_gradient_u,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_dv_pre_iteration",
            value=e_global_concentration_gradient_v,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_dw_pre_iteration",
            value=e_global_concentration_gradient_w,
        )


class LogGlobalErrorPostIterConcentrations(Hooks):
    """Logs global error of concentrations in reaction-diffusion after each iteration."""

    def post_iteration(self, step, level_number):
        r"""
        Default routine called after each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        n = P.nvars

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)
        du_ex = P.du_exact(step.time + step.dt)

        if type(P).__name__ == "ReactionDiffusionPDAE_FFT_Radau":
            uend_ex = P.dtype_u(P.physical_init)
            uend_ex[: P.N] = P.itransform(upde[: P.Nr], n=P.N)
            uend_ex[P.N : 2 * P.N] = P.itransform(upde[P.Nr : 2 * P.Nr], n=P.N)
            uend_ex[2 * P.N : 3 * P.N] = P.itransform(upde[2 * P.Nr : 3 * P.Nr], n=P.N)

            uend = P.dtype_u(P.physical_init)
            uend[: P.N] = P.itransform(L.uend[: P.Nr], n=P.N)
            uend[P.N : 2 * P.N] = P.itransform(L.uend[P.Nr : 2 * P.Nr], n=P.N)
            uend[2 * P.N : 3 * P.N] = P.itransform(L.uend[2 * P.Nr : 3 * P.Nr], n=P.N)

            uend_ex = uend_ex.flatten()
            uend = uend.flatten()

        else:
            uend_ex = upde.flatten()
            uend = L.uend.flatten()

            duend_ex = du_ex.flatten()
            duend = L.f[-1].flatten()  # Note that L.f[-1] corresponds to L.u[-1] AND L.uend!

        e_global_concentration_u = abs(uend_ex[:n] - uend[:n])
        e_global_concentration_v = abs(uend_ex[n : 2 * n] - uend[n : 2 * n])
        e_global_concentration_w = abs(uend_ex[2 * n : 3 * n] - uend[2 * n : 3 * n])

        e_global_concentration_gradient_u = abs(duend_ex[:n] - duend[:n])
        e_global_concentration_gradient_v = abs(duend_ex[n : 2 * n] - duend[n : 2 * n])
        e_global_concentration_gradient_w = abs(duend_ex[2 * n : 3 * n] - duend[2 * n : 3 * n])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_u_post_iteration",
            value=e_global_concentration_u,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_v_post_iteration",
            value=e_global_concentration_v,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_w_post_iteration",
            value=e_global_concentration_w,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_du_post_iteration",
            value=e_global_concentration_gradient_u,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_dv_post_iteration",
            value=e_global_concentration_gradient_v,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_concentration_dw_post_iteration",
            value=e_global_concentration_gradient_w,
        )


class LogGlobalErrorNodesPostIterConcentrations(Hooks):
    """Logs global error at collocation nodes of concentrations in reaction-diffusion after each iteration."""

    def post_iteration(self, step, level_number):
        r"""
        Default routine called after each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        coll_nodes = L.time + L.dt * L.sweep.coll.nodes[:]
        M = L.sweep.coll.num_nodes

        e_global_nodes_w = np.array(
            [abs(L.u[m + 1].alg[: P.N] - P.u_exact(coll_nodes[m]).alg[: P.N]) for m in range(M)]
        )
        e_global_initial_w = np.array([abs(L.u[0].alg[: P.N] - P.u_exact(0.0).alg[: P.N])])
        e_global_next_w = np.array([abs(L.uend.alg[: P.N] - P.u_exact(L.time + L.dt).alg[: P.N])])
        e_global_nodes_w_all = np.concatenate((e_global_initial_w, e_global_nodes_w, e_global_next_w))

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_nodes_w_post_iteration",
            value=e_global_nodes_w_all,
        )


class LogGlobalErrorPostIterAlgebraicEquation(Hooks):
    """Logs global error of algebraic equation in reaction-diffusion after each iteration."""

    def post_iteration(self, step, level_number):
        r"""
        Default routine called after each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        du_ex = P.du_exact(step.time + step.dt)

        f = P.eval_f(L.uend, du_ex, step.time + step.dt)

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_algebraic_eq_post_iteration",
            value=abs(f.alg[: P.N]),
        )


class ReactionDiffusionPDAE(SpectralTester):
    def __init__(
        self,
        bc="periodic",
        L=1.0,
        newton_tol=1e-12,
        newton_maxiter=10,
        nvars=4,
        spectral=True,
        stop_at_maxiter=False,
        stop_at_nan=False,
        verbose=False,
    ):
        """Initialization routine"""

        self._makeAttributeAndRegister(
            "bc",
            "L",
            "newton_tol",
            "newton_maxiter",
            "nvars",
            "spectral",
            "stop_at_maxiter",
            "stop_at_nan",
            "verbose",
            localVars=locals(),
        )

        if self.nvars % 2 != 0:
            raise ProblemError("setup requires nvars = 2^n!")

        self.N = nvars
        self.Nr = self.nvars // 2 + 1

        super().__init__(nvars=self.nvars, Nr=self.Nr, newton_tol=self.newton_tol)

        self.work_counters["rhs"] = WorkCounter()
        self.work_counters["newton"] = WorkCounter()

        self.A = -1.0
        self.B = self.A

        self.dx, self.xvalues = problem_helper.get_1d_grid(self.nvars, self.bc, left_boundary=0.0, right_boundary=1.0)

        k = 2 * np.pi / self.L * np.arange(0, self.Nr)
        self.Dx = 1j * k
        self.Lx = -(k**2)

        if self.nvars % 2 == 0:
            self.Dx[-1] = 0.0

        cut = int((2 / 3) * (self.nvars // 2))
        self.dealias = np.ones(self.Nr)
        self.dealias[cut + 1 :] = 0.0
        if self.nvars % 2 == 0:
            self.dealias[-1] = 0.0

        self.dealias = self.dealias.astype(np.complex128)

        # Mask for k > 0
        idx = np.arange(self.Nr)
        self.mask_w = idx != 0
        self.mask_g3 = idx != 0

        self.I_Nr = np.eye(self.Nr)
        self.O_Nr = np.zeros((self.Nr, self.Nr))

    def transform(self, u, n=None):
        N = self.nvars if n is None else n
        return rfft(u, n=N)

    def itransform(self, u, n=None):
        N = self.nvars if n is None else n
        return irfft(u, n=N)

    def u_ex(self, t, x_deriv, t_deriv):
        r"""
        Returns exact solution and its derivatives for :math:`u`.

        Parameters
        ----------
        t : float
            Current time.
        x_deriv : int
            Number of spatial derivative. Can be 0, 1, 2.
        t_deriv : int
            Number of time derivative. Can be 0 or 1.

        Returns
        -------
          : np.1darray
            Exact solution of :math:`u`.
        """

        if x_deriv == 0 and t_deriv == 0:
            return self.A * np.sin(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 1 and t_deriv == 0:
            return self.A * 2 * np.pi * np.cos(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 2 and t_deriv == 0:
            return -self.A * (2 * np.pi) ** 2 * np.sin(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 0 and t_deriv == 1:
            return self.A * np.sin(2 * np.pi * self.xvalues) * np.exp(t)

    def v_ex(self, t, x_deriv, t_deriv):
        r"""
        Returns exact solution and its derivatives for :math:`v`.

        Parameters
        ----------
        t : float
            Current time.
        x_deriv : int
            Number of spatial derivative. Can be 0, 1, 2.
        t_deriv : int
            Number of time derivative. Can be 0 or 1.

        Returns
        -------
          : np.1darray
            Exact solution of :math:`v`.
        """

        if x_deriv == 0 and t_deriv == 0:
            return self.B * np.sin(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 1 and t_deriv == 0:
            return self.B * 2 * np.pi * np.cos(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 2 and t_deriv == 0:
            return -self.B * (2 * np.pi) ** 2 * np.sin(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 0 and t_deriv == 1:
            return self.B * np.sin(2 * np.pi * self.xvalues) * np.exp(t)

    def w_ex(self, t, x_deriv, t_deriv=0):
        r"""
        Returns exact solution and its derivatives for :math:`w`.

        Parameters
        ----------
        t : float
            Current time.
        x_deriv : int
            Number of spatial derivative. Can be 0, 1, 2.
        t_deriv : int
            Number of time derivative. Can be 0 or 1.

        Returns
        -------
          : np.1darray
            Exact solution of :math:`w`.
        """

        if x_deriv == 0 and t_deriv == 0:
            return (self.A + self.B) / (4 * np.pi**2) * np.sin(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 1 and t_deriv == 0:
            return (self.A + self.B) / (2 * np.pi) * np.cos(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 2 and t_deriv == 0:
            return -(self.A + self.B) * np.sin(2 * np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 0 and t_deriv == 1:  # only for testing
            return (self.A + self.B) / (4 * np.pi**2) * np.sin(2 * np.pi * self.xvalues) * np.exp(t)

    def src_f_spectral(self, t, n):
        r"""
        Returns source term :math:`f` at time :math:`t`.

        Parameters
        ----------
        t : float
            Current time.
        n : int
            Number of unknowns.

        Returns
        -------
          : np.1darray
            Source term :math:`f`.
        """

        u = self.u_ex(t, 0, 0)
        u_xx = self.u_ex(t, 2, 0)
        ut = self.u_ex(t, 0, 1)
        wx = self.w_ex(t, 1)
        f = ut - u_xx - u * wx
        return self.transform(f, n=n)

    def src_g_spectral(self, t, n):
        r"""
        Returns source term :math:`g` at time :math:`t`.

        Parameters
        ----------
        t : float
            Current time.
        n : int
            Number of unknowns.

        Returns
        -------
          : np.1darray
            Source term :math:`g`.
        """

        v = self.v_ex(t, 0, 0)
        v_xx = self.v_ex(t, 2, 0)
        vt = self.v_ex(t, 0, 1)
        wx = self.w_ex(t, 1)
        g = vt - v_xx + v * wx
        return self.transform(g, n=n)

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
            The right-hand side of f (contains 3 * nvars components).
        """

        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]
        w_ = u.alg[: self.nvars]

        du_, dv_ = du.diff[: self.nvars], du.diff[self.nvars :]

        u_hat, v_hat, w_hat = self.transform(u_), self.transform(v_), self.transform(w_)

        du_hat, dv_hat = self.transform(du_), self.transform(dv_)

        Lu = self.Lx * u_hat
        Lv = self.Lx * v_hat
        Dw = self.Dx * w_hat
        uDw_hat = self.dealias * self.transform(u_ * self.itransform(Dw))
        vDw_hat = self.dealias * self.transform(v_ * self.itransform(Dw))

        src_f_hat = self.src_f_spectral(t=t, n=self.nvars)
        src_g_hat = self.src_g_spectral(t=t, n=self.nvars)

        tmp_f_diff1 = du_hat - Lu - uDw_hat - src_f_hat
        tmp_f_diff2 = dv_hat - Lv + vDw_hat - src_g_hat

        f = self.dtype_f(self.init)
        f.diff[: self.nvars] = self.itransform(tmp_f_diff1)
        f.diff[self.nvars :] = self.itransform(tmp_f_diff2)

        g = self.algebraic_constraints(u, t)
        f.alg[: self.nvars] = g
        self.work_counters["rhs"]()
        return f

    def algebraic_constraints(self, u, t):
        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]
        w_ = u.alg[: self.nvars]

        u_hat, v_hat, w_hat = self.transform(u_), self.transform(v_), self.transform(w_)

        f = self.dtype_f(self.init)
        g_hat = -u_hat - v_hat - self.Lx * w_hat
        f.alg[: self.nvars] = self.itransform(g_hat)
        return f.alg[: self.nvars]

    def _make_mult_operator_wx(self, wx_phys, n):
        """
        Builds Jacobian part of uDw_hat, vDw_hat with respect to u.

        Parameter
        ---------
        wx : np.1darray
            Vector in physical/real space.
        n : int
            Number of unknowns.

        Returns
        -------
        stacked : np.2darray
            Block of Jacobian part.
        """

        cols = []
        for j in range(self.Nr):
            e_hat = np.zeros(self.Nr)
            e_hat[j] = 1.0
            e = self.itransform(e_hat, n=n)
            z = e * wx_phys
            cols.append(self.dealias * self.transform(z, n=n))
        M = np.column_stack(cols).astype(np.complex128, copy=False)
        return M

    def _make_mult_operator_uDx(self, u_phys, n):
        """
        Builds Jacobian part of uDw_hat, vDw_hat with respect to w.

        Parameter
        ---------
        u : np.1darray
            Vector in physical/real space.
        n : int
            Number of unknowns.

        Returns
        -------
        stacked : np.2darray
            Block of Jacobian part.
        """

        u_phys = np.asarray(np.real_if_close(u_phys), dtype=float).ravel()

        cols = []
        for j in range(self.Nr):
            e = np.zeros(self.Nr)
            e[j] = 1.0
            Dz_phys = self.itransform(self.Dx * e, n=n)
            cols.append(self.dealias * self.transform(u_phys * Dz_phys, n=n))
        M = np.column_stack(cols).astype(np.complex128, copy=False)
        return M

    def g_hat(self, factor, rhs_hat, t, u_hat_):
        u_hat, v_hat = u_hat_.diff[: self.Nr], u_hat_.diff[self.Nr :]
        w_hat = u_hat_.alg[: self.Nr]

        u_, v_ = self.itransform(u_hat), self.itransform(v_hat)

        rhs_u_hat, rhs_v_hat = rhs_hat.diff[: self.Nr], rhs_hat.diff[self.Nr :]
        rhs_w_hat = rhs_hat.alg[: self.Nr]

        rhs_u, rhs_v = self.itransform(rhs_u_hat), self.itransform(rhs_v_hat)

        Lu_rhs = self.Lx * (rhs_u_hat + factor * u_hat)
        Lv_rhs = self.Lx * (rhs_v_hat + factor * v_hat)
        Dw_rhs = self.Dx * (rhs_w_hat + factor * w_hat)
        Lw_rhs = self.Lx * (rhs_w_hat + factor * w_hat)
        uDw_hat = self.dealias * self.transform((rhs_u + factor * u_) * self.itransform(Dw_rhs))
        vDw_hat = self.dealias * self.transform((rhs_v + factor * v_) * self.itransform(Dw_rhs))

        src_f_hat = self.src_f_spectral(t, n=self.nvars)
        src_g_hat = self.src_g_spectral(t, n=self.nvars)

        g1_hat = u_hat - Lu_rhs - uDw_hat - src_f_hat
        g2_hat = v_hat - Lv_rhs + vDw_hat - src_g_hat
        g3_hat = -(rhs_u_hat + factor * u_hat) - (rhs_v_hat + factor * v_hat) - Lw_rhs

        return np.concatenate((g1_hat, g2_hat, g3_hat))

    def g_phys(self, factor, rhs, t, u):
        rhs_u, rhs_v = rhs.diff[: self.nvars], rhs.diff[self.nvars :]
        rhs_w = rhs.alg[: self.nvars]

        rhs_u_hat, rhs_v_hat = self.transform(rhs_u), self.transform(rhs_v)
        rhs_w_hat = self.transform(rhs_w)

        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]
        w_ = u.alg[: self.nvars]

        u_hat, v_hat = self.transform(u_), self.transform(v_)
        w_hat = self.transform(w_)

        Lu_rhs = self.Lx * (rhs_u_hat + factor * u_hat)
        Lv_rhs = self.Lx * (rhs_v_hat + factor * v_hat)
        Dw_rhs = self.Dx * (rhs_w_hat + factor * w_hat)
        Lw_rhs = self.Lx * (rhs_w_hat + factor * w_hat)
        uDw_hat = self.dealias * self.transform((rhs_u + factor * u_) * self.itransform(Dw_rhs))
        vDw_hat = self.dealias * self.transform((rhs_v + factor * v_) * self.itransform(Dw_rhs))

        src_f_hat = self.src_f_spectral(t, n=self.nvars)
        src_g_hat = self.src_g_spectral(t, n=self.nvars)

        g1_hat = u_hat - Lu_rhs - uDw_hat - src_f_hat
        g2_hat = v_hat - Lv_rhs + vDw_hat - src_g_hat
        g3_hat = -(rhs_u_hat + factor * u_hat) - (rhs_v_hat + factor * v_hat) - Lw_rhs

        return np.concatenate((self.itransform(g1_hat), self.itransform(g2_hat), self.itransform(g3_hat)))

    def dg_hat(self, factor, rhs, u):
        """
        Computes Jacobian for PDAE system. The DC mode in w (i.e., k = 0 in Fourier coefficients) is
        the average of w being constant and is thus removed in the numerical solution and in the algebraic equation.
        """

        # Shortcuts
        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]
        w_ = u.alg[: self.nvars]

        rhs_u, rhs_v = rhs.diff[: self.nvars], rhs.diff[self.nvars :]
        rhs_w = rhs.alg[: self.nvars]

        rhs_w_hat = self.transform(rhs_w)

        w_hat = self.transform(w_)

        w_hat_approx = rhs_w_hat + factor * w_hat
        wx = self.itransform(self.Dx * w_hat_approx)

        mult_op_wx = self._make_mult_operator_wx(wx, n=self.nvars)

        u_approx, v_approx = rhs_u + factor * u_, rhs_v + factor * v_
        mult_op_uDx = self._make_mult_operator_uDx(u_approx, n=self.nvars)
        mult_op_vDx = self._make_mult_operator_uDx(v_approx, n=self.nvars)

        # Blocks for Jacobian
        J11 = self.I_Nr - factor * np.diag(self.Lx) - factor * mult_op_wx
        J22 = self.I_Nr - factor * np.diag(self.Lx) + factor * mult_op_wx
        J13 = -factor * mult_op_uDx
        J23 = factor * mult_op_vDx
        J31 = -factor * self.I_Nr
        J32 = -factor * self.I_Nr
        J33 = -factor * np.diag(self.Lx)

        # Remove DC mode (k = 0 in Fourier coefficients) in variables ..
        J13_red = J13[:, self.mask_w]
        J23_red = J23[:, self.mask_w]
        J33_red = J33[np.ix_(self.mask_g3, self.mask_w)]

        # .. and equations via mask
        J31_red = J31[self.mask_g3, :]
        J32_red = J32[self.mask_g3, :]

        J = np.block(
            [
                [J11, self.O_Nr, J13_red],
                [self.O_Nr, J22, J23_red],
                [J31_red, J32_red, J33_red],
            ]
        )

        return J

    def solve_system(self, impl_sys, rhs, factor, u0, t):
        r"""
        Dispatcher that selects where the solution is computed based on ``self.spectral``.
        If ``self.spectral`` is ``True``, the solution is computed in spectral space, otherwise,
        in physical space (i.e., FFT is applied, and the it is transformed back to physical space)

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
            Numerical solution of the nonlinear system (in physical space).
        """

        if self.spectral:
            return self.solve_in_spectral_space(impl_sys, rhs, factor, u0, t)
        else:
            return self.solve_in_physical_space(impl_sys, rhs, factor, u0, t)

    def solve_in_spectral_space(self, impl_sys, rhs, factor, u0, t):
        u = self.dtype_u(u0)

        u_hat_ = self.dtype_u(self.spectral_init)
        rhs_hat = self.dtype_u(self.spectral_init)

        rhs_u, rhs_v = rhs.diff[: self.nvars], rhs.diff[self.nvars :]
        rhs_w = rhs.alg[: self.nvars]

        rhs_u_hat, rhs_v_hat = self.transform(rhs_u), self.transform(rhs_v)
        rhs_w_hat = self.transform(rhs_w)

        rhs_hat.diff[: self.Nr], rhs_hat.diff[self.Nr :] = rhs_u_hat, rhs_v_hat
        rhs_hat.alg[: self.Nr] = rhs_w_hat

        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Shortcuts
            u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]
            w_ = u.alg[: self.nvars]

            u_hat, v_hat = self.transform(u_), self.transform(v_)
            w_hat = self.transform(w_)

            u_hat_.diff[: self.Nr], u_hat_.diff[self.Nr :] = u_hat[:], v_hat[:]
            u_hat_.alg[: self.Nr] = w_hat[:]

            g_hat = self.g_hat(factor, rhs_hat, t, u_hat_)

            # If g is close to 0, then we are done
            g = np.concatenate(
                (self.itransform(g_hat[: self.Nr]),
                self.itransform(g_hat[self.Nr : 2 * self.Nr]),
                self.itransform(g_hat[2 * self.Nr : 3 * self.Nr])),
            )

            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                g_val = g[2 * self.N : 3 * self.N]
                break

            # Apply mask to g_hat
            g3_hat = g_hat[2 * self.Nr :]
            g3_red = g3_hat[self.mask_g3]
            g_hat = np.concatenate((g_hat[: 2 * self.Nr], g3_red))

            dg_hat = self.dg_hat(factor, rhs, u)

            # J_ana_red = dg_hat.copy()

            # J_fd_full = self.fd_jacobian_full_from_g_hat(factor, self.g_hat, rhs_hat, t, u_hat_)
            # J_fd_red  = self.reduce_full_J(J_fd_full)

            # Vergleich:
            # self.compare_blocks(J_ana_red, J_fd_red)

            # self.compare_blocks_with_expected_output(factor, J_ana_red)

            # Compute Newton direction in spectral space
            dx_hat = np.linalg.solve(dg_hat, g_hat)

            du_hat = dx_hat[: self.Nr]
            dv_hat = dx_hat[self.Nr : 2 * self.Nr]
            dw_hat_tmp = dx_hat[2 * self.Nr :]
            dw_hat = np.zeros(self.Nr, dtype=complex)
            dw_hat[self.mask_w] = dw_hat_tmp

            # Update in spectral space
            u_hat[:] -= du_hat[:]
            v_hat[:] -= dv_hat[:]
            w_hat[:] -= dw_hat[:]

            u.diff[: self.nvars] = self.itransform(u_hat)
            u.diff[self.nvars :] = self.itransform(v_hat)
            u.alg[: self.nvars] = self.itransform(w_hat)

            # Increase iteration per one
            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter and self.verbose:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        u_hat_.diff[: self.Nr], u_hat_.diff[self.Nr :] = u_hat, v_hat
        u_hat_.alg[: self.Nr] = w_hat

        g_hat = self.g_hat(factor, rhs_hat, t, u_hat_)

        # If g is close to 0, then we are done
        g = np.concatenate(
            (self.itransform(g_hat[: self.Nr]),
             self.itransform(g_hat[self.Nr : 2 * self.Nr]),
             self.itransform(g_hat[2 * self.Nr : 3 * self.Nr])),
        )
        g_val = g[2 * self.N : 3 * self.N]
        self.store_g_after_newton(g_val)

        # print(n, np.linalg.norm(g[: self.N], np.inf), np.linalg.norm(g[self.N : 2 * self.N], np.inf), np.linalg.norm(g[2 * self.N : 3 * self.N], np.inf))
        # print()

        solution = self.dtype_u(self.init)
        solution[:] = u[:]
        return solution

    def store_g_after_newton(self, g_val):
        """Stores absolute value of algebraic constraints g (in physical space) that is solved in Newton."""
        self.g_val_newton = max(abs(g_val))

    def solve_in_physical_space(self, impl_sys, rhs, factor, u0, t):
        u = self.dtype_u(u0)

        n = 0
        res = 99
        while n < self.newton_maxiter:
            g_phys = self.g_phys(factor, rhs, t, u)

            res = np.linalg.norm(g_phys, np.inf)
            if res < self.newton_tol:
                break

            dg_phys = self.fd_jacobian_from_g_phys(factor, self.g_phys, rhs, t, u)

            # dx_phys = np.linalg.solve(dg_phys, g_phys)
            dx_phys = np.linalg.lstsq(dg_phys, g_phys)[0]

            du_phys = dx_phys[: self.nvars]
            dv_phys = dx_phys[self.nvars : 2 * self.nvars]
            dw_phys = dx_phys[2 * self.nvars :]

            # Update in physical space
            u.diff[: self.nvars] -= du_phys
            u.diff[self.nvars :] -= dv_phys
            u.alg[: self.nvars] -= dw_phys

            # Increase iteration per one
            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter and self.verbose:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        solution = self.dtype_u(self.init)
        solution[:] = u[:]
        return solution

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

        u_ex = self.dtype_u(self.init, val=0.0)
        u_ex.diff[: self.nvars] = self.u_ex(t, x_deriv=0, t_deriv=0)
        u_ex.diff[self.nvars :] = self.v_ex(t, x_deriv=0, t_deriv=0)
        u_ex.alg[: self.nvars] = self.w_ex(t, x_deriv=0)
        return u_ex

    def du_exact(self, t):
        r"""
        Routine for the derivative of exact solution at time :math:`t`.
        Required for Runge-Kutta methods.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        du_ex : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Derivative of exact solution.
        """

        return self.u_exact(t)


class ReactionDiffusionPDAE_Radau(ReactionDiffusionPDAE, ProblemDAE):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        bc="periodic",
        L=1,
        newton_tol=1e-14,
        newton_maxiter=10,
        nvars=4,
        stop_at_maxiter=False,
        stop_at_nan=False,
        verbose=False,
        comm=None,
    ):
        """Initialization routine"""

        self._makeAttributeAndRegister(
            "bc",
            "L",
            "comm",
            "newton_tol",
            "newton_maxiter",
            "nvars",
            "stop_at_maxiter",
            "stop_at_nan",
            "verbose",
            localVars=locals(),
        )

        if self.nvars % 2 != 0:
            raise ProblemError("setup requires nvars = 2^n!")
        
        self.N = nvars
        self.Nr = self.N // 2 + 1
        self.Nr_all = 3 * self.Nr

        # Initialize problem with number of unknowns in spectral space
        ProblemDAE.__init__(self, nvars=3 * self.Nr, newton_tol=newton_tol)

        self.work_counters["rhs"] = WorkCounter()
        self.work_counters["newton"] = WorkCounter()

        self.A = -1.0
        self.B = self.A

        self.dx, self.xvalues = problem_helper.get_1d_grid(self.N, self.bc, left_boundary=0.0, right_boundary=1.0)

        k = 2 * np.pi / self.L * np.arange(0, self.Nr)
        self.Dx = 1j * k
        self.Lx = -(k**2)

        if self.N % 2 == 0:
            self.Dx[-1] = 0.0

        cut = int((2 / 3) * (self.N // 2))
        self.dealias = np.ones(self.Nr)
        self.dealias[cut + 1 :] = 0.0
        if self.N % 2 == 0:
            self.dealias[-1] = 0.0

        self.dealias = self.dealias.astype(np.complex128)

        # Mask for k > 0
        idx = np.arange(self.Nr)
        self.mask_w = idx != 0
        self.mask_g3 = idx != 0

        # Entire mask for solution vector - DC mode is removed in w
        self.mask_x = np.ones(3 * self.Nr, dtype=bool)
        self.mask_x[2 * self.Nr + 0] = False

        # Entire mask for function g - DC mode is removed in g3
        self.mask_g = np.ones(3 * self.Nr, dtype=bool)
        self.mask_g[2 * self.Nr + 0] = False

        self.init = ((3 * self.Nr,), self.comm, np.dtype("complex128"))
        self.physical_init = ((3 * self.N,), self.comm, np.dtype("complex128"))

        self.I_Nr = np.eye(self.Nr, dtype=complex)
        self.O_Nr = np.zeros((self.Nr, self.Nr), dtype=complex)

    def src_f_spectral(self, t, n):
        """See documentation of parent class."""
        return ReactionDiffusionPDAE.src_f_spectral(self, t, n)

    def src_g_spectral(self, t, n):
        """See documentation of parent class."""
        return ReactionDiffusionPDAE.src_g_spectral(self, t, n)

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.
        The right-hand side of the implicit function is returned in spectral space.

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
            The right-hand side of f (contains 3 * nvars components).
        """

        u_hat, v_hat = u[: self.Nr], u[self.Nr : 2 * self.Nr]
        w_hat = u[2 * self.Nr : 3 * self.Nr]

        u_, v_ = self.itransform(u_hat, n=self.N), self.itransform(v_hat, n=self.N)

        du_hat, dv_hat = du[: self.Nr], du[self.Nr : 2 * self.Nr]

        Lu = self.Lx * u_hat
        Lv = self.Lx * v_hat
        Dw = self.Dx * w_hat
        Lw = self.Lx * w_hat
        uDw_hat = self.dealias * self.transform(u_ * self.itransform(Dw, n=self.N), n=self.N)
        vDw_hat = self.dealias * self.transform(v_ * self.itransform(Dw, n=self.N), n=self.N)

        src_f_hat = self.src_f_spectral(t=t, n=self.N)
        src_g_hat = self.src_g_spectral(t=t, n=self.N)

        f1_hat = du_hat - Lu - uDw_hat - src_f_hat
        f2_hat = dv_hat - Lv + vDw_hat - src_g_hat
        f3_hat = -u_hat - v_hat - Lw

        f_hat = self.dtype_f(self.init)
        f_hat[: self.Nr] = f1_hat
        f_hat[self.Nr : 2 * self.Nr] = f2_hat
        f_hat[2 * self.Nr : 3 * self.Nr] = f3_hat
        self.work_counters["rhs"]()
        return f_hat

    def dg_hat(self, dt, du, M, Qmat, u0_full):
        """
        Returns Jacobian of implicit collocation system in spectral space.

        Parameters
        ----------
        M : int

        Returns
        -------
        J : np.2darray
            Jacobian.
        """

        n_j = self.Nr_all - 1
        J = np.zeros((n_j * M, n_j * M), dtype=complex)

        u_approx_full, v_approx_full = np.zeros((self.N, M), dtype=complex), np.zeros((self.N, M), dtype=complex)
        w_hat_approx_full = np.zeros((self.Nr, M), dtype=complex)

        # Shortcuts
        u0 = u0_full[0]
        u0_hat, v0_hat = u0[: self.Nr], u0[self.Nr : 2 * self.Nr]
        w0_hat = u0[2 * self.Nr : 3 * self.Nr]

        u0_, v0_ = self.itransform(u0_hat, n=self.N), self.itransform(v0_hat, n=self.N)

        # Product u * Dw for first rows
        for j in range(M):
            w_hat_approx_full[:, j] = w0_hat.copy()
            for m in range(M):
                w_hat_approx_full[:, j] += dt * Qmat[j + 1, m + 1] * du[m][2 * self.Nr : 3 * self.Nr]

        for j in range(M):
            u_approx_full[:, j], v_approx_full[:, j] = u0_.copy(), v0_.copy()
            for m in range(M):
                u_approx_full[:, j] += dt * Qmat[j + 1, m + 1] * self.itransform(du[m][: self.Nr], n=self.N)
                v_approx_full[:, j] += dt * Qmat[j + 1, m + 1] * self.itransform(du[m][self.Nr : 2 * self.Nr], n=self.N)

        for m in range(M):
            wx = self.itransform(self.Dx * w_hat_approx_full[:, m], n=self.N)
            mult_op_wx = self._make_mult_operator_wx(wx, n=self.N)

            mult_op_uDx = self._make_mult_operator_uDx(u_approx_full[:, m], n=self.N)
            mult_op_vDx = self._make_mult_operator_uDx(v_approx_full[:, m], n=self.N)

            for j in range(M):
                q_mj = Qmat[m + 1, j + 1]

                J1 = (
                    self.I_Nr - dt * q_mj * (np.diag(self.Lx) + mult_op_wx)
                    if m == j
                    else -dt * q_mj * (np.diag(self.Lx) + mult_op_wx)
                )
                J2 = self.O_Nr
                J3 = -dt * q_mj * mult_op_uDx
                J4 = self.O_Nr
                J5 = (
                    self.I_Nr - dt * q_mj * (np.diag(self.Lx) - mult_op_wx)
                    if m == j
                    else -dt * q_mj * (np.diag(self.Lx) - mult_op_wx)
                )
                J6 = dt * q_mj * mult_op_vDx
                J7 = -dt * q_mj * self.I_Nr
                J8 = -dt * q_mj * self.I_Nr
                J9 = -dt * q_mj * np.diag(self.Lx)

                # Remove DC mode
                J3_red = J3[:, self.mask_w]
                J6_red = J6[:, self.mask_w]
                J9_red = J9[np.ix_(self.mask_g3, self.mask_w)]

                # .. and g3 via mask
                J7_red = J7[self.mask_g3, :]
                J8_red = J8[self.mask_g3, :]

                J_block = np.block(
                    [
                        [J1, J2, J3_red],
                        [J4, J5, J6_red],
                        [J7_red, J8_red, J9_red],
                    ]
                )

                J[m * n_j : (m + 1) * n_j, j * n_j : (j + 1) * n_j] = J_block

        return J

    def _apply_mask_to_g(self, g, M):
        g_red = []
        for m in range(M):
            g_piece = g[m * self.nvars : (m + 1) * self.nvars]
            g_red.append(g_piece[self.mask_g])
        return np.concatenate([g_val.flatten() for g_val in g_red])

    def _add_dc_mode(self, dx_matrix, M):
        dx_matrix_dc = np.zeros((M, self.Nr_all), dtype=complex)
        for m in range(M):
            dx_matrix_col = dx_matrix[m, :]
            col = np.zeros(self.Nr_all, dtype=complex)
            col[self.mask_x] = dx_matrix_col
            dx_matrix_dc[m, :] = col

        return dx_matrix_dc

    def _get_residual_in_physical_space(self, g_hat, M):
        n_phys = 3 * self.N
        g_physical = np.zeros(M * n_phys)
        for m in range(M):
            g_piece = g_hat[m * self.Nr_all : (m + 1) * self.Nr_all]
            g1, g2 = self.itransform(g_piece[: self.Nr], n=self.N), self.itransform(
                g_piece[self.Nr : 2 * self.Nr], n=self.N
            )
            g3 = self.itransform(g_piece[2 * self.Nr : 3 * self.Nr], n=self.N)
            g_physical[m * n_phys : (m + 1) * n_phys] = np.concatenate((g1, g2, g3))
        return np.linalg.norm(g_physical, np.inf)

    def solve_collocation_system(self, F, f_init, t, dt, M, Qmat, sweep, u0_full):
        """Solves the collocation system for Radau solver."""

        def g_hat_fun(du, **kwargs):
            return F(du, t, dt, M, self, Qmat, sweep, u0_full, **kwargs)

        du = f_init.copy()

        n = 0
        res = 99
        while n < self.newton_maxiter:
            g_hat = g_hat_fun(du)

            # If g is close to 0, then we are done
            res = self._get_residual_in_physical_space(g_hat, M)
            if res < self.newton_tol:
                break

            g_red = self._apply_mask_to_g(g_hat, M)

            # Assemble dg
            dg_hat = self.dg_hat(dt, du, M, Qmat, u0_full)

            # Newton direction dx
            dx_hat = np.linalg.solve(dg_hat, g_red)

            # Newton update: u1 = u0 - g/dg
            dx_matrix = dx_hat.reshape((M, self.nvars - 1))
            dx_matrix_dc = self._add_dc_mode(dx_matrix, M)
            du_reshape = [self.dtype_f(du_m) - dx_m for du_m, dx_m in zip(du, dx_matrix_dc)]

            du = du_reshape.copy()

            # Increase iteration per one
            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter and self.verbose:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        return du

    def u_exact(self, t, **kwargs):
        r"""
        Returns exact solution at time :math:`t` in spectral space.

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
        me[: self.Nr] = self.transform(self.u_ex(t, 0, 0), n=self.N)
        me[self.Nr : 2 * self.Nr] = self.transform(self.v_ex(t, 0, 0), n=self.N)
        me[2 * self.Nr : 3 * self.Nr] = self.transform(self.w_ex(t, 0), n=self.N)
        return me

    def du_exact(self, t):
        r"""
        Returns derivative of exact solution at time :math:`t` in spectral space.
        Required for Runge-Kutta methods.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        du_ex : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Derivative of exact solution.
        """

        return self.u_exact(t)


class SemiImplicitReactionDiffusionPDAE(ReactionDiffusionPDAE):

    def g_phys(self, factor, rhs, t, u):
        rhs_u, rhs_v = rhs.diff[: self.nvars], rhs.diff[self.nvars :]

        rhs_u_hat, rhs_v_hat = self.transform(rhs_u), self.transform(rhs_v)

        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]
        w_ = u.alg[: self.nvars]

        u_hat, v_hat = self.transform(u_), self.transform(v_)
        w_hat = self.transform(w_)

        Lu_rhs = self.Lx * (rhs_u_hat + factor * u_hat)
        Lv_rhs = self.Lx * (rhs_v_hat + factor * v_hat)
        Dw = self.Dx * w_hat
        Lw = self.Lx * w_hat
        uDw_hat = self.dealias * self.transform((rhs_u + factor * u_) * self.itransform(Dw))
        vDw_hat = self.dealias * self.transform((rhs_v + factor * v_) * self.itransform(Dw))

        src_f_hat = self.src_f_spectral(t=t, n=self.nvars)
        src_g_hat = self.src_g_spectral(t=t, n=self.nvars)

        g1_hat = u_hat - Lu_rhs - uDw_hat - src_f_hat
        g2_hat = v_hat - Lv_rhs + vDw_hat - src_g_hat
        g3_hat = -(rhs_u_hat + factor * u_hat) - (rhs_v_hat + factor * v_hat) - Lw

        return np.concatenate((self.itransform(g1_hat), self.itransform(g2_hat), self.itransform(g3_hat)))

    def g_hat(self, factor, rhs_hat, t, u_hat_):
        u_hat, v_hat = u_hat_.diff[: self.Nr], u_hat_.diff[self.Nr :]
        w_hat = u_hat_.alg[: self.Nr]

        u_, v_ = self.itransform(u_hat), self.itransform(v_hat)

        rhs_u_hat, rhs_v_hat = rhs_hat.diff[: self.Nr], rhs_hat.diff[self.Nr :]
        rhs_u, rhs_v = self.itransform(rhs_u_hat), self.itransform(rhs_v_hat)

        Lu_rhs = self.Lx * (rhs_u_hat + factor * u_hat)
        Lv_rhs = self.Lx * (rhs_v_hat + factor * v_hat)
        Dw = self.Dx * w_hat
        Lw = self.Lx * w_hat
        uDw_hat = self.dealias * self.transform((rhs_u + factor * u_) * self.itransform(Dw))
        vDw_hat = self.dealias * self.transform((rhs_v + factor * v_) * self.itransform(Dw))

        src_f_hat = self.src_f_spectral(t=t, n=self.nvars)
        src_g_hat = self.src_g_spectral(t=t, n=self.nvars)

        g1_hat = u_hat - Lu_rhs - uDw_hat - src_f_hat
        g2_hat = v_hat - Lv_rhs + vDw_hat - src_g_hat
        g3_hat = -(rhs_u_hat + factor * u_hat) - (rhs_v_hat + factor * v_hat) - Lw

        return np.concatenate((g1_hat, g2_hat, g3_hat))

    def dg_hat(self, factor, rhs, u):
        """
        Computes Jacobian for PDAE system. The DC mode in w (i.e., k = 0 in Fourier coefficients) is
        the average of w being constant and is thus removed in the numerical solution and in the algebraic equation.
        """

        # Shortcuts
        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]
        w_ = u.alg[: self.nvars]

        rhs_u, rhs_v = rhs.diff[: self.nvars], rhs.diff[self.nvars :]

        w_hat = self.transform(w_)

        wx = self.itransform(self.Dx * w_hat)
        mult_op_wx = self._make_mult_operator_wx(wx, n=self.nvars)

        u_approx, v_approx = rhs_u + factor * u_, rhs_v + factor * v_
        mult_op_uDx = self._make_mult_operator_uDx(u_approx, n=self.nvars)
        mult_op_vDx = self._make_mult_operator_uDx(v_approx, n=self.nvars)

        # Blocks for Jacobian
        J11 = self.I_Nr - factor * np.diag(self.Lx) - factor * mult_op_wx
        J22 = self.I_Nr - factor * np.diag(self.Lx) + factor * mult_op_wx
        J13 = -mult_op_uDx
        J23 = mult_op_vDx
        J31 = -factor * self.I_Nr
        J32 = -factor * self.I_Nr
        J33 = -np.diag(self.Lx)

        # Remove DC mode (k = 0 in Fourier coefficients) in variables ..
        J13_red = J13[:, self.mask_w]
        J23_red = J23[:, self.mask_w]
        J33_red = J33[np.ix_(self.mask_g3, self.mask_w)]

        # .. and equations via mask
        J31_red = J31[self.mask_g3, :]
        J32_red = J32[self.mask_g3, :]

        J = np.block(
            [
                [J11, self.O_Nr, J13_red],
                [self.O_Nr, J22, J23_red],
                [J31_red, J32_red, J33_red],
            ]
        )

        return J


class ReactionDiffusionPDAEConstrained(ReactionDiffusionPDAE):
    """Constrained formulation where only the differential equations are integrated numerically."""

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
            The right-hand side of f (contains ``3 * nvars`` components).
        """

        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]
        w_ = u.alg[: self.nvars]

        u_hat, v_hat, w_hat = self.transform(u_), self.transform(v_), self.transform(w_)

        Lu = self.Lx * u_hat
        Lv = self.Lx * v_hat
        Dw = self.Dx * w_hat
        Lw = self.Lx * w_hat
        uDw_hat = self.dealias * self.transform(u_ * self.itransform(Dw))
        vDw_hat = self.dealias * self.transform(v_ * self.itransform(Dw))

        src_f = self.src_f_spectral(t=t, n=self.nvars)
        src_g = self.src_g_spectral(t=t, n=self.nvars)

        tmp_f_diff1 = Lu + uDw_hat + src_f
        tmp_f_diff2 = Lv - vDw_hat + src_g

        f = self.dtype_f(self.init)
        f.diff[: self.nvars] = self.itransform(tmp_f_diff1)
        f.diff[self.nvars :] = self.itransform(tmp_f_diff2)

        g = self.algebraic_constraints(u, t)
        f.alg[: self.nvars] = g
        self.work_counters["rhs"]()
        return f

    def g_phys(self, factor, rhs, t, u):
        rhs_u, rhs_v = rhs.diff[: self.nvars], rhs.diff[self.nvars :]

        rhs_u_hat, rhs_v_hat = self.transform(rhs_u), self.transform(rhs_v)

        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]

        u_hat, v_hat = self.transform(u_), self.transform(v_)

        f = self.eval_f(u, t)
        f1_hat = self.transform(f.diff[: self.nvars])
        f2_hat = self.transform(f.diff[self.nvars :])
        f3_hat = self.transform(f.alg[: self.nvars])

        g1_hat = u_hat - factor * f1_hat - rhs_u_hat
        g2_hat = v_hat - factor * f2_hat - rhs_v_hat
        g3_hat = f3_hat[:]

        return np.concatenate((self.itransform(g1_hat), self.itransform(g2_hat), self.itransform(g3_hat)))

    def g_hat(self, factor, rhs_hat, t, u_hat_):
        """Defines function g in spectral space to find the root for to solve system in SDC."""

        u_hat, v_hat = u_hat_.diff[: self.Nr], u_hat_.diff[self.Nr :]
        w_hat = u_hat_.alg[: self.Nr]

        rhs_u_hat, rhs_v_hat = rhs_hat.diff[: self.Nr], rhs_hat.diff[self.Nr :]

        u = self.dtype_u(self.init)
        u.diff[: self.nvars], u.diff[self.nvars :] = self.itransform(u_hat), self.itransform(v_hat)
        u.alg[: self.nvars] = self.itransform(w_hat)

        f = self.eval_f(u, t)
        f1_hat = self.transform(f.diff[: self.nvars])
        f2_hat = self.transform(f.diff[self.nvars :])
        f3_hat = self.transform(f.alg[: self.nvars])

        g1_hat = u_hat - factor * f1_hat - rhs_u_hat
        g2_hat = v_hat - factor * f2_hat - rhs_v_hat
        g3_hat = f3_hat[:]

        return np.concatenate((g1_hat, g2_hat, g3_hat))

    def dg_hat(self, factor, rhs, u, apply_mask=True):
        """
        Computes Jacobian for PDAE system. The DC mode in w (i.e., k = 0 in Fourier coefficients) is
        the average of w being constant and is thus removed in the numerical solution and in the algebraic equation
        for the linear system to be solved.
        """

        # Shortcuts
        u_, v_ = u.diff[: self.nvars], u.diff[self.nvars :]

        w_hat = self.transform(u.alg[: self.nvars])

        wx_phys = self.itransform(self.Dx * w_hat).real

        mult_op_wx = self._make_mult_operator_wx(wx_phys, n=self.nvars)
        mult_op_uDx = self._make_mult_operator_uDx(u_, n=self.nvars)
        mult_op_vDx = self._make_mult_operator_uDx(v_, n=self.nvars)

        # Blocks for Jacobian
        J11 = self.I_Nr - factor * np.diag(self.Lx) - factor * mult_op_wx
        J22 = self.I_Nr - factor * np.diag(self.Lx) + factor * mult_op_wx
        J13 = -factor * mult_op_uDx
        J23 = factor * mult_op_vDx
        J31 = -self.I_Nr
        J32 = -self.I_Nr
        J33 = -np.diag(self.Lx)

        # Remove DC mode (k = 0 in Fourier coefficients) in variables and equations
        if apply_mask:
            J13_red = J13[:, self.mask_w]
            J23_red = J23[:, self.mask_w]
            J33_red = J33[np.ix_(self.mask_g3, self.mask_w)]

            J31_red = J31[self.mask_g3, :]
            J32_red = J32[self.mask_g3, :]
        else:
            J13_red = J13.copy()
            J23_red = J23.copy()
            J33_red = J33.copy()

            J31_red = J31.copy()
            J32_red = J32.copy()

        J = np.block(
            [
                [J11, self.O_Nr, J13_red],
                [self.O_Nr, J22, J23_red],
                [J31_red, J32_red, J33_red],
            ]
        )

        return J
    
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
