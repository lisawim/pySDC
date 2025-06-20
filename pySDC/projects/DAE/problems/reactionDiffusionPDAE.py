import numpy as np
from pathlib import Path

from pySDC.core.errors import ProblemError
from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.helpers import problem_helper


# def f_source(x, t):
#     return -0.1 * np.sin(np.pi * x) * np.exp(-0.05 * t)

# def g_source(x, t):
#     return -0.05 * np.sin(np.pi * x) * np.exp(-0.02 * t)


class ReactionDiffusionPDAE(ProblemDAE):
    def __init__(self, bc="dirichlet", newton_tol=1e-12, newton_maxiter=100, nvars=256):
        """Initialization routine"""
        super().__init__(nvars=2*nvars, newton_tol=newton_tol)

        self._makeAttributeAndRegister("bc", "newton_tol", "nvars", localVars=locals())

        self.work_counters["rhs"] = WorkCounter()
        self.work_counters["newton"] = WorkCounter()

        self.A = -1.0
        self.B = self.A

        self.dx, self.xvalues = problem_helper.get_1d_grid(self.nvars, self.bc)

        # Discretisation for spatial first derivative
        self.D, _ = problem_helper.get_finite_difference_matrix(
            derivative=1,
            order=2,
            stencil_type="center",
            dx=self.dx,
            size=self.nvars + 2,
            dim=1,
            bc=self.bc,
        )

        # Discretisation for Laplacian
        self.L, _ = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=2,
            stencil_type="center",
            dx=self.dx,
            size=self.nvars + 2,
            dim=1,
            bc=self.bc,
        )

        # Differential variable with boundary values
        self.uext = self.dtype_u((self.nvars + 2, self.init[1], self.init[2]), val=0.0)
        self.vext = self.dtype_u((self.nvars + 2, self.init[1], self.init[2]), val=0.0)

        # Algebraic variable with boundary values
        self.wext = self.dtype_u((self.nvars + 2, self.init[1], self.init[2]), val=0.0)

    def u_ex(self, t, x_deriv, t_deriv):
        if x_deriv == 0 and t_deriv == 0:
            return self.A * np.sin(np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 1 and t_deriv == 0:
            return self.A * np.pi * np.cos(np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 2 and t_deriv == 0:
            return -self.A * np.pi ** 2 * np.sin(np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 0 and t_deriv == 1:
            return self.A * np.sin(np.pi * self.xvalues) * np.exp(t)

    def v_ex(self, t, x_deriv, t_deriv):
        if x_deriv == 0 and t_deriv == 0:
            return self.B * np.sin(np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 1 and t_deriv == 0:
            return self.B * np.pi * np.cos(np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 2 and t_deriv == 0:
            return -self.B * np.pi ** 2 * np.sin(np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 0 and t_deriv == 1:
            return self.B * np.sin(np.pi * self.xvalues) * np.exp(t)

    def w_ex(self, t, x_deriv):
        if x_deriv == 0:
            return (self.A + self.B) / (np.pi ** 2) * np.sin(np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 1:
            return (self.A + self.B) / np.pi * np.cos(np.pi * self.xvalues) * np.exp(t)
        elif x_deriv == 2:
            return -(self.A + self.B) * np.sin(np.pi * self.xvalues) * np.exp(t)

    def f_source(self, t):
        u = self.u_ex(t, x_deriv=0, t_deriv=0)
        u_t = self.u_ex(t, x_deriv=0, t_deriv=1)
        u_xx = self.u_ex(t, x_deriv=2, t_deriv=0)
        w_x = self.w_ex(t, x_deriv=1)

        return u_t - u_xx - (u * w_x)

    def g_source(self, t):
        v = self.v_ex(t, x_deriv=0, t_deriv=0)
        v_t = self.v_ex(t, x_deriv=0, t_deriv=1)
        v_xx = self.v_ex(t, x_deriv=2, t_deriv=0)
        w_x = self.w_ex(t, x_deriv=1)

        return v_t - v_xx + (v * w_x)

    def eval_f(self, u, du, t): 
        self.uext.diff[0] = 0
        self.uext.diff[-1] = 0

        self.vext.diff[0] = 0
        self.vext.diff[-1] = 0

        self.wext.alg[0] = 0
        self.wext.alg[-1] = 0

        self.uext.diff[1:-1] = u.diff[: self.nvars]
        self.vext.diff[1:-1] = u.diff[self.nvars :]
        self.wext.alg[1:-1] = u.alg[: self.nvars]

        u_t, v_t = du.diff[: self.nvars], du.diff[self.nvars :]

        u_xx = self.L.dot(self.uext.diff[:])[1:-1]
        v_xx = self.L.dot(self.vext.diff[:])[1:-1]
        w_x = (self.D.dot(self.wext.alg[:])[1:-1])
        w_xx = self.L.dot(self.wext.alg[:])[1:-1]

        f = self.dtype_f(self.init)
        f.diff[: self.nvars] = (
            u_t
            - u_xx
            - self.uext.diff[1:-1] * w_x
            - self.f_source(t)
            # - f_source(self.xvalues, t)
        )
        f.diff[self.nvars :] = (
            v_t
            - v_xx
            + self.vext.diff[1:-1] * w_x
            - self.g_source(t)
            # - g_source(self.xvalues, t)
        )

        f.alg[: self.nvars] = (
            - self.uext.diff[1:-1]
            - self.vext.diff[1:-1]
            - w_xx
        )
        self.work_counters["rhs"]()
        return f

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
