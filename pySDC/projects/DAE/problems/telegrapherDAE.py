import numpy as np
from pathlib import Path

from pySDC.core.errors import ProblemError
from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.helpers import problem_helper


class TelegrapherDAE(ProblemDAE):
    def __init__(self, bc="dirichlet", newton_tol=1e-12, newton_maxiter=100, nvars=32, order=1, stencil_type="center"):
        """Initialization routine"""
        super().__init__(nvars=nvars, newton_tol=newton_tol)

        self._makeAttributeAndRegister("bc", "newton_tol", "nvars", "order", "stencil_type", localVars=locals())

        self.work_counters["rhs"] = WorkCounter()
        self.work_counters["newton"] = WorkCounter()

        self.R = 4.0 #1
        self.L = 6.0 #1
        self.G = 2.0 #
        self.C = 1.0 # 1

        self.dx, self.xvalues = problem_helper.get_1d_grid(self.nvars, self.bc)

        # Discretisation for Laplacian
        self.A, _ = problem_helper.get_finite_difference_matrix(
            derivative=1,
            order=self.order,
            stencil_type="center",
            dx=self.dx,
            size=self.nvars + 2,
            dim=1,
            bc=self.bc,
        )

        self.uext = self.dtype_u((self.init[0] + 2, self.init[1], self.init[2]), val=0.0)

    def eval_f(self, u, du, t):

        self.uext.diff[0] = 0
        self.uext.diff[-1] = 0

        self.uext.alg[0] = 0
        self.uext.alg[-1] = 0

        self.uext.diff[1:-1] = u.diff[:]
        self.uext.alg[1:-1] = u.alg[:]

        f = self.dtype_f(self.init)
        f.diff[:] = -du.diff[:] - 1 / self.C * self.A.dot(self.uext.alg[:])[1:-1]
        f.alg[:] = -self.L * du.alg[:] - self.A.dot(self.uext.diff[:])[1:-1] - self.R * self.uext.alg[1:-1]
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
        k = np.arange(1, 1001)[:, np.newaxis]
        p_val = np.sum(np.sin(np.pi * k * self.xvalues) / (k ** 1.55), axis=0)

        u_ex.diff[:] = p_val[:]
        return u_ex
