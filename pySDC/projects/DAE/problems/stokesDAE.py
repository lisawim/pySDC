import numpy as np
from pathlib import Path
from scipy.optimize import root

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.core.errors import ProblemError
from pySDC.helpers import problem_helper


class StokesDAE(ProblemDAE):
    r"""
    This class implements the linear two-dimensional Stokes equation

    .. math::
        \frac{\partial u}{\partial t} = \nu (\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}) - \frac{\partial p}{\partial x} + f1,

    .. math::
        \frac{\partial v}{\partial t} = \nu (\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}) - \frac{\partial p}{\partial y} + f2,

    .. math::
        0 = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}

    with u(x,y,t), v(x,y,t), p(x,y,t).
    """

    def __init__(
        self,
        bc="periodic",
        lintol=1e-12,
        lin_maxiter=10000,
        nvars=16,
        nu=0.1,
        order=2,
        stencil_type="center",
        solver_type="direct",
    ):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=1e-12)
        self._makeAttributeAndRegister(
            "bc",
            "lintol",
            "lin_maxiter",
            "nvars",
            "nu",
            "stencil_type",
            "solver_type",
            localVars=locals(),
        )

        self.dx, self.xvalues = problem_helper.get_1d_grid(nvars[0], self.bc)

        # Discretisation for Laplacian
        self.A, _ = problem_helper.get_finite_difference_matrix(
            derivative=2,
            order=self.order,
            stencil_type="center",
            dx=self.dx,
            size=self.nvars[0],
            dim=2,
            bc=self.bc,
        )
