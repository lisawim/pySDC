import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


class BuckConverter_DAE(ptype_dae):
    """
    Example implementing the buck converter model in DAE description
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """Initialization routine"""

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.duty = 0.5
        self.fsw = 1e3
        self.Vs = 10.0
        self.Rs = 0.5
        self.C1 = 1e-3
        self.Rp = 0.01
        self.Lp = 1e-3
        self.C2 = 1 #1e-3
        self.Rl = 10

    def eval_f(self, u, du, t):

        vRs, vC1, vRp, vLp, vC2, vRl = u[0], u[1], u[2], u[3], u[4], u[5]
        iRs, iC1, iRp, iLp, iC2, iRl = u[6], u[7], u[8], u[9], u[10], u[11]

        dvC1, dvC2, diLp = du[1], du[4], du[9]

        Tsw = 1 / self.fsw

        f = self.dtype_f(self.init)
        if 0 <= ((t / Tsw) % 1) <= self.duty:  # S1 = 1, S2 = 0
            print('If')
            f[:] = (
                vRs - self.Rs * iRs,
                dvC1 - iC1 / self.C1,
                vRp - self.Rp * iRp,
                vC1 - vRp - vLp - vC2,
                dvC2 - iC2 / self.C2,
                vRl - self.Rl * iRl,
                self.Vs - self.Rs * iRs - vC1,
                iRs - iC1 - iRp,
                iRp - iLp,
                diLp - vLp / self.Lp,
                iLp - iC2 - iRl,
                vC2 - self.Rl * iRl,
            )
        else:  # S1 = 0, S2 = 1
            print('else')
            f[:] = (
                vRs - self.Rs * iC1,
                dvC1 - iC1 / self.C1,
                vRp - self.Rp * iRp,
                vRp + vLp + vC2,
                dvC2 - iC2 / self.C2,
                vRl - self.Rl * iRl,
                self.Vs - self.Rs * iRs - vC1,
                iC1 - iRp,
                iRp - iLp,
                diLp - vLp / self.Lp,
                iLp - iC2 - iRl,
                vC2 - self.Rl * iRl,
            )

        return f

    def u_exact(self, t):
        me = self.dtype_u(self.init)
        me[0] = 0.0  # vRs
        me[1] = 0.0  # vC1
        me[2] = 0.0  # vRp
        me[3] = 0.0  # vLp
        me[4] = 0.0  # vC2
        me[5] = 0.0  # vRl
        me[6] = 0.0  # iRs
        me[7] = 0.0  # iC1
        me[8] = 0.0  # iRp
        me[9] = 0.0  # iLp
        me[10] = 0.0  # iC2
        me[11] = 0.0  # iRl
        return me
