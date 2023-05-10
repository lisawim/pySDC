import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


def e(t, E_plus, E_minus, T):
    return E_plus + ((E_plus - E_minus) / T) * t


class Maffezzoni_Example(ptype_dae):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """
        Initialization routine
        """

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.E_plus = 7.5
        self.E_minus = -2.5
        self.T = 20
        self.L = 200
        self.R = 1
        self.R_on = 0.1
        self.R_off = 1000
        self.R_diode =
        self.R_switch =

    def eval_f(self, u, du, t):
        V_1, V_2, V_3, V_4, V_diode = u[0], u[1], u[2], u[3], u[4]
        I_L, I_s, I_d = u[5], u[6], u[7]

        dV_1, dV_2, dV_3, dV_4, dV_diode = du[0], du[1], du[2], du[3], du[4]
        dI_L, dI_s, dI_d = du[5], du[6], du[7]

        e = e(t, self.E_plus, self.E_minus, self.T)

        # transition condition(s)
        if V_diode > 0:
            self.R_diode = self.R_on
        else:
            self.R_diode = self.R_off

        if V_1 - V_2 > 0:
            self.R_switch = self.R_on
        else:
            self.R_switch = self.R_off

        f = self.dtype_f(self.init)
        f[:] = (
            dI_L - V_2 + V_3,
            I_d + I_s - I_L,
            I_L - V_3 / self.R,
            V_1 - self.E,
            V_4 - e,
            V_2 - I_d * self.R_diode,
            V_1 - V_2 - I_s * self.R_switch,
            V_3 - V_4,
        )
        return f

    def u_exact(self, t):
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)
        me[0] = 0.0  # V_1
        me[1] = 0.0  # V_2
        me[2] = 0.0  # V_3
        me[3] = 0.0  # V_4
        me[4] = 0.0  # V_diode
        me[5] = 0.0  # I_L
        me[6] = 0.0  # I_s
        me[7] = 0.0  # I_d

        return me
