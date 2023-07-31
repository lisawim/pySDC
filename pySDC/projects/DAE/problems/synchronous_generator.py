import numpy as np
from scipy.interpolate import interp1d
import scipy as sp
import pandapower
import pandapower.networks as nw
import pandas as pd

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


def input_torque(t):
        if round(t, 14) < 0.2:
            Tm = 0.854
        else:
            Tm = 0.854 - 0.5
        return Tm

def update_vBus(t):
    if round(t, 14) < 1:
        vBus = 0.6920 - 0.4064j  # original value
    elif 1 <= round(t, 14) < 2:
        vBus = 0.75 * (0.6920 - 0.4064j)
    else:
        vBus = 0.6920 - 0.4064j
    return vBus

def IEEE9Bus():
    mpc = {}
    mpc['bus'] = [
        [1.0000, 3.0000, 0, 0, 0, 0, 1.0000, 1.0400, 0, 345.0000, 1.0000, 1.1000, 0.9000],
        [2.0000, 2.0000, 0, 0, 0, 0, 1.0000, 1.0250, 9.2800, 345.0000, 1.0000, 1.1000, 0.9000],
        [3.0000, 2.0000, 0, 0, 0, 0, 1.0000, 1.0250, 4.6648, 345.0000, 1.0000, 1.1000, 0.9000],
        [4.0000, 1.0000, 0, 0, 0, 0, 1.0000, 1.0258, -2.2168, 345.0000, 1.0000, 1.1000, 0.9000],
        [5.0000, 1.0000, 125.0000, 50.0000, 0, 0, 1.0000, 0.9956, -3.9888, 345.0000, 1.0000, 1.1000, 0.9000],
        [6.0000, 1.0000, 90.0000, 30.0000, 0, 0, 1.0000, 1.0127, -3.6874, 345.0000, 1.0000, 1.1000, 0.9000],
        [7.0000, 1.0000, 0, 0, 0, 0, 1.0000, 1.0258, 3.7197, 345.0000, 1.0000, 1.1000, 0.9000],
        [8.0000, 1.0000, 100.0000, 35.0000, 0, 0, 1.0000, 1.0159, 0.7275, 345.0000, 1.0000, 1.1000, 0.9000],
        [9.0000, 1.0000, 0, 0, 0, 0, 1.0000, 1.0324, 1.9667, 345.0000, 1.0000, 1.1000, 0.9000],
    ]


    mpc['gen'] = [
        [1, 71.641, 27.046, 300, -300, 1.04, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 163, 6.6537, 300, -300, 1.025, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 85, -10.86, 300, -300, 1.025, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    mpc['branch'] = [
        [1, 4, 0, 0.0576, 0, 250, 250, 250, 0, 0, 1, -360, 360, 71.641, 27.046, -71.641, -23.923],
        [4, 6, 0.017, 0.092, 0.158, 250, 250, 250, 0, 0, 1, -360, 360, 30.704, 1.03, -30.537, -16.543],
        [6, 9, 0.039, 0.17, 0.358, 150, 150, 150, 0, 0, 1, -360, 360, -59.463, -13.457, 60.817, -18.075],
        [3, 9, 0, 0.0586, 0, 300, 300, 300, 0, 0, 1, -360, 360, 85, -10.86, -85, 14.955],
        [8, 9, 0.0119, 0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360, -24.095, -24.296, 24.183, 3.1195],
        [7, 8, 0.0085, 0.072, 0.149, 250, 250, 250, 0, 0, 1, -360, 360, 76.38, -0.79733, -75.905, -10.704],
        [2, 7, 0, 0.0625, 0, 250, 250, 250, 0, 0, 1, -360, 360, 163, 6.6537, -163, 9.1781],
        [5, 7, 0.032, 0.161, 0.306, 250, 250, 250, 0, 0, 1, -360, 360, -84.32, -11.313, 86.62, -8.3808],
        [4, 5, 0.01, 0.085, 0.176, 250, 250, 250, 0, 0, 1, -360, 360, 40.937, 22.893, -40.68, -38.687],
    ]

    mpc['Ybus'] = np.array([
            [0 - 17.361j, 0 + 0j, 0 + 0j, 0 + 17.361j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 - 16j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 16j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 - 17.065j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 17.065j],
            [0 + 17.361j, 0 + 0j, 0 + 0j, 3.3074 - 39.309j, -1.3652 + 11.604j, -1.9422 + 10.511j, 0 + 0j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, -1.3652 + 11.604j, 2.5528 - 17.338j, 0 + 0j, -1.1876 + 5.9751j, 0 + 0j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, -1.9422 + 10.511j, 0 + 0j, 3.2242 - 15.841j, 0 + 0j, 0 + 0j, -1.282 + 5.5882j],
            [0 + 0j, 0 + 16j, 0 + 0j, 0 + 0j, -1.1876 + 5.9751j, 0 + 0j, 2.8047 - 35.446j, -1.6171 + 13.698j, 0 + 0j],
            [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, -1.6171 + 13.698j, 2.7722 - 23.303j, -1.1551 + 9.7843j],
            [0 + 0j, 0 + 0j, 0 + 17.065j, 0 + 0j, 0 + 0j, -1.282 + 5.5882j, 0 + 0j, -1.1551 + 9.7843j, 2.4371 - 32.154j]
        ], dtype=complex)

    return mpc


class SynchronousGenerator(ptype_dae):
    """
    Example implementing the synchronous generator model from PinTSimE project
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """
        Initialization routine
        """

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.Ld = 1.8099
        self.Lq = 1.76
        self.LF = 1.824652
        self.LD = 1.83149
        self.LQ1 = 2.578217
        self.LQ2 = 1.729817
        self.Lmd = 1.6599
        self.Lmq = 1.61
        self.Rs = 3 * 1e-3
        self.R_F = 0.000742259
        self.R_D = 0.0333636
        self.R_Q1 = 0.0104167
        self.R_Q2 = 2.5953e-2
        self.H = 6
        self.Kd = 0.0
        self.wb = 100 * np.pi
        #self.vbus = 0.5419999999999999 - 0.4064j
        self.v_F = 0.00108242466673301
        self.R_shunt = 1e9
        self.set_switch = False
        self.V_g = 0.6920 - 0.4064j
        self.G = np.real(1 / (0.1 + 0.5j))
        self.B = np.imag(1 / (0.1 + 0.5j))

    def eval_f(self, u, du, t):
        """
        Routine to evaluate implicit representation of the problem
        """

        # mechanical torque
        Tm = 0.9030

        phi_d, phi_q, phi_F, phi_D, phi_Q1, phi_Q2 = u[0], u[1], u[2], u[3], u[4], u[5]
        i_d, i_q, i_F, i_D, i_Q1, i_Q2 = u[6], u[7], u[8], u[9], u[10], u[11]
        omega_m, delta_r = u[12], u[13]
        v_d, v_q = u[14], u[15]
        V_re, V_im, I_re, I_im = u[16], u[17], u[18], u[19]

        dphi_d, dphi_q, dphi_F, dphi_D, dphi_Q1, dphi_Q2 = du[0], du[1], du[2], du[3], du[4], du[5]
        domega_m, ddelta_r = du[12], du[13]

        self.V_g = update_vBus(t)
        #V = 1.0
        #print(t, omega_m * self.wb / (2*np.pi))
        #if (omega_m * self.wb / (2*np.pi) < 49 or omega_m * self.wb / (2*np.pi) > 51) and self.set_switch == False:
        #V = self.R_shunt * (-1) * (re_I + 1j * im_I) # where R_shunt = 1e9
            #self.set_switch = True
            #print(t)
        #else:
        #V = self.vbus - self.Zline * (-1) * (re_I + 1j * im_I)

        f = self.dtype_f(self.init)
        f[:] = (
            -dphi_d + self.wb * (v_d + self.Rs * i_d + omega_m * phi_q),
            -dphi_q + self.wb * (v_q + self.Rs * i_q - omega_m * phi_d),
            -dphi_F + self.wb * (self.v_F - self.R_F * i_F),
            -dphi_D - self.wb * self.R_D * i_D,
            -dphi_Q1 - self.wb * self.R_Q1 * i_Q1,
            -dphi_Q2 - self.wb * self.R_Q2 * i_Q2,
            -ddelta_r + self.wb * (omega_m-1),
            -domega_m + (self.wb / (2 * self.H)) * (Tm - (phi_q * i_d - phi_d * i_q) - self.Kd * self.wb * (omega_m-1)),
            # algebraic generator
            -phi_d + self.Ld * i_d + self.Lmd * i_F + self.Lmd * i_D,
            -phi_q + self.Lq * i_q + self.Lmq * i_Q1 + self.Lmq * i_Q2,
            -phi_F + self.Lmd * i_d + self.LF * i_F + self.Lmd * i_D,
            -phi_D + self.Lmd * i_d + self.Lmd * i_F + self.LD * i_D,
            -phi_Q1 + self.Lmq * i_q + self.LQ1 * i_Q1 + self.Lmq * i_Q2,
            -phi_Q2 + self.Lmq * i_q + self.Lmq * i_Q1 + self.LQ2 * i_Q2,
            v_d - V_re * np.sin(delta_r) + V_im * np.cos(delta_r),
            v_q - V_re * np.cos(delta_r) - V_im * np.sin(delta_r),
            I_re - (-1) * (i_d * np.sin(delta_r) + i_q * np.cos(delta_r)),
            I_im + (-1) * (i_d * np.cos(delta_r) - i_q * np.sin(delta_r)),
            self.G * (V_re - np.real(self.V_g)) - self.B * (V_im - np.imag(self.V_g)) - I_re,
            self.B * (V_re - np.real(self.V_g)) + self.G * (V_im - np.imag(self.V_g)) - I_im,
        )
        return f

    def u_exact(self, t):
        """
        Approximating the exact solution

        Todo:docu
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.7466010519363004  # phi_d
        me[1] = -0.6693249361197048  # phi_q
        me[2] = 1.125593681235877  # phi_F
        me[3] = 0.8853384216922447  # phi_D
        me[4] = -0.6122801972459051  # phi_Q1
        me[5] = -0.6122801972459051  # phi_Q2
        me[6] = -0.9249157983734856  # i_d
        me[7] = -0.3802982591590532  # i_q
        me[8] = 1.458284327617462  # i_F
        me[9] = 0.0  # i_D
        me[10] = 0.0  # i_Q1
        me[11] = 0.0  # i_Q2
        me[12] = 1.0  # omega_m
        me[13] = 0.7295713955883498  # delta_r (rad)
        me[14] = 0.6665501887246269 # v_d
        me[15] = 0.7454601571587608 # v_q
        me[16] = 1  # V_re
        me[17] = 0  # V_im
        me[18] = 0.9 # I_re
        me[19] = -0.436  # I_im

        return me


class SynchronousGenerator_5Ybus(ptype_dae):
    """
    Example implementing the synchronous generator model from PinTSimE project
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """
        Initialization routine
        """

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.Ld = 1.8099
        self.Lq = 1.76
        self.LF = 1.824652
        self.LD = 1.83149
        self.LQ1 = 2.578217
        self.LQ2 = 1.729817
        self.Lmd = 1.6599
        self.Lmq = 1.61
        self.Rs = 3 * 1e-3
        self.R_F = 0.000742259
        self.R_D = 0.0333636
        self.R_Q1 = 0.0104167
        self.R_Q2 = 2.5953e-2
        self.H = 6
        self.Kd = 0.0
        self.wb = 100 * np.pi
        #self.vbus = 0.5419999999999999 - 0.4064j
        self.v_F = 0.00108242466673301
        self.R_shunt = 1e9
        self.set_switch = False
        self.V_g = 0.6920 - 0.4064j
        self.G = np.real(1 / (0.1 + 0.5j))
        self.B = np.imag(1 / (0.1 + 0.5j))
        self.i = np.array([0.41689141-0.28391037j, 0.+0.j, 3.21805699-1.97426777j, 0.0502718-1.8412293j, 4.62569193+0.71563242j])
        self.net = pandapower.networks.case5()
        pandapower.run.runpp(self.net)
        self.Ybus = self.net._ppc["internal"]["Ybus"].todense()
        self.Tm = 0.0
        pload = self.net.load['p_mw'].values / self.net.sn_mva
        qload = self.net.load['q_mvar'].values / self.net.sn_mva

        vload = self.net.res_bus.vm_pu[self.net.res_load.index] * np.cos(self.net.res_bus.va_degree[self.net.res_load.index]*np.pi/180) + 1j * self.net.res_bus.vm_pu[self.net.res_load.index] * np.sin(self.net.res_bus.va_degree[self.net.res_load.index]*np.pi/180)
        yload = 0j * np.zeros(np.shape(self.Ybus)[0])

        yload[self.net.res_load.index] = (pload - 1j*qload)/(abs(vload)*abs(vload))

        for i in range(np.shape(self.Ybus)[0]):
            self.Ybus[i,i] = self.Ybus[i,i] + yload[i]

    def eval_f(self, u, du, t):
        """
        Routine to evaluate implicit representation of the problem
        """

        Ybus = self.Ybus
        phi_d, phi_q, phi_F, phi_D, phi_Q1, phi_Q2 = u[0], u[1], u[2], u[3], u[4], u[5]
        i_d, i_q, i_F, i_D, i_Q1, i_Q2 = u[6], u[7], u[8], u[9], u[10], u[11]
        omega_m, delta_r = u[12], u[13]
        v_d, v_q = u[14], u[15]
        I_re, I_im = u[16], u[17]
        v1_re, v1_im, v2_re, v2_im  = u[18], u[19], u[20], u[21]
        v3_re, v3_im, v4_re, v4_im, v5_re, v5_im = u[22], u[23], u[24], u[25], u[26], u[27]

        dphi_d, dphi_q, dphi_F, dphi_D, dphi_Q1, dphi_Q2 = du[0], du[1], du[2], du[3], du[4], du[5]
        domega_m, ddelta_r = du[12], du[13]

        v_re = [v1_re, v2_re, v3_re, v4_re, v5_re]
        v_im = [v1_im, v2_im, v3_im, v4_im, v5_im]
        #print(du)
        f = self.dtype_f(self.init)
        f[:] = (
            -dphi_d + self.wb * (v_d + self.Rs * i_d + omega_m * phi_q),
            -dphi_q + self.wb * (v_q + self.Rs * i_q - omega_m * phi_d),
            -dphi_F + self.wb * (self.v_F - self.R_F * i_F),
            -dphi_D - self.wb * self.R_D * i_D,
            -dphi_Q1 - self.wb * self.R_Q1 * i_Q1,
            -dphi_Q2 - self.wb * self.R_Q2 * i_Q2,
            -ddelta_r + self.wb * (omega_m-1),
            -domega_m + (self.wb / (2 * self.H)) * (self.Tm - (phi_q * i_d - phi_d * i_q) - self.Kd * self.wb * (omega_m-1)),
            # algebraic generator
            -phi_d + self.Ld * i_d + self.Lmd * i_F + self.Lmd * i_D,
            -phi_q + self.Lq * i_q + self.Lmq * i_Q1 + self.Lmq * i_Q2,
            -phi_F + self.Lmd * i_d + self.LF * i_F + self.Lmd * i_D,
            -phi_D + self.Lmd * i_d + self.Lmd * i_F + self.LD * i_D,
            -phi_Q1 + self.Lmq * i_q + self.LQ1 * i_Q1 + self.Lmq * i_Q2,
            -phi_Q2 + self.Lmq * i_q + self.Lmq * i_Q1 + self.LQ2 * i_Q2,
            v_d - v_re[0] * np.sin(delta_r) + v_im[0] * np.cos(delta_r),
            v_q - v_re[0] * np.cos(delta_r) - v_im[0] * np.sin(delta_r),
            I_re - (-1) * (i_d * np.sin(delta_r) + i_q * np.cos(delta_r)),
            I_im + (-1) * (i_d * np.cos(delta_r) - i_q * np.sin(delta_r)),
            # 5 Ybus connections
            np.real(Ybus[0,0])*v_re[0] - np.imag(Ybus[0,0])*v_im[0] + np.real(Ybus[0,1])*v_re[1] - np.imag(Ybus[0,1])*v_im[1] + np.real(Ybus[0,2])*v_re[2] - np.imag(Ybus[0,2])*v_im[2] + np.real(Ybus[0,3])*v_re[3] - np.imag(Ybus[0,3])*v_im[3] + np.real(Ybus[0,4])*v_re[4] - np.imag(Ybus[0,4])*v_im[4] - np.real(self.i[0]),
            np.real(Ybus[1,0])*v_re[0] - np.imag(Ybus[1,0])*v_im[0] + np.real(Ybus[1,1])*v_re[1] - np.imag(Ybus[1,1])*v_im[1] + np.real(Ybus[1,2])*v_re[2] - np.imag(Ybus[1,2])*v_im[2] + np.real(Ybus[1,3])*v_re[3] - np.imag(Ybus[1,3])*v_im[3] + np.real(Ybus[1,4])*v_re[4] - np.imag(Ybus[1,4])*v_im[4] - np.real(self.i[1]),
            np.real(Ybus[2,0])*v_re[0] - np.imag(Ybus[2,0])*v_im[0] + np.real(Ybus[2,1])*v_re[1] - np.imag(Ybus[2,1])*v_im[1] + np.real(Ybus[2,2])*v_re[2] - np.imag(Ybus[2,2])*v_im[2] + np.real(Ybus[2,3])*v_re[3] - np.imag(Ybus[2,3])*v_im[3] + np.real(Ybus[2,4])*v_re[4] - np.imag(Ybus[2,4])*v_im[4] - np.real(self.i[2]),
            np.real(Ybus[3,0])*v_re[0] - np.imag(Ybus[3,0])*v_im[0] + np.real(Ybus[3,1])*v_re[1] - np.imag(Ybus[3,1])*v_im[1] + np.real(Ybus[3,2])*v_re[2] - np.imag(Ybus[3,2])*v_im[2] + np.real(Ybus[3,3])*v_re[3] - np.imag(Ybus[3,3])*v_im[3] + np.real(Ybus[3,4])*v_re[4] - np.imag(Ybus[3,4])*v_im[4] - np.real(self.i[3]),
            np.real(Ybus[4,0])*v_re[0] - np.imag(Ybus[4,0])*v_im[0] + np.real(Ybus[4,1])*v_re[1] - np.imag(Ybus[4,1])*v_im[1] + np.real(Ybus[4,2])*v_re[2] - np.imag(Ybus[4,2])*v_im[2] + np.real(Ybus[4,3])*v_re[3] - np.imag(Ybus[4,3])*v_im[3] + np.real(Ybus[4,4])*v_re[4] - np.imag(Ybus[4,4])*v_im[4] - np.real(self.i[4]),
            (np.imag(Ybus[0,0])*v_re[0] + np.real(Ybus[0,0])*v_im[0]) + (np.imag(Ybus[0,1])*v_re[1] + np.real(Ybus[0,1])*v_im[1]) + (np.imag(Ybus[0,2])*v_re[2] + np.real(Ybus[0,2])*v_im[2]) + (np.imag(Ybus[0,3])*v_re[3] + np.real(Ybus[0,3])*v_im[3]) + (np.imag(Ybus[0,4])*v_re[4] + np.real(Ybus[0,4])*v_im[4]) - np.imag(self.i[0]),
            (np.imag(Ybus[1,0])*v_re[0] + np.real(Ybus[1,0])*v_im[0]) + (np.imag(Ybus[1,1])*v_re[1] + np.real(Ybus[1,1])*v_im[1]) + (np.imag(Ybus[1,2])*v_re[2] + np.real(Ybus[1,2])*v_im[2]) + (np.imag(Ybus[1,3])*v_re[3] + np.real(Ybus[1,3])*v_im[3]) + (np.imag(Ybus[1,4])*v_re[4] + np.real(Ybus[1,4])*v_im[4]) - np.imag(self.i[1]),
            (np.imag(Ybus[2,0])*v_re[0] + np.real(Ybus[2,0])*v_im[0]) + (np.imag(Ybus[2,1])*v_re[1] + np.real(Ybus[2,1])*v_im[1]) + (np.imag(Ybus[2,2])*v_re[2] + np.real(Ybus[2,2])*v_im[2]) + (np.imag(Ybus[2,3])*v_re[3] + np.real(Ybus[2,3])*v_im[3]) + (np.imag(Ybus[2,4])*v_re[4] + np.real(Ybus[2,4])*v_im[4]) - np.imag(self.i[2]),
            (np.imag(Ybus[3,0])*v_re[0] + np.real(Ybus[3,0])*v_im[0]) + (np.imag(Ybus[3,1])*v_re[1] + np.real(Ybus[3,1])*v_im[1]) + (np.imag(Ybus[3,2])*v_re[2] + np.real(Ybus[3,2])*v_im[2]) + (np.imag(Ybus[3,3])*v_re[3] + np.real(Ybus[3,3])*v_im[3]) + (np.imag(Ybus[3,4])*v_re[4] + np.real(Ybus[3,4])*v_im[4]) - np.imag(self.i[3]),
            (np.imag(Ybus[4,0])*v_re[0] + np.real(Ybus[4,0])*v_im[0]) + (np.imag(Ybus[4,1])*v_re[1] + np.real(Ybus[4,1])*v_im[1]) + (np.imag(Ybus[4,2])*v_re[2] + np.real(Ybus[4,2])*v_im[2]) + (np.imag(Ybus[4,3])*v_re[3] + np.real(Ybus[4,3])*v_im[3]) + (np.imag(Ybus[4,4])*v_re[4] + np.real(Ybus[4,4])*v_im[4]) - np.imag(self.i[4]),
        )
        return f

    def u_exact(self, t):
        """
        Approximating the exact solution

        Todo:docu
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        phi_0 = abs(np.angle(self.i[0]))
        v1_re = 1.0
        v1_im = 0.0
        Vt0_abs = abs(v1_re+1j*v1_im)
        iline0 = self.i[0]
        I_line_0_abs = abs(iline0)
        delta_r_0 = np.arctan((self.Lq*I_line_0_abs*np.cos(phi_0)-self.Rs*I_line_0_abs*np.sin(phi_0))/(Vt0_abs+self.Rs*I_line_0_abs*np.cos(phi_0)+self.Lq * I_line_0_abs * np.sin(phi_0)))
        e_d0 = Vt0_abs * np.sin(delta_r_0)
        e_q0 = Vt0_abs * np.cos(delta_r_0)
        i_d0 = I_line_0_abs * np.sin(delta_r_0+phi_0)
        i_q0 = I_line_0_abs * np.cos(delta_r_0+phi_0)
        i_fd0 = (e_q0 + self.Rs * i_q0 + self.Ld*i_d0)/ self.Lmd
        self.v_F = self.R_F * i_fd0

        psi_fd0 = self.LF *i_fd0 - self.Lmd * i_d0
        psi_kd0 = self.Lmd * (i_fd0 - i_d0)
        psi_kq1_0 = -self.Lmq * i_q0
        psi_kq2_0 = psi_kq1_0
        psi_ds0 = -self.Ld * i_d0 + self.Lmd * i_fd0
        psi_qs0 = -self.Lq * i_q0

        P0 = self.net.res_gen.p_mw.values [0] / self.net.sn_mva
        Q0 = self.net.res_gen.q_mvar.values [0] / self.net.sn_mva

        T_e_0 = P0 + I_line_0_abs * I_line_0_abs * self.Rs  # ~= phi_q * i_d - phi_d * i_q
        self.Tm = T_e_0

        e_0 = 0
        e_kd0 = 0
        e_kq1_0 = 0
        e_kq2_0 = 0
        u_0 = [e_d0, e_q0, e_0, self.v_F, e_kd0, e_kq1_0, e_kq2_0]

        me = self.dtype_u(self.init)
        me[0] = psi_ds0 #0.7466010519363004  # phi_d
        me[1] = psi_qs0 #-0.6693249361197048  # phi_q
        me[2] = psi_fd0 #1.125593681235877  # phi_F
        me[3] = psi_kd0 #0.8853384216922447  # phi_D
        me[4] = psi_kq1_0 #-0.6122801972459051  # phi_Q1
        me[5] = psi_kq2_0 #-0.6122801972459051  # phi_Q2
        me[6] = i_d0 #-0.9249157983734856  # i_d
        me[7] = i_q0 #-0.3802982591590532  # i_q
        me[8] = i_fd0 #1.458284327617462  # i_F
        me[9] = 0.0  # i_D
        me[10] = 0.0  # i_Q1
        me[11] = 0.0  # i_Q2
        me[12] = 1.0  # omega_m
        me[13] = delta_r_0 #0.7295713955883498  # delta_r (rad)
        me[14] = e_d0 #0.6665501887246269 # v_d
        me[15] = e_q0 #0.7454601571587608 # v_q
        me[16] = np.real(self.i[0]) #0.9 # I_re
        me[17] = np.imag(self.i[0]) #-0.436  # I_im
        me[18] =  v1_re
        me[19] =  v1_im
        me[20] =  0.98926124  # v2_re
        me[21] =  0.0  # v2_im
        me[22] =  1.0  # v3_re
        me[23] =  0.0  # v3_im
        me[24] =  1.0  # v4_re
        me[25] =  0.0  # v4_im
        me[26] =  1.0  # v5_re
        me[27] =  0.0  # v5_im

        return me


class IEEE9BusSystem(ptype_dae):
    """
    Example implementing the IEEE9BusSystem.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):

        m = 3
        n = 9
        nvars = 11*m + 2*m + 2*n
        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.net = nw.case9()
        self.m = m
        self.n = n
        self.baseMVA = 100
        self.ws = 2*np.pi*60
        self.ws_vector = self.ws*np.ones(self.m)
        self.w0 = self.ws_vector

        # Machine data (MD) as a 2D NumPy array
        self.MD = np.array([
            [23.640, 6.4000, 3.0100],  # 1 - H
            [0.1460, 0.8958, 1.3125],  # 2 - Xd
            [0.0608, 0.1198, 0.1813],  # 3 - Xdp
            [0.0489, 0.0881, 0.1133],  # 4 - Xdpp
            [0.0969, 0.8645, 1.2578],  # 5 - Xq
            [0.0969, 0.1969, 0.2500],  # 6 - Xqp
            [0.0396, 0.0887, 0.0833],  # 7 - Xqpp
            [8.9600, 6.0000, 5.8900],  # 8 - Tdop
            [0.1150, 0.0337, 0.0420],  # 9 - Td0pp
            [0.3100, 0.5350, 0.6000],  # 10 - Tqop
            [0.0330, 0.0780, 0.1875],  # 11 - Tq0pp
            [0.0041, 0.0026, 0.0035],  # 12 - RS
            [0.1200, 0.1020, 0.0750],  # 13 - Xls
            [0.1 * (2 * 23.64) / self.ws, 0.2 * (2 * 6.4) / self.ws, 0.3 * (2 * 3.01) / self.ws],  # 14 - Dm (ws should be defined)
        ])

        # Excitation data (ED) as a 2D NumPy array
        self.m = 3  # The number of machines (adjust this according to your specific case)
        self.ED = np.array([
            20.000 * np.ones(self.m),   # 1- KA
            0.2000 * np.ones(self.m),   # 2- TA
            1.0000 * np.ones(self.m),   # 3- KE
            0.3140 * np.ones(self.m),   # 4- TE
            0.0630 * np.ones(self.m),   # 5- KF
            0.3500 * np.ones(self.m),   # 6- TF
            0.0039 * np.ones(self.m),   # 7- Ax
            1.5550 * np.ones(self.m),   # 8- Bx
        ])

        # Turbine data (TD) as a 2D NumPy array
        self.TD = np.array([
            0.10 * np.ones(self.m),     # 1- TCH
            0.05 * np.ones(self.m),     # 2- TSV
            0.05 * np.ones(self.m),     # 3- RD
        ])

        self.mpc = IEEE9Bus()

        self.bus1 = self.mpc['bus']
        self.branch1 = self.mpc['branch']
        self.gen1 = self.mpc['gen']


        self.Ybus1 = self.mpc['Ybus']
        self.Ybus = self.Ybus1
        self.bus = self.bus1
        self.branch = self.branch1
        self.gen = self.gen1

        self.Yabs = abs(self.mpc['Ybus'])
        self.Yang = np.angle(self.mpc['Ybus']) #rad

        self.Yabs = abs(self.Ybus)
        self.Yang = np.angle(self.Ybus)
        self.IC1 = [row[7] for row in self.bus]  # Column 8 in MATLAB is indexed as 7 in Python (0-based index)
        self.IC2 = [row[8] for row in self.bus]  # Column 9 in MATLAB is indexed as 8 in Python
        self.n = len(self.bus)  # Number of rows in 'bus' list
        self.m = len(self.gen)  # Number of rows in 'gen' list

        gen0 = [0] * self.n
        for i in range(self.m):
            gen0[i] = self.gen[i][1]
        self.genP = gen0
        self.IC3 = [val / self.baseMVA for val in self.genP]  # Assuming 'baseMVA' is defined somewhere

        gen0 = [0] * self.n
        for i in range(self.m):
            gen0[i] = self.gen[i][2]
        genQ = gen0
        for i in range(self.n):
            genQ[i] += self.bus[i][5]  # Column 6 in MATLAB is indexed as 5 in Python
        self.IC4 = [val / self.baseMVA for val in genQ]

        self.IC5 = [row[2] for row in self.bus]  # Column 3 in MATLAB is indexed as 2 in Python
        IC5 = [val / self.baseMVA for val in self.IC5]

        self.IC6 = [row[3] for row in self.bus]  # Column 4 in MATLAB is indexed as 3 in Python
        self.IC6 = [val / self.baseMVA for val in self.IC6]

        self.IC = list(zip(self.IC1, self.IC2, self.IC3, self.IC4, self.IC5, self.IC6))

        self.PL = [row[4] for row in self.IC]  # Column 5 in MATLAB is indexed as 4 in Python
        self.QL = [row[5] for row in self.IC]  # Column 6 in MATLAB is indexed as 5 in Python

        self.PG = [row[2] for row in self.IC]  # Column 3 in MATLAB is indexed as 2 in Python
        self.QG = [row[3] for row in self.IC]  # Column 4 in MATLAB is indexed as 3 in Python

        TH0 = [row[1] * np.pi / 180 for row in self.IC]
        # TH0 = np.array(TH0).reshape(-1, 1)  # Reshape to column vector
        self.TH0 = np.array(TH0)

        V0 = [row[0] for row in self.IC]
        # V0 = np.array(V0).reshape(-1, 1)  # Reshape to column vector
        self.V0 = np.array(V0)

        self.VG0 = self.V0[:self.m]
        self.THG0 = self.TH0[:self.m]

        # Extracting values from the MD array
        self.H = self.MD[0, :]
        self.Xd = self.MD[1, :]
        self.Xdp = self.MD[2, :]
        self.Xdpp = self.MD[3, :]
        self.Xq = self.MD[4, :]
        self.Xqp = self.MD[5, :]
        self.Xqpp = self.MD[6, :]
        self.Td0p = self.MD[7, :]
        self.Td0pp = self.MD[8, :]
        self.Tq0p = self.MD[9, :]
        self.Tq0pp = self.MD[10, :]
        self.Rs = self.MD[11, :]
        self.Xls = self.MD[12, :]
        self.Dm = self.MD[13, :]

        # Extracting values from the ED array
        self.KA = self.ED[0, :]
        self.TA = self.ED[1, :]
        self.KE = self.ED[2, :]
        self.TE = self.ED[3, :]
        self.KF = self.ED[4, :]
        self.TF = self.ED[5, :]
        self.Ax = self.ED[6, :]
        self.Bx = self.ED[7, :]

        # Extracting values from the TD array
        self.TCH = self.TD[0, :]
        self.TSV = self.TD[1, :]
        self.RD = self.TD[2, :]

        # Calculate MH
        self.MH = 2 * self.H / self.ws

        self.PG = np.array(self.PG)  # Convert PG to a NumPy array
        self.QG = np.array(self.QG)  # Convert QG to a NumPy array
        # Represent QG as complex numbers
        self.QG = self.QG.astype(complex)

        # Convert VG0 and THG0 to complex phasors
        self.Vphasor = self.VG0 * np.exp(1j * self.THG0)
        # Vphasor = Vphasor.T

        # Calculate Iphasor
        # Iphasor = np.conj((PG[:m] + 1j * QG[:m]) / Vphasor)
        self.Iphasor = np.conj(np.divide(self.PG[:m] + 1j * self.QG[:m], self.Vphasor))

        # Calculate E0
        self.E0 = self.Vphasor + (self.Rs + 1j * self.Xq) * self.Iphasor

        # Calculate Em, D0, Id0, and Iq0
        self.Em = np.abs(self.E0)
        self.D0 = np.angle(self.E0)
        self.Id0 = np.real(self.Iphasor * np.exp(-1j * (self.D0 - np.pi / 2)))
        self.Iq0 = np.imag(self.Iphasor * np.exp(-1j * (self.D0 - np.pi / 2)))

        # Calculate Edp0, Si2q0, Eqp0, and Si1d0
        self.Edp0 = (self.Xq - self.Xqp) * self.Iq0
        self.Si2q0 = (self.Xls - self.Xq) * self.Iq0
        self.Eqp0 = self.Rs * self.Iq0 + self.Xdp * self.Id0 + self.V0[:m] * np.cos(self.D0 - self.TH0[:m])
        self.Si1d0 = self.Eqp0 - (self.Xdp - self.Xls) * self.Id0

        # Calculate Efd0 and TM0
        self.Efd0 = self.Eqp0 + (self.Xd - self.Xdp) * self.Id0
        self.TM0 = ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * self.Eqp0 * self.Iq0 + ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * self.Si1d0 * self.Iq0 + \
            ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * self.Edp0 * self.Id0 - ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * self.Si2q0 * self.Id0 + \
            (self.Xqpp - self.Xdpp) * self.Id0 * self.Iq0

        # Calculate VR0 and RF0
        self.VR0 = (self.KE + self.Ax * np.exp(self.Bx * self.Efd0)) * self.Efd0
        self.RF0 = (self.KF / self.TF) * self.Efd0

        # Calculate Vref and PSV0
        self.Vref = self.V0[:self.m] + self.VR0 / self.KA
        self.PSV0 = self.TM0
        self.PC = self.PSV0

        # Create a list to hold the initial values of state variables
        x0 = [0] * (11 * m)

        # Assign initial values to state variables as in the MATLAB code
        x0[0:m] = self.Eqp0
        x0[m:2 * m] = self.Si1d0
        x0[2 * m:3 * m] = self.Edp0
        x0[3 * m:4 * m] = self.Si2q0
        x0[4 * m:5 * m] = self.D0
        x0[5 * m:6 * m] = self.ws_vector
        x0[6 * m:7 * m] = self.Efd0
        x0[7 * m:8 * m] = self.RF0
        x0[8 * m:9 * m] = self.VR0
        x0[9 * m:10 * m] = self.TM0
        x0[10 * m:11 * m] = self.PSV0

        # Initial values of algebraic variables
        a0 = [self.Id0, self.Iq0, self.V0, self.TH0]

        # Optionally, you can create a copy of x0 to store the initial values separately.
        x01 = x0.copy()

        x0_values = np.array([1.0591, 0.79193, 0.77098, 1.077, 0.76899, 0.71139, 0, 0.62385, 0.62505, 0.015514,
             -0.71253, -0.73358, 0.061423, 1.0645, 0.94343, 376.99, 376.99, 376.99, 1.0849,
             1.7917, 1.4051, 0.19528, 0.3225, 0.25292, 1.1077, 1.905, 1.4538, 0.71863, 1.6366,
             0.85245, 0.71863, 1.6366, 0.85245])

        self.alpha=2
        self.beta=2

        self.bb1, self.aa1 = np.meshgrid(np.arange(0, self.m), np.arange(0, self.n))
        self.bb1 = self.bb1.astype(int)
        self.aa1 = self.aa1.astype(int)
        # Create matrix grid to eliminate for-loops (load buses)
        self.bb2, self.aa2 = np.meshgrid(np.arange(self.m, self.n), np.arange(0, self.n))
        self.bb2 = self.bb2.astype(int)
        self.aa2 = self.aa2.astype(int)

    def eval_f(self, u, du, t):
        dEqp, dSi1d, dEdp, dSi2q, dDelta = du[0:self.m], du[self.m:2*self.m], du[2*self.m:3*self.m], du[3*self.m:4*self.m], du[4*self.m:5*self.m]
        dw, dEfd, dRF, dVR, dTM, dPSV = du[5*self.m:6*self.m], du[6*self.m:7*self.m], du[7*self.m:8*self.m], du[8*self.m:9*self.m], du[9*self.m:10*self.m], du[10*self.m:11*self.m]

        Eqp, Si1d, Edp, Si2q, Delta = u[0:self.m], u[self.m:2*self.m], u[2*self.m:3*self.m], u[3*self.m:4*self.m], u[4*self.m:5*self.m]
        w, Efd, RF, VR, TM, PSV = u[5*self.m:6*self.m], u[6*self.m:7*self.m], u[7*self.m:8*self.m], u[8*self.m:9*self.m], u[9*self.m:10*self.m], u[10*self.m:11*self.m]

        Id, Iq, V, TH = u[11*self.m:11*self.m + self.m], u[11*self.m + self.m:11*self.m + 2*self.m], u[11*self.m + 2*self.m:11*self.m + 2*self.m + self.n], u[11*self.m + 2*self.m + self.n:11*self.m + 2*self.m + 2 *self.n]

        COI = np.sum(w * self.MH) / np.sum(self.MH)

        # Voltage-dependent active loads
        PL2 = self.PL * ((V / self.V0) ** self.alpha)
        # Voltage-dependent reactive loads
        QL2 = self.QL * ((V / self.V0) ** self.beta)
        # V = V.reshape(-1, 1)
        V = V.T
        # Vectorized calculations
        Vectorized_angle1 = (np.array([TH.take(indices) for indices in self.bb1.T]) - np.array([TH.take(indices) for indices in self.aa1.T]) - self.Yang[:self.m, :self.n])
        Vectorized_mag1 = (V[:self.m] * V[:self.n].reshape(-1, 1)).T * self.Yabs[:self.m, :self.n]
        # Vectorized_mag1 = np.zeros(self.m)
        # for i in range(self.m):
        #    for k in range(self.n):
        #        Vectorized_mag1[i] += V[k] * self.Yabs[i, k]
        #    Vectorized_mag1[i] *= V[i]

        sum1 = np.sum(Vectorized_mag1 * np.cos(Vectorized_angle1), axis=1)
        sum2 = np.sum(Vectorized_mag1 * np.sin(Vectorized_angle1), axis=1)
        VG = V[:self.m]
        THG = TH[:self.m]
        Angle_diff = Delta - THG
        Vectorized_angle2 = (np.array([TH.take(indices) for indices in self.bb2.T]) - np.array([TH.take(indices) for indices in self.aa2.T]) - self.Yang[self.m:self.n, :self.n])
        Vectorized_mag2 = (V[self.m:self.n] * V[:self.n].reshape(-1, 1)).T * self.Yabs[self.m:self.n, : self.n]
        # Vectorized_mag2 = np.zeros(self.n - self.m)
        # for i in range(self.m, self.n):
        #    for k in range(self.n):
        #        Vectorized_mag2[i] += V[k] * self.Yabs[i, k]
        #    Vectorized_mag2[i] *= V[i]
        sum3 = np.sum(Vectorized_mag2 * np.cos(Vectorized_angle2), axis=1)
        sum4 = np.sum(Vectorized_mag2 * np.sin(Vectorized_angle2), axis=1)
        f = self.dtype_f(self.init)

        eqs = []
        eqs.append((1.0 / self.Td0p) * (-Eqp - (self.Xd - self.Xdp) * (Id - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls) ** 2) * (Si1d + (self.Xdp - self.Xls) * Id - Eqp)) + Efd) - dEqp)
        eqs.append((1.0 / self.Td0pp) * (-Si1d + Eqp - (self.Xdp - self.Xls) * Id) - dSi1d)
        eqs.append( (1.0 / self.Tq0p) * (-Edp + (self.Xq - self.Xqp) * (Iq - ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls) ** 2) * (Si2q + (self.Xqp - self.Xls) * Iq + Edp))) - dEdp)
        eqs.append( (1.0 / self.Tq0pp) * (-Si2q - Edp - (self.Xqp - self.Xls) * Iq) - dSi2q)
        eqs.append( w - COI - dDelta)
        eqs.append( (self.ws / (2.0 * self.H)) * (TM - ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * Eqp * Iq - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * Si1d * Iq - ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * Edp * Id + ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * Si2q * Id - (self.Xqpp - self.Xdpp) * Id * Iq - self.Dm * (w - self.ws)) - dw)
        eqs.append( (1.0 / self.TE) * ((-(self.KE + self.Ax * np.exp(self.Bx * Efd))) * Efd + VR) - dEfd)
        eqs.append( (1.0 / self.TF) * (-RF + (self.KF / self.TF) * Efd) - dRF)
        eqs.append( (1.0 / self.TA) * (-VR + self.KA * RF - ((self.KA * self.KF) / self.TF) * Efd + self.KA * (self.VR0 - V[:self.m])) - dVR)
        eqs.append( (1.0 / self.TCH) * (-TM + PSV) - dTM)
        eqs.append( (1.0 / self.TSV) * (-PSV + self.PSV0 - (1.0 / self.RD) * (w / self.ws - 1)) - dPSV)
        eqs.append( self.Rs * Id - self.Xqpp * Iq - ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * Edp + ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * Si2q + VG * np.sin(Angle_diff))
        eqs.append( self.Rs * Iq + self.Xdpp * Id - ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * Eqp - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * Si1d + VG * np.cos(Angle_diff))
        eqs.append( (Id * VG.T * np.sin(Angle_diff) + PL2[0:self.m] - sum1))  # (14)
        eqs.append( Id * VG.T * np.cos(Angle_diff) - QL2[0:self.m] - sum2)  # (15)
        eqs.append( -PL2[self.m:self.n] - sum3)  # (16)
        eqs.append( -QL2[self.m:self.n] - sum4)  # (17)
        eqs_flatten = [item for sublist in eqs for item in sublist]
            # Generator stator and power flow eq's
            # Non-generator power flow equations

        f[:] = eqs_flatten
        return f

    def u_exact(self, t):
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)
        me[0:self.m] = self.Eqp0
        me[self.m:2 * self.m] = self.Si1d0
        me[2 * self.m:3 * self.m] = self.Edp0
        me[3 * self.m:4 * self.m] = self.Si2q0
        me[4 * self.m:5 * self.m] = self.D0
        me[5 * self.m:6 * self.m] = self.ws_vector
        me[6 * self.m:7 * self.m] = self.Efd0
        me[7 * self.m:8 * self.m] = self.RF0
        me[8 * self.m:9 * self.m] = self.VR0
        me[9 * self.m:10 * self.m] = self.TM0
        me[10 * self.m:11 * self.m] = self.PSV0
        me[11*self.m:11*self.m + self.m] = self.Id0
        me[11*self.m + self.m:11*self.m + 2*self.m] = self.Iq0
        me[11*self.m + 2*self.m:11*self.m + 2*self.m + self.n] = self.V0
        me[11*self.m + 2*self.m + self.n:11*self.m + 2*self.m + 2 *self.n] = self.TH0
        return me


class SynchronousGenerator_Piline(ptype_dae):
    """
    Example implementing the synchronous generator model on a pi-line (?) from PinTSimE  project
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """
        Initialization routine
        """

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.L_d = 1.8099
        self.L_q = 1.76
        self.L_F = 1.8247
        self.L_D = 1.8312
        self.L_Q1 = 2.3352
        self.L_Q2 = 1.735
        self.L_md = 1.6599
        self.L_mq = 1.61
        self.R_s = 0.003
        self.R_F = 0.0006
        self.R_D = 0.0284
        self.R_Q1 = 0.0062
        self.R_Q2 = 0.0237
        self.omega_b = 376.9911184307752
        self.H_ = 3.525
        self.K_D = 0.0
        # pi line
        self.C_pi = 0.000002
        self.R_pi = 0.02
        self.L_pi = 0.00003
        # load
        self.R_L = 0.75
        self.v_F = 0.000939 #8.736809687330562e-4
        self.v_D = 0
        self.v_Q1 = 0
        self.v_Q2 = 0

    def eval_f(self, u, du, t):
        """
        Routine to evaluate the implicit representation of the problem i.e. F(u', u, t)
        Args:
            u (dtype_u): the current values. This parameter has been "hijacked" to contain [u', u] in this case to enable evaluation of the implicit representation
            t (float): current time
        Returns:
            Current value of F(), 21 components
        """

        # simulate torque change at t = 0.05
        Tm = 0.854 #input_torque(t)

        f = self.dtype_f(self.init)

        # extract variables for readability
        # algebraic components
        psi_d, psi_q, psi_F, psi_D, psi_Q1, psi_Q2 = u[0], u[1], u[2], u[3], u[4], u[5]
        i_d, i_q, i_F, i_D, i_Q1, i_Q2 = u[6], u[7], u[8], u[9], u[10], u[11]
        omega_m = u[12]
        v_d, v_q = u[13], u[14]
        iz_d, iz_q, il_d, il_q, vl_d, vl_q = u[15], u[16], u[17], u[18], u[19], u[20]

        # differential components
        # these result directly from the voltage equations, introduced e.g. pg. 145 Krause
        dpsi_d, dpsi_q, dpsi_F, dpsi_D, dpsi_Q1, dpsi_Q2 = du[0], du[1], du[2], du[3], du[4], du[5]
        # ddelta_r = du[12]
        domega_m = du[12]
        dv_d, dv_q = du[13], du[14]
        diz_d, diz_q, dvl_d, dvl_q = du[15], du[16],du[19], du[20]

        # electrical torque
        Te = psi_q * i_d - psi_d * i_q

        # algebraic variables are i_d, i_q, i_F, i_D, i_Q1, i_Q2, il_d, il_q

        f[:] = (
            # differential generator
            dpsi_d - self.omega_b * (v_d + self.R_s * i_d + omega_m * psi_q),
            dpsi_q - self.omega_b * (v_q + self.R_s * i_q - omega_m * psi_d),
            dpsi_F - self.omega_b * (self.v_F - self.R_F * i_F),
            dpsi_D - self.omega_b * (self.v_D - self.R_D * i_D),
            dpsi_Q1 - self.omega_b * (self.v_Q1 - self.R_Q1 * i_Q1),
            dpsi_Q2 - self.omega_b * (self.v_Q2 - self.R_Q2 * i_Q2),
            -domega_m + (self.omega_b / (2 * self.H_)) * (Tm - Te - self.K_D * self.omega_b * (omega_m-1)),
            # differential pi line
            dv_d - omega_m * v_q - 2/self.C_pi * (i_d - iz_d),
            dv_q + omega_m * v_q - 2/self.C_pi * (i_q - iz_q),
            dvl_d - omega_m * vl_q - 2/self.C_pi * (iz_d - il_d),
            dvl_q + omega_m * vl_q - 2/self.C_pi * (iz_q - il_q),
            diz_d + self.R_pi/self.L_pi * iz_d - omega_m * iz_q - (v_d - vl_d) / self.L_pi,
            diz_q + self.R_pi/self.L_pi * iz_q + omega_m * iz_d - (v_q - vl_q) / self.L_pi,
            # algebraic generator
            -psi_d + self.L_d * i_d + self.L_md * i_F + self.L_md * i_D,
            -psi_q + self.L_q * i_q + self.L_mq * i_Q1 + self.L_mq * i_Q2,
            -psi_F + self.L_md * i_d + self.L_F * i_F + self.L_D * i_D,
            -psi_D + self.L_md * i_d + self.L_md * i_F + self.L_D * i_D,
            -psi_Q1 + self.L_mq * i_q + self.L_Q1 * i_Q1 + self.L_mq * i_Q2,
            -psi_Q2 + self.L_mq * i_q + self.L_mq * i_Q1 + self.L_Q2 * i_Q2,
            # algebraic pi line
            -il_d + vl_d/self.R_L,
            -il_q + vl_q/self.R_L,
        )
        return f

    def u_exact(self, t):
        """
        Approximating the exact solution

        Todo:docu
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.3971299  # psi_d
        me[1] = 0.9219154 # psi_q
        me[2] = 0.8374232 # psi_F
        me[3] = 0.5795112  # psi_D
        me[4] = 0.8433430  # psi_Q1
        me[5] = 0.8433430  # psi_Q2
        me[6] = -1.215876  # i_d
        me[7] = 0.5238156  # i_q
        me[8] = 1.565 # i_F
        me[9] = 0.0 # i_D
        me[10] = 0.0 # i_Q1
        me[11] = 0.0 # i_Q2
        me[12] = 1.011581 # omega_m
        me[13] = -0.9362397 # v_d
        me[14] = 0.4033005 # v_q
        me[15] = -1.215875 # iz_d
        me[16] = 0.5238151  # iz_q
        me[17] = -1.215875  # il_d
        me[18] = 0.5238147  # il_q
        me[19] = -0.9119063 # vl_d
        me[20] = 0.3928611 # vl_q

        return me