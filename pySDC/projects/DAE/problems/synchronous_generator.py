import numpy as np
from scipy.interpolate import interp1d
import scipy as sp
import pandapower

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
        self.i = np.array([0.4   -0.3072516j , 3.2349-1.9465472j , 4.6651+0.38209623j, 0.0, 0.0])
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



