import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


def input_torque(t):
        if round(t, 14) < 0.2:
            Tm = 0.854
        else:
            Tm = 0.854 - 0.5
        return Tm


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
        print(t, du)
        # compute stator voltages
        #re_I = i_d * np.sin(delta_r) + i_q * np.cos(delta_r)
        #im_I = -i_d * np.cos(delta_r) + i_q * np.sin(delta_r)

        #V = 1.0
        #print(t, omega_m * self.wb / (2*np.pi))
        #if (omega_m * self.wb / (2*np.pi) < 49 or omega_m * self.wb / (2*np.pi) > 51) and self.set_switch == False:
        #V = self.R_shunt * (-1) * (re_I + 1j * im_I) # where R_shunt = 1e9
            #self.set_switch = True
            #print(t)
        #else:
        #V = self.vbus - self.Zline * (-1) * (re_I + 1j * im_I)

        #v_d = V.real * np.sin(delta_r) - V.imag * np.cos(delta_r)
        #v_q = V.real * np.cos(delta_r) + V.imag * np.sin(delta_r)
        #if t == 0.0015505102572168283:
        #    print(v_d, v_q)

        # electromagnetic torque
        Te = phi_q * i_d - phi_d * i_q

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



