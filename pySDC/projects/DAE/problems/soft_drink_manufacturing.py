import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class SoftDrinkManufacturing(ptype_dae):
    """
    Example implementing the Soft-dring production system.
    See: "Sliding motion of discontinuous dynamical systems described by semi-implicit index one
    differential algebraic equations" written by J. Agrawal, K. K. Moudgalya, A. K. Pani (2006)
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, nvars, newton_tol):
        """
        Initialization routine
        """

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.rho_L = 50
        self.rho_a = 16
        self.V = 10
        self.V_d = 2.25
        self.T = 293
        self.P_out = 1
        self.x = 1.0
        self.k_L = 2.5
        self.k_G = 3.0
        self.F_1 = 0.5
        self.F_2 = 7.5
        self.kappa_c = 0.433 / 4000
        self.sigma = 1640
        self.R = 0.0820574587

    def eval_f(self, u, du, t):

        f = self.dtype_f(self.init)

        M1, M2, M3, G = u[0], u[1], u[2], u[3]

        dM1, dM2, dM3 = du[0], du[1], du[2]


        P = (M1 * self.R * self.T) / (self.V - M2 / self.rho_L - M3 / self.rho_a)
        Ml = ((M2 + M3) / (self.sigma - P)) * P

        if (M2 / self.rho_L) + (M3 / self.rho_a) - self.V_d <= 0:  # gas model
            f.expl[:] = (
                self.F_1 - G - self.kappa_c * (M1 * M2) / self.V,
                self.F_2 - self.kappa_c * (M1 * M2) / self.V,
                self.kappa_c * (M1 * M2) / self.V,
            )
            f.impl[:] = G - self.k_G * self.x * (P - self.P_out)

        else:  # liquid model
            f.expl[:] = (
                self.F_1 - G - self.kappa_c * (M1 * M2) / self.V,
                self.F_2 - M2 / (Ml + M2 + M3) * self.k_L * self.x * (P - self.P_out) - self.kappa_c * (M1 * M2) / self.V,
                - M3 / (Ml + M2 + M3) * self.k_L * self.x * (P - self.P_out) + self.kappa_c * (M1 * M2) / self.V
            )
            f.impl[:] = G - self.k_L * self.x * (P - self.P_out) - (M2 + M3) / (Ml + M2 + M3) * self.k_L * self.x * (P - self.P_out)

        return f

    def u_exact(self, t):

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.72  # M1
        me[1] = 95.0  # M2
        me[2] = 0.0  # M3
        me[3] = 3.4  # G

        return me


class SoftDrinkManufacturing_fully_implicit(ptype_dae):
    """
    Example implementing the Soft-dring production system.
    See: "Sliding motion of discontinuous dynamical systems described by semi-implicit index one
    differential algebraic equations" written by J. Agrawal, K. K. Moudgalya, A. K. Pani (2006)
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

        self.rho_L = 50
        self.rho_a = 16
        self.V = 10
        self.V_d = 2.25
        self.T = 293
        self.P_out = 1
        self.x = 1.0
        self.k_L = 2.5
        self.k_G = 3.0
        self.F_1 = 0.5
        self.F_2 = 7.5
        self.kappa_c = 0.4333 / 4000
        self.sigma = 1640
        self.R = 0.0820574587

    def eval_f(self, u, du, t):

        f = self.dtype_f(self.init)

        M1, M2, M3, G = u[0], u[1], u[2], u[3]

        dM1, dM2, dM3 = du[0], du[1], du[2]


        P = (M1 * self.R * self.T) / (self.V - (M2 / self.rho_L) - (M3 / self.rho_a))
        Ml = ((M2 + M3) / (self.sigma - P)) * P

        if (M2 / self.rho_L) + (M3 / self.rho_a) - self.V_d < 0:  # gas model
            f[:] = (
                dM1 - self.F_1 + G + self.kappa_c * ((M1 * M2) / self.V),
                dM2 - self.F_2 + self.kappa_c * ((M1 * M2) / self.V),
                dM3 - self.kappa_c * ((M1 * M2) / self.V),
                G - self.k_G * self.x * (P - self.P_out)
                #M1 - (Ml + Mg),
                #P - (self.sigma * Ml)/(Ml + M2 + M3),
                #self.V - ((M1 * self.R * self.T)/P - M2 / self.rho_L - M3 / self. rho_a),
                #L1 - self.k_G * self.x * (P - self.P_out),
                #0,
                #0,
            )
            print(t, 'Gas model')

        elif (M2 / self.rho_L) + (M3 / self.rho_a) - self.V_d > 0:  # liquid model
            f[:] = (
                dM1 - self.F_1 + G + self.kappa_c * ((M1 * M2) / self.V),
                dM2 - self.F_2 + (M2 / (Ml + M2 + M3)) * self.k_L * self.x * (P - self.P_out) + self.kappa_c * ((M1 * M2) / self.V),
                dM3 + (M3 / (Ml + M2 + M3)) * self.k_L * self.x * (P - self.P_out) - self.kappa_c * ((M1 * M2) / self.V),
                G - self.k_L * self.x * (P - self.P_out) + ((M2 + M3) / (Ml + M2 + M3)) * self.k_L * self.x * (P - self.P_out)
                #M1 - (Ml + Mg),
                #P - (self.sigma * Ml)/(Ml + M2 + M3),
                #self.V - ((M1 * self.R * self.T)/P - M2 / self.rho_L - M3 / self. rho_a),
                #M2 / (Ml + M2 + M3) - L2 / (L1 + L2 + L3),
                #M3 / (Ml + M2 + M3) - L3 / (L1 + L2 + L3),
                #L1 + L2 + L3 - self.k_L * self.x * (P - self.P_out),
            )
            print(t, 'Liquid model')

        return f

    def u_exact(self, t):

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.72  # M1
        me[1] = 95.0  # M2
        me[2] = 0.0  # M3
        me[3] = 0.084  # Ml
        me[4] = 0.636  # Mg
        me[5] = 1.454  # L1
        me[6] = 0.001  # L2
        me[7] = 1.1357  # L3
        me[8] = 0.0  # P

        return me


class IdealGasLiquid(ptype_dae):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol):
        """
        Initialization routine
        """

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals(), readOnly=True)

        self.rho_L = 50
        self.V = 10
        self.V_d = 5
        self.T = 300
        self.P_out = 1
        self.x = 0.1
        self.k_L = 1.0
        self.k_G = 1.0
        self.F_G = 2.0
        self.F_L = 1.0
        self.R = 0.0820574587
        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, du, t):

        f = self.dtype_f(self.init)

        ML, MG, P, G = u[0], u[1], u[2], u[3]

        dML, dMG = du[0], du[1]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if ML / self.rho_L > self.V_d or (ML / self.rho_L > self.V_d and t >= t_switch):  # liquid
            f[:] = (
                dMG - self.F_G,
                dML - self.F_L + G,
                self.V - ((MG * self.R * self.T) / P + ML / self.rho_L),
                G - self.k_L * self.x * (P - self.P_out)
            )
            #print(t, 'Liquid')

        elif ML / self.rho_L < self.V_d or (ML / self.rho_L < self.V_d and t >= t_switch):  # gas
            f[:] = (
                dMG - self.F_G + G,
                dML - self.F_L,
                self.V - ((MG * self.R * self.T) / P + ML / self.rho_L),
                G - self.k_G * self.x * (P - self.P_out)
            )
            #print(t, 'Gas')

        return f

    def u_exact(self, t):

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 251.0  # ML
        me[1] = 0.20229727148  # MG
        me[2] = 1.0  # P
        me[3] = 0.0  # G

        return me

    def get_switching_info(self, u, t):
        """
        Provides information about a discrete event for one subinterval.

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            switch_detected (bool): Indicates if a switch is found or not
            m_guess (np.int): Index of collocation node inside one subinterval of where the discrete event was found
            vC_switch (list): Contains function values of switching condition (for interpolation)
        """

        switch_detected = False
        m_guess = -100

        ML = [u[m][0] for m in range(1, len(u))]
        for m in range(1, len(ML)):
            if ML[m - 1] / self.rho_L - self.V_d > 0 and ML[m] / self.rho_L - self.V_d <= 0:
                print('Condition 1')
                switch_detected = True
                m_guess = m - 1
                break

            elif ML[m - 1] / self.rho_L - self.V_d <= 0 and ML[m] / self.rho_L - self.V_d > 0:
                print('Condition 2')
                switch_detected = True
                m_guess = m - 1
                break

        vC_switch = [ML[m] / self.rho_L - self.V_d for m in range(len(ML))] if switch_detected else []

        return switch_detected, m_guess, vC_switch

    def count_switches(self):
        """
        Function counts number of switches found from the switch_estimator. This function is called there.
        """

        self.nswitches += 1
