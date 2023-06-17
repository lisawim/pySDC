import numpy as np

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class IdealGasLiquid(ptype_dae):
    r"""
    Class that implements the problem of a ideal gas-liquid model [1]_ in description of differential-algebraic
    equations (DAEs). The DAE system is given by

    if :math:`\dfrac{M_L}{\rho_L} > V_d`: (liquid model)

    .. math::
        \dfrac{d M_G}{dt} = F_G,

    .. math::
        \dfrac{d M_L}{dt} = F_L - L,

    .. math::
        0 = V - M_G \dfrac{RT}{P} - \dfrac{M_L}{\rho_L},

    .. math::
        0 = L - k_L x (P - P_{out})

    if :math:`\dfrac{M_L}{\rho_L} < V_d`: (gas model)

    .. math::
        \dfrac{d M_G}{dt} = F_G - G,

    .. math::
        \dfrac{d M_L}{dt} = F_L,

    .. math::
        0 = V - M_G \dfrac{RT}{P} - \dfrac{M_L}{\rho_L},

    .. math::
        0 = G - k_G x (P - P_{out})

    The model can be defined on three different regions:

    .. math::
        S_+ = \{M_L : \dfrac{M_L}{\rho_L} > V_d\}

    .. math::
        S_- = \{M_L : \dfrac{M_L}{\rho_L} < V_d\}

    .. math::
        S_0 = \{M_L : \dfrac{M_L}{\rho_L} = V_d\}

    :math:`S_0` defines the switching surface. In this case, the model is known to have a sliding solution.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.
    rho_L : float
        Density of liquid in moles per unit volume.
    V : float
        Total volume of reactor (in liter).
    V_d : float
        Volume below the dip tube in the reactor (in liter).
    T : float
        Temperature (in Kelvin).
    P_out : float
        Outlet pressure (in atm).
    x : float
        Valve opening.
    k_L : float
        Valve coefficient for gas flow.
    k_G : float
        Valve coefficient for liquid flow.
    F_G : float
        Gas feed rate (in mol/s).
    F_L : float
        Liquid feed rate (in mol/s).
    R : float
        Gas constant (in atm/(mol K)).

    Attributes
    ----------
    t_switch: float
        Time point of the discrete event found by switch estimation.
    nswitches: int
        Number of switches found by switch estimation.

    Note
    ----
    The initial conditions used in `u_exact` method are computed for the default parameters. Changing them would lead to
    probably no longer consistent initial conditions.

    References
    ----------
    .. [1] K. Moudgalya, V. Ryali. A class of discontinuous dynamical system I. An ideal gas-liquid system.
        Chemical Engineering Science (2001).
    """
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
        self,
        nvars=4,
        newton_tol=1e-7,
        rho_L=50,
        V=10,
        V_d=5.0,
        T=300,
        P_out=1,
        x=0.1,
        k_L=1.0,
        k_G=1.0,
        F_G=2.0,
        F_L=1.0,
        R=0.0820574587,
    ):
        """
        Initialization routine
        """

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister(
            'nvars',
            'newton_tol',
            'rho_L',
            'V',
            'V_d',
            'T',
            'P_out',
            'x',
            'k_L',
            'k_G',
            'F_G',
            'F_L',
            'R',
            localVars=locals(),
            readOnly=True,
        )

        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the right-hand side of the problem. Defining an state function :math:`h(M_L) = \dfrac{M_L}{\rho_L} - V_d`,
        for each time :math:`t` is is proven whether :math:`h(M_L) > 0` (resulting in the liquid model) or :math:`h(M_L) < 0`
        (resulting in the gas model).
        """
        f = self.dtype_f(self.init)

        ML, MG, P, G = u[0], u[1], u[2], u[3]

        dML, dMG = du[0], du[1]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        # state function
        h = ML / self.rho_L - self.V_d
        if h > 0 or (h > 0 and t >= t_switch):
            f[:] = (
                dMG - self.F_G,
                dML - self.F_L + G,
                self.V - ((MG * self.R * self.T) / P + ML / self.rho_L),
                G - self.k_L * self.x * (P - self.P_out),
            )
            print(t, 'Liquid')

        elif h < 0 or (h < 0 and t >= t_switch):  # gas
            f[:] = (
                dMG - self.F_G + G,
                dML - self.F_L,
                self.V - ((MG * self.R * self.T) / P + ML / self.rho_L),
                G - self.k_G * self.x * (P - self.P_out),
            )
            print(t, 'Gas')

        return f

    def u_exact(self, t):
        """
        Computes the exact solution.
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 249.0  # ML
        me[1] = 0.203922149167  # MG
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

        ML = [u[m][0] for m in range(len(u))]
        for m in range(len(ML)):
            if ML[m - 1] / self.rho_L - self.V_d > 0 and ML[m] / self.rho_L - self.V_d < 0:
                print('Condition 1')
                switch_detected = True
                m_guess = m - 1
                break

            elif ML[m - 1] / self.rho_L - self.V_d < 0 and ML[m] / self.rho_L - self.V_d > 0:
                print('Condition 2')
                switch_detected = True
                m_guess = m - 1
                break
        ML_tmp = [u[m][0] for m in range(len(u))]
        print([ML_tmp[m] / self.rho_L - self.V_d for m in range(len(ML))])
        vC_switch = [ML[m] / self.rho_L - self.V_d for m in range(len(ML))] if switch_detected else []

        return switch_detected, m_guess, vC_switch

    def count_switches(self):
        """
        Function counts number of switches found from the switch_estimator. This function is called there.
        """

        self.nswitches += 1
