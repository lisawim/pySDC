import numpy as np
from scipy.interpolate import interp1d

from pySDC.core.Errors import ParameterError
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


class Piline_DAE(ptype_dae):
    r"""
    This class implements the piline from the PinTSimE project modelled as system of differential-algebraic
    equations (DAEs). It serves as a transmission line in an energy grid. The voltage source :math:`V_s` is a
    generator that provides energy. Hence, the consumer modelled as resistor :math:`R_\ell` is always supplied
    with :math:`c V_s` with :math:`0 < c < 1`. At time :math:`t_{outage}` there is a power outage (and :math:`V_s`
    will be set to zero). When the energy :math:`V_{R_\ell}` drops below :math:`c V_s`, i.e., when the
    state function :math:`h(V_{R_\ell}) := V_{R_\ell} - c V_s` changes from positive to negative, a backup
    generator starts to continue to supply the consumer with energy, and :math:`V_s` will be set back to its
    old value. The system of DAEs is given by:

    .. math::
        \dfrac{d V_{C_1} (t)}{dt} = \dfrac{1}{C_1} i_{C_1} (t),

    .. math::
        \dfrac{d V_{C_2} (t)}{dt} = \dfrac{1}{C_2} i_{C_2} (t),

    .. math::
        \dfrac{d i_{L_\pi} (t)}{dt} = \dfrac{1}{L_\pi} V_{L_\pi} (t),

    .. math::
        0 = V_{R_s} (t) - R_s i_{R_s} (t),

    .. math::
        0 = V_{R_\pi} (t) - R_\pi i_{R_\pi} (t),

    .. math::
        0 = V_{R_\ell} (t) - R_\ell i_{R_\ell} (t),

    .. math::
        0 = V_{C_1} (t) - V_{R_\pi} (t) - V_{L_\pi} (t) - V_{C_2} (t),

    .. math::
        0 = V_s - V_{R_s} (t) - V_{C_1} (t),

    .. math::
        0 = V_{C_2} (t) - V_{R_\ell} (t),

    .. math::
        0 = i_{R_s} (t) - i_{C_1} (t) - i_{R_\pi} (t),

    .. math::
        0 = i_{R_\pi} (t) - i_{L_\pi} (t),

    .. math::
        0 = i_{L_\pi} (t) - i_{C_2} (t) - i_{R_\ell} (t).

    The first three equations describe the i-v characteristics of both capacitors, :math:`C_1,\, C_2` and the
    inductor :math:`L_\pi`. Equations 4 to 6 are Ohm's law for restistors :math:`R_s,\, R_\pi,\, R_\ell`. In
    equations 7 to 9, and 10 to 12 Kirchhoff's voltage laws, and Kirchhoff's current laws can be found.

    Parameters
    ----------
    Vs : float, optional
        Voltage at the voltage source :math:`V_s`.
    Rs : float, optional
        Resistance of the resistor :math:`R_s` at the voltage source.
    C1 : float, optional
        Capacitance of the capacitor :math:`C_1`.
    Rp : float, optional
        Resistance of the resistor in front of the inductor.
    Lp : float, optional
        Inductance of the inductor :math:`L_\pi`.
    C2 : float, optional
        Capacitance of the capacitor :math:`C_2`.
    Rl : float, optional
        Resistance of the resistor :math:`R_\ell`
    c : float, optional
        Value between 0 and 1. Indicates when the backup generator must restart.
    newton_tol : float, optional
        Tolerance of the Newton-like solver.

    Attributes
    ----------
    nvars: int
        Number of unknowns in the DAE system.
    power_outage: bool
        Indicates if the power outage already takes place or not.
    t_outage: float
        Time from when the power outage occurs.
    t_switch: float
        Time point of the discrete event found by switch estimation.
    nswitches: int
        Number of switches found by switch estimation.

    Note
    ----
    When default parameters will be changed, also initial conditions need to be adapted.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, Vs=150.0, Rs=1.0, C1=1e-2, Rp=0.2, Lp=1e-2, C2=1e-2, Rl=0.5, c=0.2, newton_tol=1e-12):
        """Initialization routine"""

       # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister(
            'Vs',
            'Rs',
            'C1',
            'Rp',
            'Lp',
            'C2',
            'Rl',
            'c',
            'newton_tol',
            localVars=locals(),
            readOnly=True,
        )

        assert 0 < self.c < 1, 'c needs to be between 0 and 1!'

        self.nvars = 12
        self.power_outage = False
        self.t_outage = 0.05
        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            Current value of the right-hand side of f (which includes 12 components).
        """

        vRs, vC1, vRp, vLp, vC2, vRl = u[0], u[1], u[2], u[3], u[4], u[5]
        iRs, iC1, iRp, iLp, iC2, iRl = u[6], u[7], u[8], u[9], u[10], u[11]

        dvC1, dvC2, diLp = du[1], du[4], du[9]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if t < self.t_outage:
            Vs = self.Vs
        elif t >= self.t_outage and vRl > self.c * self.Vs and not self.power_outage:
            Vs = 0
        elif t >= self.t_outage and vRl <= self.c * self.Vs or t >= t_switch:
            Vs = self.Vs
            self.power_outage = True
        else:
            Vs = self.Vs

        f = self.dtype_f(self.init)
        f[:] = (
            dvC1 - iC1 / self.C1,
            dvC2 - iC2 / self.C2,
            diLp - vLp / self.Lp,
            vRs - self.Rs * iRs,
            vRp - self.Rp * iRp,
            vRl - self.Rl * iRl,
            vC1 - vRp - vLp - vC2,
            Vs - vRs - vC1,
            vC2 - vRl,
            iRs - iC1 - iRp,
            iRp - iLp,
            iLp - iC2 - iRl,
        )
        return f

    def u_exact(self, t):
        """
        Routine for the exact solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object containing 12 components.
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)
        me[0] = 0.0  # vRs
        me[1] = 61.764705876572364  # vC1
        me[2] = 0.0  # vRp
        me[3] = 0.0  # vLp
        me[4] = 44.11764706251437  # vC2
        me[5] = 44.11764706251437  # vRl
        me[6] = 0.0  # iRs
        me[7] = 0.0  # iC1
        me[8] = 0.0  # iRp
        me[9] = 88.23529411989813  # iLp
        me[10] = 0.0  # iC2
        me[11] = 0.0  # iRl

        if me[5] < self.c * self.Vs:
            raise ParameterError(f"vRl has to be initialized greater than {self.c*self.Vs}!")

        return me

    def get_switching_info(self, u, t):
        """
        Provides information about the state function of the problem. When the state function changes its sign,
        typically an event occurs.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        switch_detected : bool
            Indicates whether a discrete event is found or not.
        m_guess : int
            The index before the sign changes.
        state_function : list
            Defines the values for finding the root at collocation nodes.
        """
        switch_detected = False
        m_guess = -100

        for m in range(1, len(u)):
            h_prev = u[m - 1][5] - self.c * self.Vs
            h_curr = u[m][5] - self.c * self.Vs
            if h_prev > 0 and h_curr <= 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [u[m][5] - self.c * self.Vs for m in range(len(u))] if switch_detected else []

        return switch_detected, m_guess, vC_switch

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1
