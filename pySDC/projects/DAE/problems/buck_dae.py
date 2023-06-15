import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


class BuckConverterDAE(ptype_dae):
    """
    Example implementing the buck converter model in a manipulated fashion, modelled as system of differential-algebraic
    equations (DAEs). When the state function :math:`h(V_{C_2} (t)) := V_{refmax} - V_{C_2}(t)` changes the sign, the
    switching states will be changed. This model is defined as follows:

        - :math:`V_{refmax} > V_{C_2}(t)` (S_1 = 1,\,S_2 = 0):

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
                0 = i_{L_\pi} (t) - i_{C_2} (t) - i_{R_\ell} (t)

        - :math:`V_{refmax} \leq V_{C_2}(t)` (S_1 = 0,\,S_2 = 1):

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
                0 = V_{R_\pi} (t) + V_{L_\pi} (t) + V_{C_2} (t),

            .. math::
                0 = V_s - V_{R_s} (t) - V_{C_1} (t),

            .. math::
                0 = V_{C_2} (t) - V_{R_\ell} (t),

            .. math::
                0 = i_{R_s} (t) - i_{C_1} (t),

            .. math::
                0 = i_{C_1} (t) - i_{R_\pi} (t),

            .. math::
                0 = i_{R_\pi} (t) - i_{L_\pi} (t),

            .. math::
                0 = i_{L_\pi} (t) - i_{C_2} (t) - i_{R_\ell} (t).

    If :math:`h(V_{C_2} (t)) \leq 0` the DAE system consists of 13 equations whereas in the case of :math:`h(V_{C_2} (t)) > 0`
    it only contains 12 equations.

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
    V_refmax : float, optional
        Reference at which the states will be changed.
    nvars : int, optional
        Number of unknowns in the DAE system.
    newton_tol : float, optional
        Tolerance of the Newton-like solver.

    Attributes
    ----------
    t_switch: float
        Time point of the discrete event found by switch estimation.
    nswitches: int
        Number of switches found by switch estimation.
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, Vs=10.0, Rs=1.0, C1=1.0, Rp=0.2, Lp=1.0, C2=1.0, Rl=5.0, V_refmax=8, nvars=13, newton_tol=1e-12):
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
            'V_refmax',
            'nvars',
            'newton_tol',
            localVars=locals(),
            readOnly=True,
        )

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
            Current value of the right-hand side of f (which includes 13 components).
        """
        vRs, vC1, vRp, vLp, vC2, vRl = u[0], u[1], u[2], u[3], u[4], u[5]
        iRs, iC1, iRp, iLp, iC2, iRl = u[6], u[7], u[8], u[9], u[10], u[11]

        dvC1, dvC2, diLp = du[1], du[4], du[9]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        first_state_f = (
            dvC1 - iC1 / self.C1,
            dvC2 - iC2 / self.C2,
            diLp - vLp / self.Lp,
            vRs - self.Rs * iRs,
            vRp - self.Rp * iRp,
            vRl - self.Rl * iRl,
            vC1 - vRp - vLp - vC2,
            self.Vs - vRs - vC1,
            vC2 - vRl,
            iRs - iC1 - iRp,
            iRp - iLp,
            iLp - iC2 - iRl,
            0,
        )

        second_state_f = (
            dvC1 - iC1 / self.C1,
            dvC2 - iC2 / self.C2,
            diLp - vLp / self.Lp,
            vRs - self.Rs * iRs,
            vRp - self.Rp * iRp,
            vRl - self.Rl * iRl,
            vRp + vLp + vC2,
            self.Vs - vRs - vC1,
            vC2 - vRl,
            iRs - iC1,
            iC1 - iRp,
            iRp - iLp,
            iLp - iC2 - iRl,
        )

        f = self.dtype_f(self.init)
        if self.V_refmax > u[4] and t < t_switch:
            f[:] = first_state_f
        elif self.V_refmax <= u[4] and t >= t_switch:
            f[:] = second_state_f
        elif self.V_refmax <= u[4] and t < t_switch:
            f[:] = second_state_f
        elif self.V_refmax > u[4] and t >= t_switch:
            f[:] = first_state_f

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
            The reference solution as mesh object containing 13 components (last one is always zero).
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'

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

    def get_switching_info(self, u, t):
        """
        Provides information about the state function of the problem. When the state function changes its sign,
        typically an event occurs. Usually, this model contains more than one discrete event, so it has to be
        proven whether the sign changes from positive to negative, or vice versa.

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
            Defines the values of the state function at collocation nodes where it changes the sign.
        """

        switch_detected = False
        m_guess = -100

        for m in range(len(u)):
            if self.V_refmax - u[m - 1][4] > 0 and self.V_refmax - u[m][4] <= 0:
                switch_detected = True
                m_guess = m - 1
                break

            elif self.V_refmax - u[m - 1][4] <= 0 and self.V_refmax - u[m][4] > 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [self.V_refmax - u[m][4] for m in range(len(u))] if switch_detected else []

        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1
