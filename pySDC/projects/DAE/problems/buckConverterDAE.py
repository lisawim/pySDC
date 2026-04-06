import numpy as np

from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class BuckConverterDAE(ProblemDAE):
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

    def __init__(
            self,
            duty=0.5,
            fsw=1e3,
            Vs=10.0,
            Rs=5.0,
            C1=1e-3,
            Rp=0.01,
            Lp=1e-3,
            C2=1e-3,
            Rl=10.0,
            nvars=12,
            newton_tol=1e-12,
        ):
        """Initialization routine"""

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister(
            "duty",
            "fsw",
            "Vs",
            "Rs",
            "C1",
            "Rp",
            "Lp",
            "C2",
            "Rl",
            "nvars",
            "newton_tol",
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

        VC1, VC2, iLpi = u.diff[0], u.diff[1], u.diff[2]
        dVC1, dVC2, diLpi = du.diff[0], du.diff[1], du.diff[2]

        VRs, iRs, VRpi, iRpi, VLpi = u.alg[0], u.alg[1], u.alg[2], u.alg[3], u.alg[4]
        VRl, iRl, iC1, iC2 = u.alg[5], u.alg[6], u.alg[7], u.alg[8]

        f = self.dtype_f(self.init)

        Tsw = 1 / self.fsw

        f.diff[0] = dVC1 - iC1 / self.C1
        f.diff[1] = dVC2 - iC2 / self.C2
        f.diff[2] = diLpi - VLpi / self.Lp
        if 0 <= ((t / Tsw) % 1) <= self.duty:
            f.alg[0] = VRs - self.Rs * iRs
            f.alg[1] = VRpi - self.Rp * iRpi
            f.alg[2] = VRl - self.Rl * iRl
            f.alg[3] = VC1 - VRpi - VLpi - VC2
            f.alg[4] = self.Vs - VRs - VC1
            f.alg[5] = VC2 - VRl
            f.alg[6] = iRs - iC1 - iRpi
            f.alg[7] = iRpi - iLpi
            f.alg[8] = iLpi - iC2 - iRl

        else:
            f.alg[0] = VRs - self.Rs * iC1
            f.alg[1] = VRpi - self.Rp * iRpi
            f.alg[2] = VRl - self.Rl * iRl
            f.alg[3] = VRs + VC1 - self.Vs
            f.alg[4] = VRpi + VLpi + VC2
            f.alg[5] = VRl - VC2
            f.alg[6] = iRpi - iC1
            f.alg[7] = iLpi - iRpi
            f.alg[8] = iC2 + iRl - iLpi

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

        assert t == 0.0, "ERROR: u_exact only valid for t=0"

        me = self.dtype_u(self.init)
        me.diff[0], me.diff[1], me.diff[2] = 0.0, 0.0, 0.0

        me.alg[0], me.alg[1], me.alg[2], me.alg[3], me.alg[4] = 0.0, 0.0, 0.0, 0.0, 0.0
        me.alg[5], me.alg[6], me.alg[7], me.alg[8] = 0.0, 0.0, 0.0, 0.0
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
        #print([self.V_ref - u[m][1] for m in range(len(u))])
        for m in range(len(u)):
            if self.V_ref - u[m - 1][1] > 0 and self.V_ref - u[m][1] <= 0:
                switch_detected = True
                m_guess = m - 1
                break

            elif self.V_ref - u[m - 1][1] <= 0 and self.V_ref - u[m][1] > 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [self.V_ref - u[m][1] for m in range(len(u))] if switch_detected else []

        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1
