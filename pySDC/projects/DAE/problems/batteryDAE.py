import numpy as np

from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class BatteryDAE(ProblemDAE):
    r"""
    Battery drain model as a switched DAE.

    The state function is
        h(V_C(t)) = V_C(t) - V_ref.

    The active subsystem is chosen as

        - System (12), if h(V_C(t)) <= 0
        - System (6),  otherwise

    as stated in the PDF.

    Unknowns
    --------
    Differential variables:
        u.diff = [i_L, V_C]

    Algebraic variables:
        u.alg = [V_L, i_C, V_R, i_R, V_Rs, i_Rs, i_Vs]

    Hence, the ordering is
        y := [i_L, V_C]
        z := [V_L, i_C, V_R, i_R, V_Rs, i_Rs, i_Vs]

    Parameters
    ----------
    L : float
        Inductance.
    C : float
        Capacitance.
    R : float
        Resistance.
    Rs : float
        Source resistance.
    Vs : float
        Source voltage.
    V_ref : float
        Reference voltage for switching.
    newton_tol : float
        Tolerance for Newton solver.

    Notes
    -----
    According to the PDF, the switched model is:

    State 1: SC = 1, SV = 0  (battery supplies energy)
        d/dt i_L  = (1/L) V_L
        C d/dt V_C = i_C
        0 = V_R  - R i_R
        0 = V_Rs - Rs i_Rs
        0 = V_C - V_R
        0 = i_Vs + i_Rs
        0 = i_L - i_Rs
        0 = i_L
        0 = i_R + i_C

    State 2: SC = 0, SV = 1  (household switches to local power grid)
        d/dt i_L  = (1/L) V_L
        C d/dt V_C = i_C
        0 = V_R  - R i_R
        0 = V_Rs - Rs i_Rs
        0 = Vs - V_Rs - V_L - V_R
        0 = i_Vs + i_Rs
        0 = i_L - i_Rs
        0 = i_R - i_L
        0 = i_C
    """

    def __init__(self, Vs=5.0, Rs=0.5, C=1.0, R=1.0, L=1.0, alpha=5.0, V_ref=1.0, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=9, newton_tol=newton_tol)
        self._makeAttributeAndRegister("Vs", "Rs", "C", "R", "L", "alpha", "V_ref", localVars=locals(), readOnly=True
        )

        self.t_switch = None
        self.nswitches = 0

        self.work_counters["rhs"] = WorkCounter()

    def eval_f(self, u, du, t):
        r"""
        Evaluate the implicit residual F(u, u', t).

        Parameters
        ----------
        u : dtype_u
            Current numerical solution.
        du : dtype_u
            Current derivative.
        t : float
            Current time.

        Returns
        -------
        f : dtype_f
            Residual of the switched DAE system.
        """

        iL, VC = u.diff[0], u.diff[1]
        diL, dVC = du.diff[0], du.diff[1]

        VL, iC, VR, iR = u.alg[0], u.alg[1], u.alg[2], u.alg[3]
        VRs, iRs, iVs = u.alg[4], u.alg[5], u.alg[6]

        f = self.dtype_f(self.init)

        h = VC - self.V_ref

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if h <= 0 or t >= t_switch:
            f.diff[0] = diL - VL / self.L
            f.diff[1] = self.C * dVC - iC

            f.alg[0] = VR - self.R * iR
            f.alg[1] = VRs - self.Rs * iRs
            f.alg[2] = self.Vs - VRs - VL - VR
            f.alg[3] = iVs + iRs
            f.alg[4] = iL - iRs
            f.alg[5] = iR - iL
            f.alg[6] = iC

        else:
            f.diff[0] = diL - VL / self.L
            f.diff[1] = self.C * dVC - iC

            f.alg[0] = VR - self.R * iR
            f.alg[1] = VRs - self.Rs * iRs
            f.alg[2] = VC - VR
            f.alg[3] = iVs + iRs
            f.alg[4] = iL - iRs
            f.alg[5] = -iL
            f.alg[6] = iR + iC

        self.work_counters["rhs"]()
        return f

    def u_exact(self, t, **kwargs):
        r"""
        Routine to compute the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """

        assert t == 0.0, "ERROR: u_exact only valid for t=0"

        me = self.dtype_u(self.init)

        iL, VC = 0.0, self.alpha * self.V_ref

        VL, iC, VR, iR = 0.0, 0.0, 0.0, 0.0
        VRs, iRs, iVs = 0.0, 0.0, 0.0

        me.diff[0], me.diff[1] = iL, VC

        me.alg[0], me.alg[1], me.alg[2], me.alg[3] = VL, iC, VR, iR
        me.alg[4], me.alg[5], me.alg[6] = VRs, iRs, iVs
        return me

    def get_switching_info(self, u, t):
        r"""
        Detect a sign change of h(V_C) = V_C - V_ref along the collocation nodes.

        Parameters
        ----------
        u : list[dtype_u]
            Values at collocation nodes.
        t : float
            Current left interval point (not used explicitly here).

        Returns
        -------
        switch_detected : bool
            True if a sign change is detected.
        m_guess : int
            Index before the sign change.
        state_function : list[float]
            Values of h(V_C) at the nodes.
        """

        switch_detected = False
        m_guess = -100

        for m in range(1, len(u)):
            h_prev = u[m - 1].diff[1] - self.V_ref
            h_curr = u[m].diff[1] - self.V_ref

            if h_prev > 0 and h_curr <= 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [u[m].diff[1] - self.V_ref for m in range(len(u))]
        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Update the number of detected switches.
        """
        self.nswitches += 1
