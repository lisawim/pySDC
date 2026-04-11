import numpy as np

from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class PilineDAE(ProblemDAE):
    r"""
    This class implements the piline from the PinTSimE project modelled as system of differential-algebraic
    equations (DAEs). It serves as a transmission line in an energy grid. The system of DAEs modelling the
    purely piline is given by:

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
    nvars : int
        Number of unknowns in the DAE system.

    Note
    ----
    When default parameters will be changed, also initial conditions need to be adapted.
    """

    def __init__(
        self,
        Vs=100.0,
        Rs=1.0,
        C1=1.0,
        Rp=0.2,
        Lp=1.0,
        C2=1.0,
        Rl=5.0,
        nvars=12,
        newton_tol=1e-12,
    ):
        """Initialization routine"""

        # invoke super init, passing number of dofs
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister(
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

        VC1, VC2, iLpi = u.diff[0], u.diff[1], u.diff[2]
        dVC1, dVC2, diLpi = du.diff[0], du.diff[1], du.diff[2]

        VRs, iRs, VRpi, iRpi, VLpi = u.alg[0], u.alg[1], u.alg[2], u.alg[3], u.alg[4]
        VRl, iRl, iC1, iC2 = u.alg[5], u.alg[6], u.alg[7], u.alg[8]

        f = self.dtype_f(self.init)

        f.diff[0] = dVC1 - iC1 / self.C1
        f.diff[1] = dVC2 - iC2 / self.C2
        f.diff[2] = diLpi - VLpi / self.Lp

        f.alg[0] = VRs - self.Rs * iRs
        f.alg[1] = VRpi - self.Rp * iRpi
        f.alg[2] = VRl - self.Rl * iRl
        f.alg[3] = VC1 - VRpi - VLpi - VC2
        f.alg[4] = self.Vs - VRs - VC1
        f.alg[5] = VC2 - VRl
        f.alg[6] = iRs - iC1 - iRpi
        f.alg[7] = iRpi - iLpi
        f.alg[8] = iLpi - iC2 - iRl
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
        me.diff[0], me.diff[1], me.diff[2] = 0.0, 0.0, 0.0

        me.alg[0], me.alg[1], me.alg[2], me.alg[3], me.alg[4] = 0.0, 0.0, 0.0, 0.0, 0.0
        me.alg[5], me.alg[6], me.alg[7], me.alg[8] = 0.0, 0.0, 0.0, 0.0
        return me
