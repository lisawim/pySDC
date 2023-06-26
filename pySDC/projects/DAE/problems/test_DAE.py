import warnings
import numpy as np

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae


class ScalarTestDAE(ptype_dae):
    r"""
    This class implements a scalar test discontinuous differential-algebraic equation (DAE) [1]_. The event function
    is defined as :math:`h(y, z):= 2yz - 100`. Then, the discontinuous DAE model reads:

    - if :math:`h(y, z) \leq 0`:

        .. math::
            \dfrac{d y(t)}{dt} = z(t),

        .. math::
            y(t)^2 - z(t)^2 - 1 = 0,

    else:

        .. math::
            \dfrac{d y(t)}{dt} = 0,

        .. math::
            y(t)^2 - z(t)^2 - 1 = 0,

    for :math:`t \geq 1`. If :math:`h(y, z) < 0`, the solution is given by

    .. math::
        (y(t), z(t)) = (cosh(t), sinh(t)),

    and the event takes place at :math:`t^* = 0.5 * arcsinh(100) = 2.6492` and event point :math:`(cosh(t^*), sinh(t^*))`.
    As soon as :math:`t \geq t^*`, i.e., for :math:`h(y, z) \geq 0`, the solution is given by the constant value
    :math:`(cosh(t^*), sinh(t^*))`.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    t_switch_exact: float
        Exact time of the event.
    t_switch: float
        Time point of the event found by switch estimation.
    nswitches: int
        Number of switches found by switch estimation.

    References
    ----------
    .. [1] L. Lopez, S. Maset. Numerical event location techniques in discontinuous differential algebraic equations.
        Appl. Numer. Math. 178, 98-122 (2022)
    """

    def __init__(self, nvars=2, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars, newton_tol)
        self._makeAttributeAndRegister('nvars', 'newton_tol', localVars=locals())

        self.t_switch_exact = 0.5 * np.arcsinh(100)
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
            Current value of the right-hand side of f (which includes three components).
        """

        y, z = u[0], u[1]
        dy = du[0]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = 2 * y * z - 100
        f = self.dtype_f(self.init)

        f_before_event = (
            dy - z,
            y**2 - z**2 - 1,
        )

        f_after_event = (
            dy,
            y**2 - z**2 - 1,
        )

        if self.t_switch is not None:
            f[:] = f_before_event if t <= self.t_switch else f_after_event
        else:
            f[:] = f_before_event if h <= 0 else f_after_event
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
            The reference solution as mesh object containing three components.
        """

        assert t >= 1, 'ERROR: u_exact only valid for t>=1'

        me = self.dtype_u(self.init)
        if t <= self.t_switch_exact:
            me[:] = (np.cosh(t), np.sinh(t))
        else:
            me[:] = (np.cosh(self.t_switch_exact), np.sinh(self.t_switch_exact))
        return me

    def get_switching_info(self, u, t):
        """
        Provides information about the state function of the problem. When the state function changes its sign,
        typically an event occurs. So the check for an event should be done in the way that the state function
        is checked for a sign change. If this is the case, the intermediate value theorem states a root in this
        step.

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

        for m in range(1, len(u)):
            h_prev_node = 2 * u[m - 1][0] * u[m - 1][1] - 100
            h_curr_node = 2 * u[m][0] * u[m][1] - 100
            if h_prev_node <= 0 and h_curr_node > 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [2 * u[m][0] * u[m][1] - 100 for m in range(len(u))] if switch_detected else []

        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1

