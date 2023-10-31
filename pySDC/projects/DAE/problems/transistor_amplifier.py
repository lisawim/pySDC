import warnings
import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.core.Problem import WorkCounter


# Helper function
def _transistor(u_in):
    return 1e-6 * (np.exp(u_in / 0.026) - 1)


class one_transistor_amplifier(ptype_dae):
    r"""
    The one transistor amplifier example from pg. 404 in [1]_. The problem is an index-1 differential-algebraic equation
    (DAE) having the equations

    .. math::
        \frac{U_e (t)}{R_0} - \frac{U_1 (t)}{R_0} + C_1 (\frac{d U_2 (t)}{dt} - \frac{d U_1 (t)}{dt}) = 0,

    .. math::
        \frac{U_b}{R_2} - U_2 (t) (\frac{1}{R_1} + \frac{1}{R_2}) + C_1 (\frac{d U_1 (t)}{dt} - \frac{d U_2 (t)}{dt}) - 0.01 f(U_2 (t) - U_3 (t)) = 0,

    .. math::
        f(U_2 (t) - U_3 (t)) - \frac{U_3 (t)}{R_3} - C_2 \frac{d U_3 (t)}{dt} = 0,

    .. math::
        \frac{U_b}{R_4} - \frac{U_4 (t)}{R_4} + C_3 (\frac{d U_5 (t)}{dt} - \frac{d U_4 (t)}{dt}) - 0.99 f(U_2 (t) - U_3 (t)) = 0,

    .. math::
        -\frac{U_5 (t)}{R_5} + C_3 (\frac{d U_4 (t)}{dt} - \frac{d U_5 (t)}{dt}) = 0,

    with

    .. math::
        f(U(t)) = 10^{-6} (exp(\frac{U (t)}{0.026}) - 1).

    The initial signal :math:`U_e (t)` is defined as

    .. math::
        U_e (t) = 0.4 \sin(200 \pi t).

    Constants are fixed as :math:`U_b = 6`, :math:`R_0 = 1000`, :math:`R_k = 9000` for :math:`k=1,..,5`,
    `C_j = j \cdot 10^{-6}` for :math:`j=1,2,3`.They are also defined in the method `eval_f`.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    t_end: float
        The end time at which the reference solution is determined.
    work_counters : WorkCounter
        Counts the work, i.e., number of function calls of right-hand side is called and stored in
        ``work_counters['rhs']``.

    References
    ----------
    .. [1] E. Hairer, G. Wanner. Solving ordinary differential equations II: Stiff and differential-algebraic problems.
        Springer (2009).
    """

    def __init__(self, nvars, newton_tol):
        super().__init__(nvars, newton_tol)
        # load reference solution
        # data file must be generated and stored under misc/data and self.t_end = t[-1]
        # data = np.load(r'pySDC/projects/DAE/misc/data/one_trans_amp.npy')
        # x = data[:, 0]
        # # The last column contains the input signal
        # y = data[:, 1:-1]
        # self.u_ref = interp1d(x, y, kind='cubic', axis=0, fill_value='extrapolate')
        self.t_end = 0.0

        self.work_counters['rhs'] = WorkCounter()

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
            Current value of the right-hand side of f (which includes five components).
        """
        u_b = 6.0
        u_e = 0.4 * np.sin(200 * np.pi * t)
        alpha = 0.99
        r_0 = 1000
        r_k = 9000
        c_1, c_2, c_3 = 1e-6, 2e-6, 3e-6
        f = self.dtype_f(self.init)
        f[:] = (
            (u_e - u[0]) / r_0 + c_1 * (du[1] - du[0]),
            (u_b - u[1]) / r_k - u[1] / r_k + c_1 * (du[0] - du[1]) - (1 - alpha) * _transistor(u[1] - u[2]),
            _transistor(u[1] - u[2]) - u[2] / r_k - c_2 * du[2],
            (u_b - u[3]) / r_k + c_3 * (du[4] - du[3]) - alpha * _transistor(u[1] - u[2]),
            -u[4] / r_k + c_3 * (du[3] - du[4]),
        )
        self.work_counters['rhs']()
        return f

    def u_exact(self, t):
        """
        Approximation of the exact solution generated by spline interpolation of an extremely accurate numerical
        reference solution.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object containing five components and fixed initial conditions.
        """
        me = self.dtype_u(self.init)

        if t == 0:
            me[:] = (0, 3, 3, 6, 0)
        elif t < self.t_end:
            me[:] = self.u_ref(t)
        else:
            self.logger.warning("Requested time exceeds domain of the reference solution. Returning zero.")
            me[:] = (0, 0, 0, 0, 0)
        return me


class two_transistor_amplifier(ptype_dae):
    r"""
    The two transistor amplifier example from page 108 in [1]_. The problem is an index-1 differential-algebraic equation
    (DAE) having the equations

    .. math::
        \frac{U_e (t)}{R_0} - \frac{U_1 (t)}{R_0} + C_1 (\frac{d U_2 (t)}{dt} - \frac{d U_1 (t)}{dt}) = 0,

    .. math::
        \frac{U_b}{R_2} - U_2 (t) (\frac{1}{R_1} + \frac{1}{R_2}) + C_1 (\frac{d U_1 (t)}{dt} - \frac{d U_2 (t)}{dt}) - (\alpha - 1) f(U_2 (t) - U_3 (t)) = 0,

    .. math::
        f(U_2 (t) - U_3 (t)) - \frac{U_3 (t)}{R_3} - C_2 \frac{d U_3 (t)}{dt} = 0,

    .. math::
        \frac{U_b}{R_4} - \frac{U_4 (t)}{R_4} + C_3 (\frac{d U_5 (t)}{dt} - \frac{d U_4 (t)}{dt}) - \alpha f(U_2 (t) - U_3 (t)) = 0,

    .. math::
        \frac{U_b}{R_6} - U_5 (t) (\frac{1}{R_5} + \frac{1}{R_6}) + C_3 (\frac{d U_4 (t)}{dt} - \frac{d U_5 (t)}{dt}) + (\alpha - 1) f(U_5 (t) - U_6 (t)) = 0,

    .. math::
        f(U_5 (t) - U_6 (t)) - \frac{U_6 (t)}{R_7} - C_4 \frac{d U_6 (t)}{dt} = 0,

    .. math::
        \frac{U_b}{R_8} - \frac{U_7 (t)}{R_8} - C_5 (\frac{d U_7 (t)}{dt} - \frac{d U_8 (t)}{dt}) - \alpha f(U_5 (t) - U_6 (t)) = 0,

    .. math::
        \frac{U_8 (t)}{R_9} - C_5 (\frac{d U_7 (t)}{dt} - \frac{d U_7 (t)}{dt}) = 0,

    with

    .. math::
        f(U_i (t) - U_j (t)) = \beta (\exp(\frac{U_i (t) - U_j (t)}{U_F}) - 1).

    The initial signal :math:`U_e (t)` is defined as

    .. math::
        U_e (t) = 0.1 \sin(200 \pi t).

    Constants are fixed as :math:`U_b = 6`, :math:`U_F = 0.026`, :math:`\alpha = 0.99`, :math:`\beta = 10^{-6}`, :math:`R_0 = 1000`,
    :math:`R_k = 9000` for :math:`k=1,..,9`, `C_j = j \cdot 10^{-6}` for :math:`j=1,..,5`. They are also defined in the
    method `eval_f`.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs.
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    t_end: float
        The end time at which the reference solution is determined.
    work_counters : WorkCounter
        Counts the work, i.e., number of function calls of right-hand side is called and stored in
        ``work_counters['rhs']``.

    References
    ----------
    .. [1] E. Hairer, C. Lubich, M. Roche. The numerical solution of differential-algebraic systems by Runge-Kutta methods.
        Lect. Notes Math. (1989).
    """

    def __init__(self, nvars, newton_tol):
        super().__init__(nvars, newton_tol)
        # load reference solution
        # data file must be generated and stored under misc/data and self.t_end = t[-1]
        # data = np.load(r'pySDC/projects/DAE/misc/data/two_trans_amp.npy')
        # x = data[:, 0]
        # The last column contains the input signal
        # y = data[:, 1:-1]
        # self.u_ref = interp1d(x, y, kind='cubic', axis=0, fill_value='extrapolate')
        self.t_end = 0.0

        self.work_counters['rhs'] = WorkCounter()

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
            Current value of the right-hand side of f (which includes eight components).
        """
        u_b = 6.0
        u_e = 0.1 * np.sin(200 * np.pi * t)
        alpha = 0.99
        r_0 = 1000.0
        r_k = 9000.0
        c_1, c_2, c_3, c_4, c_5 = 1e-6, 2e-6, 3e-6, 4e-6, 5e-6
        f = self.dtype_f(self.init)
        f[:] = (
            (u_e - u[0]) / r_0 - c_1 * (du[0] - du[1]),
            (u_b - u[1]) / r_k - u[1] / r_k + c_1 * (du[0] - du[1]) + (alpha - 1) * _transistor(u[1] - u[2]),
            _transistor(u[1] - u[2]) - u[2] / r_k - c_2 * du[2],
            (u_b - u[3]) / r_k - c_3 * (du[3] - du[4]) - alpha * _transistor(u[1] - u[2]),
            (u_b - u[4]) / r_k - u[4] / r_k + c_3 * (du[3] - du[4]) + (alpha - 1) * _transistor(u[4] - u[5]),
            _transistor(u[4] - u[5]) - u[5] / r_k - c_4 * du[5],
            (u_b - u[6]) / r_k - c_5 * (du[6] - du[7]) - alpha * _transistor(u[4] - u[5]),
            -u[7] / r_k + c_5 * (du[6] - du[7]),
        )
        self.work_counters['rhs']()
        return f

    def u_exact(self, t):
        """
        Dummy exact solution that should only be used to get initial conditions for the problem. This makes
        initialisation compatible with problems that have a known analytical solution. Could be used to output a
        reference solution if generated/available.

        Parameters
        ----------
        t : float
            The time of the reference solution.

        Returns
        -------
        me : dtype_u
            The reference solution as mesh object containing eight components and fixed initial conditions.
        """
        me = self.dtype_u(self.init)

        if t == 0:
            me[:] = (0, 3, 3, 6, 3, 3, 6, 0)
        elif t < self.t_end:
            me[:] = self.u_ref(t)
        else:
            self.logger.warning("Requested time exceeds domain of the reference solution. Returning zero.")
            me[:] = (0, 0, 0, 0, 0, 0, 0, 0)
        return me
