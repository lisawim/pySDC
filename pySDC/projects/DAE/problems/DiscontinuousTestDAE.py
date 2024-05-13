import numpy as np

from pySDC.core.Problem import WorkCounter
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.core.Errors import ProblemError


class DiscontinuousTestDAE(ptype_dae):
    r"""
    This class implements a scalar test discontinuous differential-algebraic equation (DAE) similar to [1]_. The event function
    is defined as :math:`h(y):= 2y - 100`. Then, the discontinuous DAE model reads:

    - if :math:`h(y) \leq 0`:

        .. math::
            \dfrac{d y(t)}{dt} = z(t),

        .. math::
            y(t)^2 - z(t)^2 - 1 = 0,

    else:

        .. math::
            \dfrac{d y(t)}{dt} = 0,

        .. math::
            y(t)^2 - z(t)^2 - 1 = 0,

    for :math:`t \geq 1`. If :math:`h(y) < 0`, the solution is given by

    .. math::
        (y(t), z(t)) = (cosh(t), sinh(t)),

    and the event takes place at :math:`t^* = 0.5 * arccosh(50) = 4.60507` and event point :math:`(cosh(t^*), sinh(t^*))`.
    As soon as :math:`t \geq t^*`, i.e., for :math:`h(y) \geq 0`, the solution is given by the constant value
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
        Exact time of the event :math:`t^*`.
    t_switch: float
        Time point of the event found by switch estimation.
    nswitches: int
        Number of switches found by switch estimation.

    References
    ----------
    .. [1] L. Lopez, S. Maset. Numerical event location techniques in discontinuous differential algebraic equations.
        Appl. Numer. Math. 178, 98-122 (2022).
    """

    def __init__(self, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=2, newton_tol=newton_tol)

        self.t_switch_exact = np.arccosh(50)
        self.t_switch = None
        self.nswitches = 0
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
            The right-hand side of f (contains two components).
        """

        y, z = u.diff[0], u.alg[0]
        dy = du.diff[0]

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = 2 * y - 100
        f = self.dtype_f(self.init)

        if h >= 0 or t >= t_switch:
            f.diff[0] = dy
            f.alg[0] = y**2 - z**2 - 1
        else:
            f.diff[0] = dy - z
            f.alg[0] = y**2 - z**2 - 1
        self.work_counters['rhs']()
        return f

    def u_exact(self, t, **kwargs):
        r"""
        Routine for the exact solution at time :math:`t \leq 1`. For this problem, the exact
        solution is piecewise.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        assert t >= 1, 'ERROR: u_exact only available for t>=1'

        me = self.dtype_u(self.init)
        if t <= self.t_switch_exact:
            me.diff[0] = np.cosh(t)
            me.alg[0] = np.sinh(t)
        else:
            me.diff[0] = np.cosh(self.t_switch_exact)
            me.alg[0] = np.sinh(self.t_switch_exact)
        return me

    def du_exact(self, t, **kwargs):
        r"""
        Routine for the exact solution at time :math:`t \leq 1`. For this problem, the exact
        solution is piecewise.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        assert t >= 1, 'ERROR: u_exact only available for t>=1'

        me = self.dtype_u(self.init)
        if t <= self.t_switch_exact:
            me.diff[0] = np.sinh(t)
            me.alg[0] = np.cosh(t)
        else:
            me.diff[0] = np.sinh(self.t_switch_exact)
            me.alg[0] = np.cosh(self.t_switch_exact)
        return me

    def get_switching_info(self, u, t):
        r"""
        Provides information about the state function of the problem. A change in sign of the state function
        indicates an event. If a sign change is detected, a root can be found within the step according to the
        intermediate value theorem.

        The state function for this problem is given by

        .. math::
           h(y(t)) = 2 y(t) - 100.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time :math:`t`.
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
            h_prev_node = 2 * u[m - 1].diff[0] - 100
            h_curr_node = 2 * u[m].diff[0] - 100
            if h_prev_node < 0 and h_curr_node >= 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [2 * u[m].diff[0] - 100 for m in range(len(u))]
        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1


class DiscontinousTestDAEIntegralFormulation(DiscontinuousTestDAE):
    def __init__(self, nvars=2, newton_tol=1e-12, newton_maxiter=100, stop_at_maxiter=False, stop_at_nan=True, eps=0.001):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister('newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', 'eps', localVars=locals())
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

    def eval_f(self, u, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains two components).
        """

        y, z = u.diff[0], u.alg[0]

        t_switch = np.inf if self.t_switch is None else self.t_switch
        h = 2 * y - 100

        f = self.dtype_f(self.init)
        f.diff[0] = 0 if h >= 0 or t >= t_switch else z
        f.alg[0] = y**2 - z**2 - 1
        self.work_counters['rhs']()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the nonlinear system.
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (required here for the BC).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """

        u = self.dtype_u(u0)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        # start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            # form the function g(u), such that the solution to the nonlinear problem is a root of g

            f = self.eval_f(u, t)
            y, z = u.diff[0], u.alg[0]
            g = np.array([y - factor * f.diff[0] - rhs.diff[0], f.alg[0]])

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            # print(n, res)
            if res < self.newton_tol:
                break
            print(n, res)
            h = 2 * y - 100
            if h >= 0 or t >= t_switch:
                dg = np.array([[1, 0], [2 * y, -2 * z]])
            else:
                dg = np.array([[1, -factor], [2 * y, -2 * z]])
            
            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        # if n == self.newton_maxiter:
        #     msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
        #     if self.stop_at_maxiter:
        #         raise ProblemError(msg)
        #     else:
        #         self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me

