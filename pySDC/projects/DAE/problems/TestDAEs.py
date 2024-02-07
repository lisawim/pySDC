import numpy as np
from scipy.optimize import root

from pySDC.core.Problem import WorkCounter
from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.projects.DAE.misc.dae_mesh import DAEMesh
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import ParameterError


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

    dtype_u = DAEMesh
    dtype_f = DAEMesh

    def __init__(self, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=(1, 1), newton_tol=newton_tol)
        self._makeAttributeAndRegister('newton_tol', localVars=locals())

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

        y, z = u.diff, u.alg
        dy = du.diff

        t_switch = np.inf if self.t_switch is None else self.t_switch

        h = 2 * y - 100
        f = self.dtype_f(self.init)

        if h >= 0 or t >= t_switch:
            f.diff[:] = dy
            f.alg[:] = y**2 - z**2 - 1
        else:
            f.diff[:] = dy - z
            f.alg[:] = y**2 - z**2 - 1
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
            me.diff[:] = np.cosh(t)
            me.alg[:] = np.sinh(t)
        else:
            me.diff[:] = np.cosh(self.t_switch_exact)
            me.alg[:] = np.sinh(self.t_switch_exact)
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
            h_prev_node = 2 * u[m - 1].diff - 100
            h_curr_node = 2 * u[m].diff - 100
            if h_prev_node < 0 and h_curr_node >= 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [2 * u[m].diff - 100 for m in range(len(u))]
        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1


class DiscontinuousTestDAEWithAlgebraicStateFunction(DiscontinuousTestDAE):
    r"""
    This class implements the scalar test discontinuous DAE as in ``DiscontinuousTestDAE``. The difference to
    the parent class is the different state function :math:`h(y):= 2yz - 100`, which uses the algebraic variable
    :math:`z` here. Hence, also the exact event time ``t_switch_exact`` changes.

    Note
    ----
    For this class, the dynamics are the same as for the parent class. The exact event time ``t_switch_exact`` is
    different, which means that the root of the state function. i.e., the discrete event occurs earlier.

    All attributes will be inherited here except for ``t_switch_exact`` which is overwritten.

    Moreover, this class is interesting to observe in combination with the SDC sweeper ``SemiExplicitDAE`` since the
    parent class implements a semi-explicit DAE.
    """

    def __init__(self, nvars=2, newton_tol=1e-12):
        """Initialization routine"""
        nvars = 2
        super().__init__(nvars, newton_tol)

        self.t_switch_exact = 0.5 * np.arcsinh(100)
        self.diff_nvars = 1

    def eval_state_function(self, u, t):
        r"""
        Evaluates the state function :math:`h(y(t), z(t)) = 2 y(t) z(t) - 100`.

        Parameters
        ----------
        u : dtype_u
            Current numerical solution.
        t : float
            Current time :math:`t`.

        Returns
        -------
        state_function : float
            Value of state function at time :math:`t`.
        """
        state_function = 2 * u[0] * u[1] - 100
        return state_function


class LinearTestDAE(ptype_dae):
    """
    Semi-explicit linear DAE of index one. It reads

    .. math::
        \dfrac{d}{dt}u_d = \lambda_d u_d + \lambda_a u_a,

    .. math::
        0 = \lambda_d u_d - \lambda_a u_a,

    where :math:`u_d` is the differential variable and :math:`u_a` is denoted as the algebraic
    variable. :math:`\lambda_d` and :math:`\lambda_a` are non-zero fixed parameters.

    Parameters
    ----------
    lamb_diff : float
        Parameter :math:`\lambda_d`.
    lamb_alg : float
        Parameter :math:`\lambda_a`.
    newton_tol : float
        Tolerance for inner solver to terminate.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts work, here, the number of right-hand side evaluations and work in inner solver
        are counted.
    """
    # dtype_u = mesh
    # dtype_f = mesh

    def __init__(self, lamb_diff=1.0, lamb_alg=1.0, newton_tol=1e-12):
        """Initialization routine"""
        super().__init__(nvars=(1, 1), newton_tol=newton_tol)
        self._makeAttributeAndRegister('newton_tol', 'lamb_diff', 'lamb_alg', localVars=locals())
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['newton'] = WorkCounter()

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

        u_diff, u_alg = u.diff, u.alg
        du_diff = du.diff

        f = self.dtype_f(self.init)
        f.diff[:] = du_diff - self.lamb_diff * u_diff - self.lamb_alg * u_alg
        f.alg[:] = self.lamb_diff * u_diff - self.lamb_alg * u_alg
        return f

    def u_exact(self, t, **kwargs):
        r"""
        Routine for the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[:] = np.exp(2 * self.lamb_diff * t)
        me.alg[:] = self.lamb_diff / self.lamb_alg * np.exp(2 * self.lamb_diff * t)
        return me

    def du_exact(self, t, **kwargs):
        r"""
        Routine for the derivative exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution's derivative.

        Returns
        -------
        me : dtype_u
            Derivative exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[:] = 2 * self.lamb_diff * np.exp(2 * self.lamb_diff * t)
        me.alg[:] = (2 * self.lamb_diff ** 2) / self.lamb_alg * np.exp(2 * self.lamb_diff * t)
        return me


class LinearTestDAEMinion(ptype_dae):
    r"""
    TODO: Write docu about class!
    """

    def __init__(self, newton_tol=1e-12, method='gmres'):
        """Initialization routine"""
        super().__init__(nvars=(3, 1), newton_tol=newton_tol, method=method)
        self._makeAttributeAndRegister('newton_tol', localVars=locals())
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
            The right-hand side of f (contains four components).
        """
        c = 1e4
        u1, u2, u3, u4 = u.diff[0], u.diff[1], u.diff[2], u.alg[0]
        du1, du2, du3 = du.diff[0], du.diff[1], du.diff[2]

        f = self.dtype_f(self.init)
        f.diff[:] = (
            u1 - u3 + u4 - du1,
            -c * u2 + (1 + c) * np.exp(t) - du2,
            u1 - du3,
        )
        f.alg[:] = u1 + u2 + u4 - np.exp(t)
        self.work_counters['rhs']()
        return f

    def u_exact(self, t, **kwargs):
        r"""
        Routine for the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            Exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[:] = (np.cos(t), np.exp(t), np.sin(t))
        me.alg[:] = -np.cos(t)
        return me

    def du_exact(self, t, **kwargs):
        r"""
        Routine for the derivative exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution's derivative.

        Returns
        -------
        me : dtype_u
            Derivative exact solution.
        """

        me = self.dtype_u(self.init)
        me.diff[:] = (-np.sin(t), np.exp(t), np.cos(t))
        me.alg[:] = np.sin(t)
        return me


class LinearTestDAEReduced(LinearTestDAE):
    r"""
    Reduced system of problem ``LinearTestDAE`` of the form

    .. math::
        \dfrac{d}{dt} u_d = 2 \lambda_d u_d.

    The function to update the algebraic variable is given by

    .. math::
        G(u_d) = \dfrac{\lambda_d}{\lambda_a}u_d = u_a,

    where :math:`\lambda_a \neq 0`.
    """

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

        u_diff = u.diff
        du_diff = du.diff

        f = self.dtype_f(self.init)

        f.diff[:] = du_diff - 2 * self.lamb_diff * u_diff
        self.work_counters['rhs']()
        return f

    def solve_system(self, impl_sys, u0, t):
        r"""
        Solver for nonlinear implicit system (defined in sweeper).

        Parameters
        ----------
        impl_sys : callable
            Implicit system to be solved.
        u0 : dtype_u
            Initial guess for solver.
        t : float
            Current time :math:`t`.

        Returns
        -------
        me : dtype_u
            Numerical solution.
        """
        me = self.dtype_u(self.init)
        def implSysReducedAsNumpy(unknowns, **kwargs):
            me.diff[:] = unknowns[: np.size(me.diff)].reshape(me.diff.shape)
            # me.alg[:] =  unknowns[np.size(me.diff) :].reshape(me.alg.shape)
            sys = impl_sys(me, **kwargs)
            return sys.diff.flatten()  # TODO: more efficient way?

        opt = root(
            implSysReducedAsNumpy,
            u0.diff.flatten(),
            method='hybr',
            tol=self.newton_tol,
        )
        me.diff[:] = opt.x[: np.size(me.diff)].reshape(me.diff.shape)
        # me.alg[:] = opt.x[np.size(me.diff) :].reshape(me.alg.shape)
        self.work_counters['newton'].niter += opt.nfev
        return me

    def eval_G(self, u, t):
        r"""
        Returns the value for the algebraic variable :math:`u_a`, see the documentation above.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        G : dtype_f
            Value of function G.
        """

        G = self.dtype_f(self.init)
        G.alg[:] = self.lamb_diff / self.lamb_alg * u.diff
        return G
