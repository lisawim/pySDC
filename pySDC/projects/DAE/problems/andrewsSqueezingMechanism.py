import numpy as np

from pySDC.core.errors import ProblemError
from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE


class AndrewsSqueezingMechanismDAE(ProblemDAE):
    def __init__(self, newton_tol=1e-12, index=1):
        """Initialization routine"""
        super().__init__(nvars=27, newton_tol=newton_tol)
        self._makeAttributeAndRegister('newton_tol', 'index', localVars=locals())
        self.work_counters['rhs'] = WorkCounter()
        self.work_counters['newton'] = WorkCounter()

        self.m1 = 0.04325
        self.m2 = 0.00365
        self.m3 = 0.02373
        self.m4 = 0.00706
        self.m5 = 0.07050
        self.m6 = 0.00706
        self.m7 = 0.05498

        self.xa = -0.06934
        self.ya = -0.00227
        self.xb = -0.03635
        self.yb = 0.03273
        self.xc = 0.014
        self.yc = 0.072

        self.c0 = 4530

        self.I1 = 2.194e-6
        self.I2 = 4.410e-7
        self.I3 = 5.255e-6
        self.I4 = 5.667e-7
        self.I5 = 1.169e-5
        self.I6 = 5.667e-7
        self.I7 = 1.912e-5

        self.d = 0.028
        self.da = 0.0115
        self.e = 0.02
        self.ea = 0.01421
        self.rr = 0.007
        self.ra = 0.00092

        self.l0 = 0.07785

        self.ss = 0.035
        self.sa = 0.01874
        self.sb = 0.01043
        self.sc = 0.018
        self.sd = 0.02

        self.ta = 0.02308
        self.tb = 0.00916

        self.u = 0.04
        self.ua = 0.01228
        self.ub = 0.00449

        self.zf = 0.02
        self.zt = 0.04

        self.fa = 0.01421
        self.mom = 0.033

        self.M = np.zeros((7, 7))
        self.G = np.zeros((6, 7))
        self.func = np.zeros(7)
        self.g = np.zeros(6)
        self.gqqv = np.zeros(6)

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
            The right-hand side of f (contains 27 components).
        """

        # Shortcuts
        q, v = u.diff[: 7], u.diff[7 : 14]
        w = u.alg[: 7]

        dq, dv = du.diff[0 : 7], du.diff[7 : 14]

        f = self.dtype_f(self.init)
        f.diff[0 : 7] = dq[:] - v[:]
        f.diff[7 : 14] = dv[:] - w[:]

        f.alg[:] = self.algebraicConstraints(u, t)
        return f

    def algebraicConstraints(self, u, t):
        r"""
        Returns the algebraic constraints of the semi-explicit DAE system.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            Algebraic part of right-hand side of f (contains 13 components).
        """

        # Shortcuts
        q, v = u.diff[0 : 7], u.diff[7 : 14]
        w, l = u.alg[0 : 7], u.alg[7 : 13]

        # Get matrices and functions for algebraic part of right-hand side
        self.getM(q)
        self.get_func(q, v)
        self.getG(q)

        f = self.dtype_f(self.init)
        f.alg[0 : 7] = self.M.dot(w) - self.func + self.G.T.dot(l)
        if self.index == 3:
            self.get_g(q)

            f.alg[7 : 13] = self.g

        elif self.index == 2:
            f.alg[7 : 13] = self.G.dot(v)

        elif self.index == 1:
            self.get_gqq(q, v)

            f.alg[7 : 13] = self.gqqv + self.G.dot(w)
        else:
            raise NotImplementedError

        return f.alg[:]

    def get_func(self, q, v):
        r"""
        Returns function :math:`f` for algebraic equation of DAE system.

        Parameters
        ----------
        q : dtype_u
            Unknown :math:`q`.
        v : dtype_u
            Unknown :math:`v`.
        """

        # Shortcuts
        q1, q2, q3, q4, q5, q6, q7 = q[0], q[1], q[2], q[3], q[4], q[5], q[6]
        v1, v2, v3, v4, v5, v6, v7 = v[0], v[1], v[2], v[3], v[4], v[5], v[6]

        # Define entities
        xd = self.sd * np.cos(q3) + self.sc * np.sin(q3) + self.xb
        yd = self.sd * np.sin(q3) - self.sc * np.cos(q3) + self.yb
        L = np.sqrt((xd - self.xc) ** 2 + (yd - self.yc) ** 2)
        F = -self.c0 * (L - self.l0) / L
        Fx = F * (xd - self.xc)
        Fy = F * (yd - self.yc)

        # Initialize vector
        self.func[0] = self.mom - self.m2 * self.da * self.rr * v2 * (v2 + 2 * v1) * np.sin(q2)
        self.func[1] = self.m2 * self.da * self.rr * v1 ** 2 * np.sin(q2)
        self.func[2] = Fx * (self.sc * np.cos(q3) - self.sd * np.sin(q3)) + Fy * (self.sd * np.cos(q3) + self.sc * np.sin(q3))
        self.func[3] = self.m4 * self.zt * (self.e - self.ea) * v5 ** 2 * np.cos(q4)
        self.func[4] = -self.m4 * self.zt * (self.e - self.ea) * v4 * (v4 + 2 * v5) * np.cos(q4)
        self.func[5] = -self.m6 * self.u * (self.zf - self.fa) * v7 ** 2 * np.cos(q6)
        self.func[6] = self.m6 * self.u * (self.zf - self.fa) * v6 * (v6 + 2 * v7) * np.cos(q6)

    def getG(self, q):
        r"""
        Returns Jacobian of function g.

        Parameters
        ----------
        q : dtype_u
        """

        q1, q2, q3, q4, q5, q6, q7 = q[0], q[1], q[2], q[3], q[4], q[5], q[6]

        self.G[0, 0] = -self.rr * np.sin(q1) + self.d * np.sin(q1 + q2)
        self.G[0, 1] = self.d * np.sin(q1 + q2)
        self.G[0, 2] = -self.ss * np.cos(q3)        

        self.G[1, 0] = self.rr * np.cos(q1) - self.d * np.cos(q1 + q2)
        self.G[1, 1] = -self.d * np.cos(q1 + q2)
        self.G[1, 2] = -self.ss * np.sin(q3)

        self.G[2, 0] = -self.rr * np.sin(q1) + self.d * np.sin(q1 + q2)
        self.G[2, 1] = self.d * np.sin(q1 + q2)
        self.G[2, 3] = -self.e * np.cos(q4 + q5)
        self.G[2, 4] = -self.e * np.cos(q4 + q5) + self.zt * np.sin(q5)

        self.G[3, 0] = self.rr * np.cos(q1) - self.d * np.cos(q1 + q2)
        self.G[3, 1] = -self.d * np.cos(q1 + q2)
        self.G[3, 3] = -self.e * np.sin(q4 + q5)
        self.G[3, 4] = -self.e * np.sin(q4 + q5) - self.zt * np.cos(q5)

        self.G[4, 0] = -self.rr * np.sin(q1) + self.d * np.sin(q1 + q2)
        self.G[4, 1] = self.d * np.sin(q1 + q2)
        self.G[4, 5] = self.zf * np.sin(q6 + q7)
        self.G[4, 6] = self.zf * np.sin(q6 + q7) - self.u * np.cos(q7)

        self.G[5, 0] = self.rr * np.cos(q1) - self.d * np.cos(q1 + q2)
        self.G[5, 1] = -self.d * np.cos(q1 + q2)
        self.G[5, 5] = -self.zf * np.cos(q6 + q7)
        self.G[5, 6] = -self.zf * np.cos(q6 + q7) - self.u * np.sin(q7)

    def get_g(self, q):
        r"""
        Returns function :math:`g` for algebraic equation of DAE system.

        Parameters
        ----------
        q : dtype_u
            Unknown :math:`q`.

        Returns
        -------
        f : np.1darray
            Function :math:`g`.
        """

        # Shortcuts
        q1, q2, q3, q4, q5, q6, q7 = q[0], q[1], q[2], q[3], q[4], q[5], q[6]

        self.g[0] = self.rr * np.cos(q1) - self.d * np.cos(q1 + q2) - self.ss * np.sin(q3) - self.xb
        self.g[1] = self.rr * np.sin(q1) - self.d * np.sin(q1 + q2) + self.ss * np.cos(q3) - self.yb
        self.g[2] = self.rr * np.cos(q1) - self.d * np.cos(q1 + q2) - self.e * np.sin(q4 + q5) - self.zt * np.cos(q5) - self.xa
        self.g[3] = self.rr * np.sin(q1) - self.d * np.sin(q1 + q2) + self.e * np.cos(q4 + q5) - self.zt * np.sin(q5) - self.ya
        self.g[4] = self.rr * np.cos(q1) - self.d * np.cos(q1 + q2) - self.zf * np.cos(q6 + q7) - self.u * np.sin(q7) - self.xa
        self.g[5] = self.rr * np.sin(q1) - self.d * np.sin(q1 + q2) - self.zf * np.sin(q6 + q7) + self.u * np.cos(q7) - self.ya

    def get_gqq(self, q, v):
        r"""
        Returns the second derivative of q applied to v.
        
        Parameters
        ----------
        q : dtype_u
            Unknown :math:`q`.
        v : dtype_u
            Unknown :math:`v`.
        """

        q1, q2, q3, q4, q5, q6, q7 = q[0], q[1], q[2], q[3], q[4], q[5], q[6]
        v1, v2, v3, v4, v5, v6, v7 = v[0], v[1], v[2], v[3], v[4], v[5], v[6]

        self.gqqv[0] = -self.rr * np.cos(q1) * v1 ** 2 + self.d * np.cos(q1 + q2) * (v1 + v2) ** 2 + self.ss * np.sin(q3) * v3 ** 2
        self.gqqv[1] = -self.rr * np.sin(q1) * v1 ** 2 + self.d * np.sin(q1 + q2) * (v1 + v2) ** 2 - self.ss * np.cos(q3) * v3 ** 2
        self.gqqv[2] = -self.rr * np.cos(q1) * v1 ** 2 + self.d * np.cos(q1 + q2) * (v1 + v2) ** 2 + self.e * np.sin(q4 + q5) * (v4 + v5) ** 2 + self.zt * np.cos(q5) * v5 ** 2
        self.gqqv[3] = -self.rr * np.sin(q1) * v1 ** 2 + self.d * np.sin(q1 + q2) * (v1 + v2) ** 2 - self.e * np.cos(q4 + q5) * (v4 + v5) ** 2 + self.zt * np.sin(q5) * v5 ** 2
        self.gqqv[4] = -self.rr * np.cos(q1) * v1 ** 2 + self.d * np.cos(q1 + q2) * (v1 + v2) ** 2 + self.zf * np.cos(q6 + q7) * (v6 + v7) ** 2 + self.u * np.sin(q7) * v7 ** 2
        self.gqqv[5] = -self.rr * np.sin(q1) * v1 ** 2 + self.d * np.sin(q1 + q2) * (v1 + v2) ** 2 + self.zf * np.sin(q6 + q7) * (v6 + v7) ** 2 - self.u * np.cos(q7) * v7 ** 2


    def getM(self, q):
        r"""
        Returns matrix :math:`M` for algebraic equationsin system of DAEs.

        Parameters
        ----------
        q : dtype_u
            Differential variables q.
        """

        # Shortcuts
        q1, q2, q3, q4, q5, q6, q7 = q[0], q[1], q[2], q[3], q[4], q[5], q[6]

        self.M[0, 0] = self.m1 * self.ra ** 2 + self.m2 * (self.rr ** 2 - 2 * self.da * self.rr * np.cos(q2) + self.da ** 2) + self.I1 + self.I2

        self.M[1, 0] = self.m2 * (self.da ** 2 - self.da * self.rr * np.cos(q2)) + self.I2
        self.M[0, 1] = self.M[1, 0]

        self.M[1, 1] = self.m2 * self.da ** 2 + self.I2

        self.M[2, 2] = self.m3 * (self.sa ** 2 + self.sb ** 2) + self.I3

        self.M[3, 3] = self.m4 * (self.e - self.ea) ** 2 + self.I4

        self.M[4, 3] = self.m4 * ((self.e - self.ea) ** 2 + self.zt * (self.e - self.ea) * np.sin(q4)) + self.I4
        self.M[3, 4] = self.M[4, 3]

        self.M[4, 4] = self.m4 * (self.zt ** 2 + 2 * self.zt * (self.e - self.ea) * np.sin(q4) + (self.e - self.ea) ** 2) + self.m5 * (self.ta ** 2 + self.tb ** 2) + self.I4 + self.I5

        self.M[5, 5] = self.m6 * (self.zf - self.fa) ** 2 + self.I6

        self.M[6, 5] = self.m6 * ((self.zf - self.fa) ** 2 - self.u * (self.zf - self.fa) * np.sin(q6)) + self.I6
        self.M[5, 6] = self.M[6, 5]

        self.M[6, 6] = self.m6 * ((self.zf - self.fa) ** 2 - 2 * self.u * (self.zf - self.fa) * np.sin(q6) + self.u ** 2) + self.m7 * (self.ua ** 2 + self.ub ** 2) + self.I6 + self.I7

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

        assert (
            t == 0.0 or t == 0.03
        ), f"u_exact only provides initial conditions for time 0.0 and reference solution for time 0.03 for q!"

        me = self.dtype_u(self.init)
        if t == 0.0:
            me.diff[0 : 7] = (
                -0.0617138900142764496358948458001,
                0,
                0.455279819163070380255912382449,
                0.222668390165885884674473185609,
                0.487364979543842550225598953530,
                -0.222668390165885884674473185609,
                1.23054744454982119249735015568,
            )  # q
            me.diff[7 : 14] = (0, 0, 0, 0, 0, 0, 0)  # v = q'
            me.alg[0 : 7] = (14222.4439199541138705911625887, -10666.8329399655854029433719415, 0, 0, 0, 0, 0)  # w = q''
            me.alg[7 : 13] = (98.56687039624108960576549821700, -6.12268834425566265503114393122, 0, 0, 0, 0)  # l

        elif t == 0.03:
            me.diff[0 : 7] = (
                0.1581077119629904e2,
                -0.1575637105984298e2,
                0.4082224013073101e-1,
                -0.5347301163226948,
                0.5244099658805304,
                0.5347301163226948,
                0.1048080741042263*10,
            )
        return me


class AndrewsSqueezingMechanismDAEConstrained(AndrewsSqueezingMechanismDAE):
    r"""
    For this class no quadrature is used for the algebraic constraints, i.e., system for algebraic constraints is solved directly.
    Note that in this class the index 2 formulation of Andrews' squeezer is used.
    """

    def __init__(self, index=3, nvars=27, newton_tol=1e-12, newton_maxiter=50, stop_at_maxiter=False, stop_at_nan=False):
        """Initialization routine"""
        super().__init__()
        self._makeAttributeAndRegister(
            'index', 'newton_tol', 'newton_maxiter', 'stop_at_maxiter', 'stop_at_nan', localVars=locals()
        )
        self.work_counters['newton'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        self.nq, self.nv, self.nw, self.nl = 7, 7, 7, 6

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
            The right-hand side of f (contains 27 components).
        """

        # Shortcuts
        v = u.diff[7 : 14]
        w = u.alg[0 : 7]

        f = self.dtype_f(self.init)
        f.diff[0 : 7] = v[:]
        f.diff[7 : 14] = w[:]

        f.alg[:] = self.algebraicConstraints(u, t)
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

        rhs_diff1, rhs_diff2 = rhs.diff[0 : 7], rhs.diff[7 : 14]

        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Shortcuts
            q, v = u.diff[0 : 7], u.diff[7 : 14]
            w = u.alg[0 : 7]

            h1 = q[:] - factor * v[:] - rhs_diff1[:]
            h2 = v[:] - factor * w[:] - rhs_diff2[:]
            f_alg = self.algebraicConstraints(u, t)[0 : 13]

            # Form the function h(u), such that the solution to the nonlinear problem is a root of g
            h = np.array([*h1, *h2, *f_alg])

            # If g is close to 0, then we are done
            res = np.linalg.norm(h, np.inf)
            if res < self.newton_tol:
                break

            # Assemble dh
            dh = self.update_Jacobian(factor)

            # Newton direction dx
            dx = np.linalg.solve(dh, h)

            # Newton update: u1 = u0 - g/dg
            u.diff[0 : 14] -= dx[0 : 14]
            u.alg[0 : 13] -= dx[14 : 27]

            # Increase iteration per one
            n += 1
            self.work_counters['newton']()
        # print(n, res)
        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)
        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]
        return me

    def update_Jacobian(self, factor):
        r"""
        Updates the Jacobian for the system to be solved by Newton. Note that the Jacobian of
        the right-hand side of the DAE system is approximated. Here, the derivatives of
        :math:`f`, :math:`M` and :math:`G` are neglected.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).

        Returns
        -------
        J : np.2darray
            Jacobian.
        """

        if self.index == 3:
            J = np.block(
                [
                    [np.eye(self.nq), -factor * np.eye(self.nv), np.zeros((self.nq, self.nw)), np.zeros((self.nq, self.nl))],
                    [np.zeros((self.nv, self.nq)), np.eye(self.nv), -factor * np.eye(self.nw), np.zeros((self.nv, self.nl))],
                    [np.zeros((self.nw, self.nq)), np.zeros((self.nw, self.nv)), self.M, self.G.T],
                    [self.G, np.zeros((self.nl, self.nv)), np.zeros((self.nl, self.nw)), np.zeros((self.nl, self.nl))],
                ]
            )
        elif self.index == 2:
            J = np.block(
                [
                    [np.eye(self.nq), -factor * np.eye(self.nv), np.zeros((self.nq, self.nw)), np.zeros((self.nq, self.nl))],
                    [np.zeros((self.nv, self.nq)), np.eye(self.nv), -factor * np.eye(self.nw), np.zeros((self.nv, self.nl))],
                    [np.zeros((self.nw, self.nq)), np.zeros((self.nw, self.nv)), self.M, self.G.T],
                    [np.zeros((self.nl, self.nq)), self.G, np.zeros((self.nl, self.nw)), np.zeros((self.nl, self.nl))],
                ]
            )
        elif self.index == 1:
            J = np.block(
                [
                    [np.eye(self.nq), -factor * np.eye(self.nv), np.zeros((self.nq, self.nw)), np.zeros((self.nq, self.nl))],
                    [np.zeros((self.nv, self.nq)), np.eye(self.nv), -factor * np.eye(self.nw), np.zeros((self.nv, self.nl))],
                    [np.zeros((self.nw, self.nq)), np.zeros((self.nw, self.nv)), self.M, self.G.T],
                    [np.zeros((self.nl, self.nq)), np.zeros((self.nl, self.nv)), self.G, np.zeros((self.nl, self.nl))],
                ]
            )
        else:
            raise NotImplementedError

        return J


class AndrewsSqueezingMechanismDAEEmbedded(AndrewsSqueezingMechanismDAEConstrained):
    r"""
    For this class the naively approach of embedded SDC is used.
    """

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

        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Shortcuts
            q, v = u.diff[: 7], u.diff[7 : 14]
            w = u.alg[: 7]

            h1 = q - factor * v - rhs.diff[: 7]
            h2 = v - factor * w - rhs.diff[7 : 14]
            h3 = -factor * self.algebraicConstraints(u, t)[:13] - rhs.alg[:13]

            # Form the function g(u), such that the solution to the nonlinear problem is a root of g
            h = np.array([*h1, *h2, *h3])

            # If g is close to 0, then we are done
            res = np.linalg.norm(h, np.inf)
            if res < self.newton_tol:
                break

            # # Assemble dh 
            dh = self.update_Jacobian(factor)

            # Newton direction dx
            dx = np.linalg.solve(dh, h)

            # Newton update: u1 = u0 - g/dg
            u.diff[0 : 14] -= dx[0 : 14]
            u.alg[0 : 13] -= dx[14 : 27]

            # Increase iteration per one
            n += 1
            self.work_counters['newton']()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)
        if n == self.newton_maxiter:
            msg = 'Newton did not converge after %i iterations, error is %s' % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        me = self.dtype_u(self.init)
        me[:] = u[:]
        return me

    def update_Jacobian(self, factor):
        r"""
        Updates the Jacobian for the system to be solved by Newton. Note that the Jacobian of
        the right-hand side of the DAE system is approximated. Here, the derivatives of
        :math:`f`, :math:`M` and :math:`G` are neglected. Also, the Jacobian matrix is different
        from the one of the parent class since spectral integration is applied to all equations here.

        Parameters
        ----------
        factor : float
            Abbrev. for the node-to-node stepsize (or any other factor required).
        G : np.2darray
            Jacobian of function :math:`g` of DAE system.
        M : np.2darray
            Matrix :math:`M` of DAE system.

        Returns
        -------
        J : np.2darray
            Jacobian.
        """

        if self.index == 3:
            J = np.block(
                [
                    [np.eye(self.nq), -factor * np.eye(self.nv), np.zeros((self.nq, self.nw)), np.zeros((self.nq, self.nl))],
                    [np.zeros((self.nv, self.nq)), np.eye(self.nv), -factor * np.eye(self.nw), np.zeros((self.nv, self.nl))],
                    [np.zeros((self.nw, self.nq)), np.zeros((self.nw, self.nv)), -factor * self.M, -factor * self.G.T],
                    [-factor * self.G, np.zeros((self.nl, self.nv)), np.zeros((self.nl, self.nw)), np.zeros((self.nl, self.nl))]
                ]
            )

        elif self.index == 2:
            J = np.block(
                [
                    [np.eye(self.nq), -factor * np.eye(self.nv), np.zeros((self.nq, self.nw)), np.zeros((self.nq, self.nl))],
                    [np.zeros((self.nv, self.nq)), np.eye(self.nv), -factor * np.eye(self.nw), np.zeros((self.nv, self.nl))],
                    [np.zeros((self.nw, self.nq)), np.zeros((self.nw, self.nv)), -factor * self.M, -factor * self.G.T],
                    [np.zeros((self.nl, self.nq)), -factor * self.G, np.zeros((self.nl, self.nw)), np.zeros((self.nl, self.nl))],
                ]
            )

        elif self.index == 1:
            J = np.block(
                [
                    [np.eye(self.nq), -factor * np.eye(self.nv), np.zeros((self.nq, self.nw)), np.zeros((self.nq, self.nl))],
                    [np.zeros((self.nv, self.nq)), np.eye(self.nv), -factor * np.eye(self.nw), np.zeros((self.nv, self.nl))],
                    [np.zeros((self.nw, self.nq)), np.zeros((self.nw, self.nv)), -factor * self.M, -factor * self.G.T],
                    [np.zeros((self.nl, self.nq)), np.zeros((self.nl, self.nv)), -factor * self.G, np.zeros((self.nl, self.nl))],
                ]
            )

        else:
            raise NotImplementedError

        return J
