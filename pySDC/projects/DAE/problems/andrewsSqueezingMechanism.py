import numpy as np
from pathlib import Path
from scipy.optimize import root

from pySDC.core.errors import ProblemError
from pySDC.core.hooks import Hooks
from pySDC.core.problem import WorkCounter
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.implementations.datatype_classes.mesh import mesh


# Problem specific hooks
class LogGlobalErrorPreIterMechanicalVars(Hooks):
    """Logs global error of Andrews' squeezer components after prediction."""

    def pre_iteration(self, step, level_number):
        r"""
        Default routine called before each iteration.

        Parameters
        ----------
        step : pySDC.core.step.Step
            Current step.
        level_number : pySDC.core.level.Level
            Current level number.
        """

        super().pre_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        upde = P.u_exact(step.time + step.dt)

        uend_ex = upde.flatten()
        uend = L.u[-1].flatten()

        e_global_position = abs(uend_ex[: 7] - uend[: 7])
        e_global_velocity = abs(uend_ex[7 : 14] - uend[7 : 14])
        e_global_acceleration = abs(uend_ex[14 : 21] - uend[14 : 21])
        e_global_lagrange = abs(uend_ex[21 :] - uend[21 :])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_position_pre_iteration",
            value=e_global_position,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_velocity_pre_iteration",
            value=e_global_velocity,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_acceleration_pre_iteration",
            value=e_global_acceleration,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_lagrange_pre_iteration",
            value=e_global_lagrange,
        )


class LogGlobalErrorPostIterMechanicalVars(Hooks):
    """Logs global error of position variables after iterations."""

    def post_iteration(self, step, level_number):
        super().post_iteration(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        upde = P.u_exact(step.time + step.dt)

        uend_ex = upde.flatten()
        uend = L.u[-1].flatten()

        e_global_position = abs(uend_ex[: 7] - uend[: 7])
        e_global_velocity = abs(uend_ex[7 : 14] - uend[7 : 14])
        e_global_acceleration = abs(uend_ex[14 : 21] - uend[14 : 21])
        e_global_lagrange = abs(uend_ex[21 :] - uend[21 :])

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_position_post_iteration",
            value=e_global_position,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_velocity_post_iteration",
            value=e_global_velocity,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_acceleration_post_iteration",
            value=e_global_acceleration,
        )

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type="e_global_lagrange_post_iteration",
            value=e_global_lagrange,
        )


def qend_ref_testset(t):
    """Returns solution of Andrews' problem for q-values at end of interval."""

    assert np.isclose(t, 0.03, atol=1e-14)

    return np.array([
        0.1581077119629904 * 1e2,
        -0.1575637105984298 * 1e2,
        0.4082224013073101 * 1e-1,
        -0.5347301163226948,
        0.5244099658805304,
        0.5347301163226948,
        0.1048080741042263 * 10,
    ])


class AndrewsSqueezingMechanismDAE(ProblemDAE):
    def __init__(
            self,
            nvars=14,
            newton_tol=1e-14,
            index=1,
            newton_maxiter=10,
            solver_type="hybr",
            stop_at_maxiter=False,
            stop_at_nan=False,
            verbose=False,
        ):
        """Initialization routine"""

        super().__init__(nvars=nvars, newton_tol=newton_tol)
        self._makeAttributeAndRegister(
            "newton_tol",
            "index",
            "newton_maxiter",
            "solver_type",
            "stop_at_maxiter",
            "stop_at_nan",
            "verbose",
            localVars=locals(),
        )

        self.work_counters["rhs"] = WorkCounter()
        self.work_counters[self.solver_type] = WorkCounter()

        self._allocate_internal_arrays()

        self._init_external_input()
        self._init_masses_and_inertias()
        self._init_geometrical_params()
        self._init_coordinates()

        self._load_reference_solution()

        self._init_nvars()

    def _allocate_internal_arrays(self):
        """Allocates memory for internal working arrays used during evaluation."""

        self.M = np.zeros((7, 7))
        self.G = np.zeros((6, 7))
        self.func = np.zeros(7)
        self.g = np.zeros(6)
        self.gqqv = np.zeros(6)

    def _init_external_input(self):
        """Initializes constant external forces or moments."""

        self.mom = 0.033

    def _init_masses_and_inertias(self):
        """Sets the attributes with values of masses and inertias."""

        self.m1 = 0.04325
        self.m2 = 0.00365
        self.m3 = 0.02373
        self.m4 = 0.00706
        self.m5 = 0.07050
        self.m6 = 0.00706
        self.m7 = 0.05498

        self.I1 = 2.194e-6
        self.I2 = 4.410e-7
        self.I3 = 5.255e-6
        self.I4 = 5.667e-7
        self.I5 = 1.169e-5
        self.I6 = 5.667e-7
        self.I7 = 1.912e-5

    def _init_geometrical_params(self):
        """Sets attributes with geometrical parameters."""

        self.c0 = 4530

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

    def _init_coordinates(self):
        """Sets attributes for coordinates."""

        self.xa = -0.06934
        self.ya = -0.00227
        self.xb = -0.03635
        self.yb = 0.03273
        self.xc = 0.014
        self.yc = 0.072

    def _init_nvars(self):
        """Initializes the number of the differential variables in problem."""

        self.nq, self.nv, self.nw, self.nl = 7, 7, 7, 6

    def _load_reference_solution(self):
        """Loads reference solution from stored files."""

        for path in [
            Path("/Users/lisa/Projects/Python/pySDC/pySDC/projects/DAE/data/"),
            Path("/beegfs/wimmer/pySDC/projects/DAE/data/")
        ]:
            if path.exists():
                path_to_data = path
                break
        else:
            raise FileNotFoundError("Could not locate data directory.")

        self.t_ref = np.load(path_to_data / "t_solve_andrews_constrainedDAE.npy")
        self.u_diff_ref = np.load(path_to_data / "u_diff_solve_andrews_constrainedDAE.npy")
        self.u_alg_ref = np.load(path_to_data / "u_alg_solve_andrews_constrainedDAE.npy")


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

        self.g[0] = (
            self.rr * np.cos(q1)
            - self.d * np.cos(q1 + q2)
            - self.ss * np.sin(q3)
            - self.xb
        )
        self.g[1] = (
            self.rr * np.sin(q1)
            - self.d * np.sin(q1 + q2)
            + self.ss * np.cos(q3)
            - self.yb
        )
        self.g[2] = (
            self.rr * np.cos(q1)
            - self.d * np.cos(q1 + q2)
            - self.e * np.sin(q4 + q5)
            - self.zt * np.cos(q5)
            - self.xa
        )
        self.g[3] = (
            self.rr * np.sin(q1)
            - self.d * np.sin(q1 + q2)
            + self.e * np.cos(q4 + q5)
            - self.zt * np.sin(q5)
            - self.ya
        )
        self.g[4] = (
            self.rr * np.cos(q1)
            - self.d * np.cos(q1 + q2)
            - self.zf * np.cos(q6 + q7)
            - self.u * np.sin(q7)
            - self.xa
        )
        self.g[5] = (
            self.rr * np.sin(q1)
            - self.d * np.sin(q1 + q2)
            - self.zf * np.sin(q6 + q7)
            + self.u * np.cos(q7)
            - self.ya
        )

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

        self.gqqv[0] = (
            - self.rr * np.cos(q1) * v1 ** 2
            + self.d * np.cos(q1 + q2) * (v1 + v2) ** 2
            + self.ss * np.sin(q3) * v3 ** 2
        )
        self.gqqv[1] = (
            - self.rr * np.sin(q1) * v1 ** 2
            + self.d * np.sin(q1 + q2) * (v1 + v2) ** 2
            - self.ss * np.cos(q3) * v3 ** 2
        )
        self.gqqv[2] = (
            - self.rr * np.cos(q1) * v1 ** 2
            + self.d * np.cos(q1 + q2) * (v1 + v2) ** 2
            + self.e * np.sin(q4 + q5) * (v4 + v5) ** 2
            + self.zt * np.cos(q5) * v5 ** 2
        )
        self.gqqv[3] = (
            - self.rr * np.sin(q1) * v1 ** 2
            + self.d * np.sin(q1 + q2) * (v1 + v2) ** 2
            - self.e * np.cos(q4 + q5) * (v4 + v5) ** 2
            + self.zt * np.sin(q5) * v5 ** 2
        )
        self.gqqv[4] = (
            - self.rr * np.cos(q1) * v1 ** 2
            + self.d * np.cos(q1 + q2) * (v1 + v2) ** 2
            + self.zf * np.cos(q6 + q7) * (v6 + v7) ** 2
            + self.u * np.sin(q7) * v7 ** 2
        )
        self.gqqv[5] = (
            - self.rr * np.sin(q1) * v1 ** 2
            + self.d * np.sin(q1 + q2) * (v1 + v2) ** 2
            + self.zf * np.sin(q6 + q7) * (v6 + v7) ** 2
            - self.u * np.cos(q7) * v7 ** 2
        )


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

        self.M[0, 0] = (
            self.m1 * self.ra ** 2
            + self.m2 * (self.rr ** 2 - 2 * self.da * self.rr * np.cos(q2) + self.da ** 2)
            + self.I1 + self.I2
        )

        self.M[1, 0] = self.m2 * (self.da ** 2 - self.da * self.rr * np.cos(q2)) + self.I2
        self.M[0, 1] = self.M[1, 0]

        self.M[1, 1] = self.m2 * self.da ** 2 + self.I2

        self.M[2, 2] = self.m3 * (self.sa ** 2 + self.sb ** 2) + self.I3

        self.M[3, 3] = self.m4 * (self.e - self.ea) ** 2 + self.I4

        self.M[4, 3] = (
            self.m4 * (
                (self.e - self.ea) ** 2 + self.zt * (self.e - self.ea) * np.sin(q4)
            ) + self.I4
        )
        self.M[3, 4] = self.M[4, 3]

        self.M[4, 4] = (
            self.m4 * (
                self.zt ** 2 + 2 * self.zt * (self.e - self.ea) * np.sin(q4) + (self.e - self.ea) ** 2
            )
            + self.m5 * (self.ta ** 2 + self.tb ** 2)
            + self.I4
            + self.I5
        )

        self.M[5, 5] = self.m6 * (self.zf - self.fa) ** 2 + self.I6

        self.M[6, 5] = (
            self.m6 * (
                (self.zf - self.fa) ** 2 - self.u * (self.zf - self.fa) * np.sin(q6)
            )
            + self.I6
        )
        self.M[5, 6] = self.M[6, 5]

        self.M[6, 6] = (
            self.m6 * (
                (self.zf - self.fa) ** 2 - 2 * self.u * (self.zf - self.fa) * np.sin(q6) + self.u ** 2
            )
            + self.m7 * (self.ua ** 2 + self.ub ** 2)
            + self.I6
            + self.I7
        )

    def solve_system(self, impl_sys, rhs, factor, u0, t):
        r"""
        Dispatcher that selects the appropriate solver backend based on ``self.solver_type``.
        Possible solvers are:

        - "hybr": solves the system using SciPy's root finder with hybrid methods
        - "newton": solves the system via a custom Newton-Raphson method (only available
           for SDC-C and SDC-E)

        Parameters
        ----------
        impl_sys : callable
            The function representing the fully implicit system (required for 'hybr').
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.

        Raises
        ------
        ProblemError
            If ``self.solver_type`` is not recognized.
        """

        if self.solver_type == "hybr":
            return self.solve_with_hybr(rhs, factor, u0, t, impl_sys)
        elif self.solver_type == "newton":
            return self.solve_with_newton(rhs, factor, u0, t, impl_sys)
        else:
            raise ProblemError(f"Unknown solver_type: {self.solver_type}")

    def dg(self, factor):
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
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        factor * self.M,
                        factor * self.G.T,
                    ],
                    [
                        factor * self.G,
                        np.zeros((self.nl, self.nv)),
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ]
                ]
            )

        elif self.index == 2:
            J = np.block(
                [
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        factor * self.M,
                        factor * self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        factor * self.G,
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )

        elif self.index == 1:
            J = np.block(
                [
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        factor * self.M,
                        factor * self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        np.zeros((self.nl, self.nv)),
                        factor * self.G,
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )
        else:
            raise NotImplementedError

        return J

    def solve_with_hybr(self, u_approx, factor, u0, t, impl_sys=None):
        r"""
        Root solver for the nonlinear system using SciPy's ``optimize.root`` with
        'hybr' method. The method calls the one from parent class ``ProblemDAE``.

        Parameters
        ----------
        u_approx : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Approximation of u in the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.
        impl_sys : callable, optional
            Implicit system needed for the routine; is defined in
            ``fullyImplicitDAE`` or ``semiImplicitDAE`` sweeper.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """
        return super().solve_system(impl_sys, u_approx, factor, u0, t)
    
    def solve_with_newton(self, rhs, factor, u0, t, impl_sys):
        r"""
        Placeholder for solve with Newton's method that can be written
        some time.

        Parameters
        ----------
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.

        Raises
        ------
        NotImplementedError
        """

        u = self.dtype_u(u0)

        def impl_sys_numpy(u_me, **kwargs):
            sys = impl_sys(u_me, self, factor, rhs, t, **kwargs)
            sys_numpy = np.array([*sys.diff[:14], *sys.alg[:13]])
            return sys_numpy

        n = 0
        res = 99
        while n < self.newton_maxiter:
            g = impl_sys_numpy(u)

            # If h is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Assemble dh
            dg = self.dg(factor)

            # Newton direction dx
            dx = np.linalg.solve(dg, g)

            # Newton update: u1 = u0 - g/dg
            u.diff[: 14] -= dx[: 14]
            u.alg[: 13] -= dx[14 : 27]

            # Increase iteration per one
            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter and self.verbose:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        solution = self.dtype_u(self.init)
        solution[:] = u[:]
        return solution

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

        elif t > 0.0:
            i = np.searchsorted(self.t_ref, t)

            if i < len(self.t_ref) and np.isclose(self.t_ref[i], t, atol=1e-14):
                ind = i
            elif i > 0 and np.isclose(self.t_ref[i-1], t, atol=1e-14):
                ind = i - 1
            else:
                print("No suitable entry found.")

            u_ref_diff = self.u_diff_ref[ind, :]
            u_ref_alg = self.u_alg_ref[ind, :]

            me.diff[0 : 7] = u_ref_diff[0 : 7]  # q
            me.diff[7 : 14] = u_ref_diff[7 : 14]  # v

            me.alg[0 : 7] = u_ref_alg[0 : 7]  # w
            me.alg[7 : 13] = u_ref_alg[7 : 13]  # l

        return me

    def du_exact(self, t):
        r"""
        Routine for the initial condition of derivative of exact solution
        at time :math:`t`. Required for Runge-Kutta methods.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        du_ex : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Derivative of exact solution.
        """

        assert t == 0.0, f"ERROR: Only initial condition at time 0.0 available!"

        u0 = self.u_exact(t)
        q, v = u0.diff[: 7], u0.diff[7 : 14]
        w = u0.alg[: 7]

        du_ex = self.dtype_f(self.init)
        du_ex.diff[0 : 7] = v[:]
        du_ex.diff[7 : 14] = w[:]
        du_ex.alg[:] = self.algebraicConstraints(u0, t)[:]
        return du_ex


class AndrewsSqueezingMechanismDAE_Radau(AndrewsSqueezingMechanismDAE, ProblemDAE):
    dtype_u = mesh
    dtype_f = mesh

    def __init__(
            self,
            nvars=27,
            newton_tol=5e-12,
            index=1,
            newton_maxiter=20,
            solver_type="hybr",
            stop_at_maxiter=False,
            stop_at_nan=False,
            verbose=False,
        ):
        """Initialization routine"""

        ProblemDAE.__init__(self, nvars=nvars, newton_tol=newton_tol)

        self._makeAttributeAndRegister(
            "newton_tol",
            "nvars",
            "index",
            "newton_maxiter",
            "solver_type",
            "stop_at_maxiter",
            "stop_at_nan",
            "verbose",
            localVars=locals(),
        )

        self.work_counters["rhs"] = WorkCounter()
        self.work_counters[self.solver_type] = WorkCounter()

        AndrewsSqueezingMechanismDAE._allocate_internal_arrays(self)

        AndrewsSqueezingMechanismDAE._init_external_input(self)
        AndrewsSqueezingMechanismDAE._init_masses_and_inertias(self)
        AndrewsSqueezingMechanismDAE._init_geometrical_params(self)
        AndrewsSqueezingMechanismDAE._init_coordinates(self)
        AndrewsSqueezingMechanismDAE._init_nvars(self)

        self.I0 = np.zeros((self.nvars, self.nvars))
        self.I0[: self.nq, : self.nq] = np.eye(self.nq)
        self.I0[self.nq : self.nq + self.nv, self.nq : self.nq + self.nv] = np.eye(self.nv)

        AndrewsSqueezingMechanismDAE._load_reference_solution(self)

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
        q, v = u[: 7], u[7 : 14]
        w = u[14 : 21]

        dq, dv = du[: 7], du[7 : 14]

        # Get matrices and functions for algebraic part of right-hand side
        self.getM(q)
        self.get_func(q, v)
        self.getG(q)

        f = self.dtype_f(self.init)
        f[: 7] = dq[:] - v[:]
        f[7 : 14] = dv[:] - w[:]

        f[14 :] = self.algebraicConstraints(u, t)[:]

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
        q, v = u[: 7], u[7 : 14]
        w, l = u[14 : 21], u[21 :]

        # Get matrices and functions for algebraic part of right-hand side
        self.getM(q)
        self.get_func(q, v)
        self.getG(q)

        f = self.dtype_f(self.init)

        f[14 : 21] = self.M.dot(w) - self.func + self.G.T.dot(l)
        if self.index == 3:
            self.get_g(q)

            f[21 :] = self.g

        elif self.index == 2:
            f[21 :] = self.G.dot(v)

        elif self.index == 1:
            self.get_gqq(q, v)

            f[21 :] = self.gqqv + self.G.dot(w)
        else:
            raise NotImplementedError

        return f[14 :]

    def dg(self, dt, M, Qmat):
        r"""
        Updates the Jacobian for the system to be solved by Newton. Note that the Jacobian of
        the right-hand side of the DAE system is approximated. Here, the derivatives of
        :math:`f`, :math:`M` and :math:`G` are neglected.

        Parameters
        ----------
        dt : float
            Time step size.
        Qmat : np.2darray
            Spectral integration matrix

        Returns
        -------
        J : np.2darray
            Jacobian.
        """

        if self.index == 3:
            J_approx = np.block(
                [
                    [
                        np.eye(self.nq),
                        np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        -self.M,
                        -self.G.T,
                    ],
                    [
                        -self.G,
                        np.zeros((self.nl, self.nv)),
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ]
                ]
            )

        elif self.index == 2:
            J_approx = np.block(
                [
                    [
                        np.eye(self.nq),
                        np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        -self.M,
                        -self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        -self.G,
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )

        elif self.index == 1:
            J_approx = np.block(
                [
                    [
                        np.zeros((self.nq, self.nq)),
                        np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.zeros((self.nv, self.nv)),
                        np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        -self.M,
                        -self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        np.zeros((self.nl, self.nv)),
                        -self.G,
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )
        else:
            raise NotImplementedError

        J = np.kron(np.identity(M), self.I0) - dt * np.kron(Qmat[1 :, 1 :], J_approx)

        return J

    def solve_collocation_system(self, F, f_init, t, dt, M, Qmat, sweep, u0_full):
        """Solves the collocation system for Radau solver."""

        def impl_sys(du, **kwargs):
            return F(du, t, dt, M, self, Qmat, sweep, u0_full, **kwargs)

        du = f_init.copy()

        n = 0
        res = 99
        while n < self.newton_maxiter:
            g = impl_sys(du)

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Assemble dg
            dg = self.dg(dt, M, Qmat)

            # # Newton direction dx
            dx = np.linalg.solve(dg, g)

            # Newton update: u1 = u0 - g/dg
            dx_matrix = dx.reshape((M, self.nvars))
            du_reshape = [self.dtype_f(du_m) - dx_m for du_m, dx_m in zip(du, dx_matrix)]

            du = du_reshape.copy()

            # Increase iteration per one
            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter and self.verbose:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        return du

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
        if t == 0.0:
            me[: 7] = (
                -0.0617138900142764496358948458001,
                0,
                0.455279819163070380255912382449,
                0.222668390165885884674473185609,
                0.487364979543842550225598953530,
                -0.222668390165885884674473185609,
                1.23054744454982119249735015568,
            )  # q
            me[7 : 14] = (0, 0, 0, 0, 0, 0, 0)  # v = q'
            me[14 : 21] = (14222.4439199541138705911625887, -10666.8329399655854029433719415, 0, 0, 0, 0, 0)  # w = q''
            me[21 :] = (98.56687039624108960576549821700, -6.12268834425566265503114393122, 0, 0, 0, 0)  # l

        elif t > 0.0:
            i = np.searchsorted(self.t_ref, t)

            if i < len(self.t_ref) and np.isclose(self.t_ref[i], t, atol=1e-14):
                ind = i
            elif i > 0 and np.isclose(self.t_ref[i-1], t, atol=1e-14):
                ind = i - 1
            else:
                print("No suitable entry found.")

            u_ref_diff = self.u_diff_ref[ind, :]
            u_ref_alg = self.u_alg_ref[ind, :]

            me[: 7] = u_ref_diff[0 : 7]  # q
            me[7 : 14] = u_ref_diff[7 : 14]  # v

            me[14 : 21] = u_ref_alg[0 : 7]  # w
            me[21 :] = u_ref_alg[7 : 13]  # l

        return me

    def du_exact(self, t):
        r"""
        Routine for the initial condition of derivative of exact solution
        at time :math:`t`. Required for Runge-Kutta methods.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        du_ex : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Derivative of exact solution.
        """

        assert t == 0.0, f"ERROR: Only initial condition at time 0.0 available!"

        u0 = self.u_exact(t)
        v, w = u0[7 : 14], u0[14 : 21]

        du_ex = self.dtype_f(self.init)
        du_ex[: 7] = v[:]
        du_ex[7 : 14] = w[:]
        du_ex[14 :] = self.algebraicConstraints(u0, t)[:]
        return du_ex


class SemiImplicitAndrewsSqueezingMechanismDAE(AndrewsSqueezingMechanismDAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dg(self, factor):
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
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        self.M,
                        self.G.T,
                    ],
                    [
                        self.G,
                        np.zeros((self.nl, self.nv)),
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )
        elif self.index == 2:
            J = np.block(
                [
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        self.M,
                        self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        self.G,
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )
        elif self.index == 1:
            J = np.block(
                [
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -np.eye(self.nw), 
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        self.M,
                        self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        np.zeros((self.nl, self.nv)),
                        self.G,
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )
        else:
            raise NotImplementedError

        return J


class AndrewsSqueezingMechanismDAEConstrained(AndrewsSqueezingMechanismDAE):
    """Constrained formulation where only the differential equations are integrated numerically"""

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
        self.work_counters["rhs"]()
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Wrapper for the base class solver interface with omitted implicit system.

        Parameters
        ----------
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the nonlinear system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time point.

        Returns
        -------
        me : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the nonlinear system.
        """

        return super().solve_system(None, rhs, factor, u0, t)

    def solve_with_newton(self, rhs, factor, u0, t, impl_sys=None):
        r"""
        Newton's method to solve the nonlinear system.

        Parameters
        ----------
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        u = self.dtype_u(u0)

        rhs_diff1, rhs_diff2 = rhs.diff[0 : 7], rhs.diff[7 : 14]

        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Shortcuts
            q, v = u.diff[0 : 7], u.diff[7 : 14]
            w = u.alg[0 : 7]

            g1 = q[:] - factor * v[:] - rhs_diff1[:]
            g2 = v[:] - factor * w[:] - rhs_diff2[:]
            f_alg = self.algebraicConstraints(u, t)[: 13]

            # Form the function h(u), such that the solution to the nonlinear problem is a root of h
            g = np.array([*g1, *g2, *f_alg])

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # Assemble dh
            dg = self.dg(factor)

            # Newton direction dx
            dx = np.linalg.solve(dg, g)

            # Newton update: u1 = u0 - g/dg
            u.diff[: 14] -= dx[: 14]
            u.alg[: 13] -= dx[14 : 27]

            # Increase iteration per one
            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter and self.verbose:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        solution = self.dtype_u(self.init)
        solution[:] = u[:]
        return solution

    def solve_with_hybr(self, rhs, factor, u0, t, impl_sys=None):
        r"""
        Root solver for the nonlinear system using SciPy's ``optimize.root`` with
        'hybr' method. The function to be find the root for is defined in this
        method.

        Parameters
        ----------
        u_approx : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Approximation of u in the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.
        impl_sys : callable, optional
            Implicit system needed for the routine; is defined in
            ``fullyImplicitDAE`` or ``semiImplicitDAE`` sweeper.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        solution = self.dtype_u(self.init)

        rhs_diff1, rhs_diff2 = rhs.diff[0 : 7], rhs.diff[7 : 14]

        # Form the function, such that the solution to the nonlinear problem is a root of it
        def andrews_dae(u):
            q, v = u[: 7], u[7 : 14]
            w, l = u[14 : 21], u[21 :]

            # Get matrices and functions for algebraic part of right-hand side
            self.getM(q)
            self.get_func(q, v)
            self.getG(q)

            f1 = q[:] - factor * v[:] - rhs_diff1[:]
            f2 = v[:] - factor * w[:] - rhs_diff2[:]
            f3 = self.M.dot(w) - self.func + self.G.T.dot(l)
            if self.index == 3:
                self.get_g(q)

                f4 = self.g

            elif self.index == 2:
                f4 = self.G.dot(v)

            elif self.index == 1:
                self.get_gqq(q, v)

                f4 = self.gqqv + self.G.dot(w)
            else:
                raise NotImplementedError

            return np.array([*f1, *f2, *f3, *f4])

        q0, v0 = u0.diff[0 : 7], u0.diff[7 : 14]
        w0, lamb0 = u0.alg[0 : 7], u0.alg[7 : 13]
        u0_vec = np.array([*q0, *v0, *w0, *lamb0])

        opt = root(
            andrews_dae,
            u0_vec,
            method=self.solver_type,
            tol=self.newton_tol,
        )

        solution = self.dtype_u(self.init)
        solution.diff[: 14] = opt.x[: 14]
        solution.alg[: 13] = opt.x[14 :]
        self.work_counters["hybr"].niter += opt.nfev
        return solution

    def dg(self, factor):
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
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        self.M,
                        self.G.T,
                    ],
                    [
                        self.G,
                        np.zeros((self.nl, self.nv)),
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )
        elif self.index == 2:
            J = np.block(
                [
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        self.M,
                        self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        self.G,
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )
        elif self.index == 1:
            J = np.block(
                [
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        self.M,
                        self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        np.zeros((self.nl, self.nv)),
                        self.G,
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )
        else:
            raise NotImplementedError

        return J


class AndrewsSqueezingMechanismDAEEmbedded(AndrewsSqueezingMechanismDAEConstrained):
    """Problem class for an embedded method where only the algebraic constraints are enforced"""

    def solve_with_newton(self, rhs, factor, u0, t, impl_sys):
        r"""
        Newton's method to solve the nonlinear system.

        Parameters
        ----------
        rhs : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Right-hand side of the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        u = self.dtype_u(u0)

        n = 0
        res = 99
        while n < self.newton_maxiter:
            # Shortcuts
            q, v = u.diff[: 7], u.diff[7 : 14]
            w = u.alg[: 7]

            g1 = q - factor * v - rhs.diff[: 7]
            g2 = v - factor * w - rhs.diff[7 : 14]
            g3 = -factor * self.algebraicConstraints(u, t)[:13] - rhs.alg[:13]

            # Form the function h(u), such that the solution to the nonlinear problem is a root of h
            g = np.array([*g1, *g2, *g3])

            # If g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)
            if res < self.newton_tol:
                break

            # # Assemble dh 
            dg = self.dg(factor)

            # Newton direction dx
            dx = np.linalg.solve(dg, g)

            # Newton update: u1 = u0 - g/dg
            u.diff[0 : 14] -= dx[0 : 14]
            u.alg[0 : 13] -= dx[14 : 27]

            # Increase iteration per one
            n += 1
            self.work_counters["newton"]()

        if np.isnan(res) and self.stop_at_nan:
            raise ProblemError("Newton got nan after %i iterations, aborting..." % n)
        elif np.isnan(res):
            self.logger.warning("Newton got nan after %i iterations..." % n)
        if n == self.newton_maxiter and self.verbose:
            msg = "Newton did not converge after %i iterations, error is %s" % (n, res)
            if self.stop_at_maxiter:
                raise ProblemError(msg)
            else:
                self.logger.warning(msg)

        solution = self.dtype_u(self.init)
        solution[:] = u[:]
        return solution
    
    def solve_with_hybr(self, rhs, factor, u0, t, impl_sys=None):
        r"""
        Root solver for the nonlinear system using SciPy's ``optimize.root`` with
        'hybr' method. The function to be find the root for is defined in this
        method.

        Parameters
        ----------
        u_approx : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Approximation of u in the implicit system to be solved.
        factor : float
            Step size-related factor (e.g., node-to-node step size).
        u0 : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Initial guess for the solution.
        t : float
            Current time.
        impl_sys : callable, optional
            Implicit system needed for the routine; is defined in
            ``fullyImplicitDAE`` or ``semiImplicitDAE`` sweeper.

        Returns
        -------
        solution : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Numerical solution of the linear system.
        """

        solution = self.dtype_u(self.init)

        # Form the function, such that the solution to the nonlinear problem is a root of it
        def func(u):
            q, v = u[: 7], u[7 : 14]
            w, l = u[14 : 21], u[21 :]

            # Get matrices and functions for algebraic part of right-hand side
            self.getM(q)
            self.get_func(q, v)
            self.getG(q)

            f1 = q - factor * v - rhs.diff[: 7]
            f2 = v - factor * w - rhs.diff[7 : 14]
            f3 = -factor * (self.M.dot(w) - self.func + self.G.T.dot(l)) - rhs.alg[: 7]
            if self.index == 3:
                self.get_g(q)

                f4 = -factor * self.g - rhs.alg[7 : 13]

            elif self.index == 2:
                f4 = -factor * self.G.dot(v) - rhs.alg[7 : 13]

            elif self.index == 1:
                self.get_gqq(q, v)

                f4 = -factor * (self.gqqv + self.G.dot(w)) - rhs.alg[7 : 13]
            else:
                raise NotImplementedError

            return np.array([*f1, *f2, *f3, *f4])

        u0_vec = np.array([*u0.diff[: 14], *u0.alg[: 13]])

        opt = root(
            func,
            u0_vec,
            method=self.solver_type,
            tol=self.newton_tol,
        )

        solution = self.dtype_u(self.init)
        solution.diff[: 14] = opt.x[: 14]
        solution.alg[: 13] = opt.x[14 :]
        self.work_counters["hybr"].niter += opt.nfev
        return solution

    def dg(self, factor):
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
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        -factor * self.M,
                        -factor * self.G.T,
                    ],
                    [
                        -factor * self.G,
                        np.zeros((self.nl, self.nv)),
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ]
                ]
            )

        elif self.index == 2:
            J = np.block(
                [
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        -factor * self.M,
                        -factor * self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        -factor * self.G,
                        np.zeros((self.nl, self.nw)),
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )

        elif self.index == 1:
            J = np.block(
                [
                    [
                        np.eye(self.nq),
                        -factor * np.eye(self.nv),
                        np.zeros((self.nq, self.nw)),
                        np.zeros((self.nq, self.nl)),
                    ],
                    [
                        np.zeros((self.nv, self.nq)),
                        np.eye(self.nv),
                        -factor * np.eye(self.nw),
                        np.zeros((self.nv, self.nl)),
                    ],
                    [
                        np.zeros((self.nw, self.nq)),
                        np.zeros((self.nw, self.nv)),
                        -factor * self.M,
                        -factor * self.G.T,
                    ],
                    [
                        np.zeros((self.nl, self.nq)),
                        np.zeros((self.nl, self.nv)),
                        -factor * self.G,
                        np.zeros((self.nl, self.nl)),
                    ],
                ]
            )

        else:
            raise NotImplementedError

        return J
