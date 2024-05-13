import numpy as np

from pySDC.implementations.sweeper_classes.Runge_Kutta import (
    ButcherTableau,
    RungeKutta,
    BackwardEuler,
    ImplicitMidpointMethod,
    CrankNicholson,
)


class RungeKuttaDAE(RungeKutta):
    r"""
    Custom sweeper class to implement Runge-Kutta (RK) methods for general differential-algebraic equations (DAEs)
    of the form

    .. math::
        0 = F(u, u', t).
    
    RK methods for general DAEs have the form

    .. math::
        0 = F(u_0 + \Delta t \sum_{j=1}^M a_{i,j} U_j, U_m),

    .. math::
        u_M = u_0 + \Delta t \sum_{j=1}^M b_j U_j.

    In pySDC, RK methods are implemented in the way that the coefficient matrix :math:`A` in the Butcher
    tableau is a lower triangular matrix so that the stages are updated node-by-node. This class therefore only supports
    RK methods with lower triangular matrices in the tableau.

    Parameters
    ----------
    params : dict
        Parameters passed to the sweeper.

    Attributes
    ----------
    du_init : dtype_f
        Stores the initial condition for each step.

    Note
    ----
    When using a RK sweeper to simulate a problem make sure the DAE problem class has a ``du_exact`` method since RK methods needs an initial
    condition for :math:`u'(t_0)` as well.

    In order to implement a new RK method for DAEs a new tableau can be added in ``pySDC.implementations.sweeper_classes.Runge_Kutta.py``.
    For example, a new method called ``newRungeKuttaMethod`` with nodes :math:`c=(c_1, c_2, c_3)`, weights :math:`b=(b_1, b_2, b_3)` and
    coefficient matrix

    ..math::
        \begin{eqnarray}
            A = \begin{pmatrix}
                a_{11} & 0 & 0 \\
                a_{21} & a_{22} & 0 \\
                a_{31} & a_{32} & & 0 \\
            \end{pmatrix}
        \end{eqnarray}

    can be implemented as follows:

    >>> class newRungeKuttaMethod(RungeKutta):
    >>>     nodes = np.array([c1, c2, c3])
    >>>     weights = np.array([b1, b2, b3])
    >>>     matrix = np.zeros((3, 3))
    >>>     matrix[0, 0] = a11
    >>>     matrix[1, :2] = [a21, a22]
    >>>     matrix[2, :] = [a31, a32, a33]
    >>>     ButcherTabelauClass = ButcherTableau

    The new class ``newRungeKuttaMethodDAE`` can then be used by defining the DAE class inheriting from this base class and class containing
    the Butcher tableau:

    >>> class newRungeKuttaMethodDAE(RungeKuttaDAE, newRungeKuttaMethod):
    >>>     pass

    More details can be found [here](https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/sweeper_classes/Runge_Kutta.py).
    """

    def __init__(self, params):
        super().__init__(params)
        self.du_init = None

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep.
        """

        # get current level and problem
        lvl = self.level
        prob = lvl.prob

        du_init = self.du_init[:] if self.du_init is not None else prob.du_exact(lvl.time)[:]
        lvl.f[0] = prob.dtype_f(du_init)
        for m in range(1, self.coll.num_nodes + 1):
            lvl.u[m] = prob.dtype_u(init=prob.init, val=0.0)
            lvl.f[m] = prob.dtype_f(init=prob.init, val=0.0)

        lvl.status.unlocked = True
        lvl.status.updated = True

    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        # get current level and problem description
        lvl = self.level
        prob = lvl.prob

        # only if the level has been touched before
        assert lvl.status.unlocked
        assert lvl.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        M = self.coll.num_nodes
        for m in range(M):
            u_approx = prob.dtype_u(lvl.u[0])
            for j in range(1, m + 1):
                u_approx += lvl.dt * self.QI[m + 1, j] * lvl.f[j][:]

            def F(du):
                r"""
                This function builds the implicit system to be solved for a DAE of the form

                .. math::
                    0 = F(u, u', t)

                Applying a RK method yields the (non)-linear system to be solved

                .. math::
                    0 = F(u_0 + \Delta t \sum_{j=1}^m a_{ij} U_j, U_m, \tau_m),

                which is solved for the derivative of u.

                Parameters
                ----------
                du : dtype_u
                    Unknowns of the system (derivative of solution u).

                Returns
                -------
                sys : dtype_f
                    System to be solved.
                """
                du_mesh = prob.dtype_f(du)

                local_u_approx = prob.dtype_u(u_approx)

                # defines the "implicit" factor, note that for explicit RK the diagonal element is zero
                local_u_approx += lvl.dt * self.QI[m + 1, m + 1] * du_mesh

                sys = prob.eval_f(local_u_approx, du_mesh, lvl.time + lvl.dt * self.coll.nodes[m + 1])
                return sys

            finit = prob.dtype_f(lvl.f[m])
            lvl.f[m + 1][:] = prob.solve_system(F, finit.flatten(), lvl.time + lvl.dt * self.coll.nodes[m + 1])

        # Update numerical solution - update value only at last node
        lvl.u[-1][:] = lvl.u[0]
        for j in range(1, M + 1):
            lvl.u[-1][:] += lvl.dt * self.coll.Qmat[-1, j] * lvl.f[j][:]

        self.du_init = prob.dtype_f(lvl.f[-1])

        lvl.status.updated = True

        return None


# class RungeKuttaIMEXDAE(RungeKuttaDAE):
#     r"""
#     This is an IMEX base class for implementing Runge-Kutta methods for DAEs, where only the differential
#     variables will be integrated. This is useful for DAEs of semi-explicit form.
#     """

#     def integrate(self):
#         """
#         Integrates the right-hand side

#         Returns:
#             list of dtype_u: containing the integral as values
#         """

#         # get current level and problem
#         L = self.level
#         P = L.prob

#         me = []

#         # integrate RHS over all collocation nodes
#         for m in range(1, self.coll.num_nodes + 1):
#             # new instance of dtype_u, initialize values with 0
#             me.append(P.dtype_u(P.init, val=0.0))
#             for j in range(1, self.coll.num_nodes + 1):
#                 me[-1][: P.diff_nvars] += L.dt * self.coll.Qmat[m, j] * L.f[j][: P.diff_nvars]
#                 me[-1][P.diff_nvars :] += L.u[j][P.diff_nvars :]
#                 print()

#         return me

#     def update_nodes(self):
#         r"""
#         Updates the values of solution ``u`` and their gradient stored in ``f``.
#         """

#         # get current level and problem description
#         L = self.level
#         P = L.prob

#         # only if the level has been touched before
#         assert L.status.unlocked
#         assert L.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

#         # get number of collocation nodes for easier access
#         M = self.coll.num_nodes
#         u0 = L.u[0]
#         print('M:', M)
#         for m in range(0, M):
#             u_approx = P.dtype_u(u0)  # indices correct here?
#             for j in range(1, m + 1):
#                 u_approx += L.dt * self.QI[m + 1, j] * L.f[j]
#             u_approx[P.diff_nvars :] = L.u[m + 1][P.diff_nvars :]
#             def implSystem(unknowns):
#                 """
#                 Build implicit system to solve in order to find the unknowns for the derivative
#                 of u.

#                 Parameters
#                 ----------
#                 unknowns : dtype_u
#                     Unknowns of the system.

#                 Returns
#                 -------
#                 sys : dtype_f
#                     System to be solved.
#                 """
#                 unknowns_mesh = P.dtype_f(P.init)
#                 unknowns_mesh[:] = unknowns

#                 local_u_approx = u_approx

#                 # defines the "implicit" factor, note that for explicit RK the diagonal element is zero
#                 local_u_approx += L.dt * self.QI[m + 1, m + 1] * unknowns_mesh
#                 local_u_approx[P.diff_nvars :] = unknowns_mesh[P.diff_nvars :]
#                 sys = P.eval_f(local_u_approx, unknowns_mesh[: P.diff_nvars], L.time + L.dt * self.coll.nodes[m])
#                 return sys

#             # implicit solve with prefactor stemming from the diagonal of Qd, use previous stage as initial guess
#             U0_diff, p0_alg = np.array(L.f[m][: P.diff_nvars]), np.array(L.u[m][P.diff_nvars :])
#             du0 = np.concatenate((U0_diff, p0_alg))
#             du_new = P.solve_system(implSystem, du0, L.time + L.dt * self.coll.nodes[m])

#             L.f[m + 1][: P.diff_nvars] = du_new[: P.diff_nvars]
#             L.u[m + 1][P.diff_nvars :] = du_new[P.diff_nvars :]

#         # Update numerical solution
#         integral = self.integrate()
#         for k in range(M):
#             L.u[k + 1][: P.diff_nvars] = L.u[0][: P.diff_nvars] + integral[k][: P.diff_nvars]

#         # store value at last node as initial condition for next step
#         self.f_init = L.f[-1]

#         # indicate presence of new values at this level
#         L.status.updated = True

#         return None



class TrapezoidalRule(RungeKutta):
    """
    Famous trapezoidal rule of second order. Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), third one in eq. (213).
    """

    nodes = np.array([0.0, 1.0])
    weights = np.array([1.0 / 2.0, 1.0 / 2.0])
    matrix = np.zeros((2, 2))
    matrix[0, 0] = 0.0
    matrix[1, :] = [1.0 / 2.0, 1.0 / 2.0]
    ButcherTableauClass = ButcherTableau


class DIRK43_2(RungeKutta):
    """
    L-stable Diagonally Implicit RK method with four stages of order 3.
    Taken from https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
    """

    nodes = np.array([0.5, 2.0 / 3.0, 0.5, 1.0])
    weights = np.array([3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5])
    matrix = np.zeros((4, 4))
    matrix[0, 0] = 0.5
    matrix[1, :2] = [1.0 / 6.0, 0.5]
    matrix[2, :3] = [-0.5, 0.5, 0.5]
    matrix[3, :] = [3.0 / 2.0, -3.0 / 2.0, 0.5, 0.5]
    ButcherTableauClass = ButcherTableau


class MillerDIRK(RungeKutta):
    """
    Miller's DIRK method with three stages. Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), second one in eq. (217).
    """

    nodes = np.array([1.0, 1.0 / 3.0, 1.0])
    weights = np.array([0.0, 3.0, 1.0]) / 4.0
    matrix = np.zeros((3, 3))
    matrix[0, 0] = 1.0
    matrix[1, :2] = [-1.0 / 12.0, 5.0 / 12.0]
    matrix[2, :3] = [0.0, 3.0 / 4.0, 1.0 / 4.0]
    ButcherTableauClass = ButcherTableau


class EDIRK4(RungeKutta):
    """
    Stiffly accurate, fourth-order EDIRK with four stages. Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), second one in eq. (216).
    """

    nodes = np.array([0.0, 3.0 / 2.0, 7.0 / 5.0, 1.0])
    weights = np.array([13.0, 84.0, -125.0, 70.0]) / 42.0
    matrix = np.zeros((4, 4))
    matrix[0, 0] = 0
    matrix[1, :2] = [3.0 / 4.0, 3.0 / 4.0]
    matrix[2, :3] = [447.0 / 675.0, -357.0 / 675.0, 855.0 / 675.0]
    matrix[3, :] = [13.0 / 42.0, 84.0 / 42.0, -125.0 / 42.0, 70.0 / 42.0]
    ButcherTableauClass = ButcherTableau


class KurdiEDIRK45(RungeKutta):
    """
    Stiffly-accurate fourth-order EDIRK with five stages.
    Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), first one in eq. (218).
    """

    nodes = np.array([0.0, 2.0, 1.0, 1.0 / 2.0, 1.0])
    weights = np.array([1.0 / 6.0, 0.0, -5.0 / 12.0, 2.0 / 3.0, 7.0 / 12.0])
    matrix = np.zeros((5, 5))
    matrix[0, 0] = 0.0
    matrix[1, :2] = [1.0, 1.0]
    matrix[2, :3] = [5.0 / 12.0, -1.0 / 12.0, 2.0 / 3.0]
    matrix[3, :4] = [5.0 / 24.0, 0.0, -1.0 / 24.0, 1.0 / 3.0]
    matrix[4, :] = [1.0 / 6.0, 0.0, -5.0 / 12.0, 2.0 / 3.0, 7.0 / 12.0]
    ButcherTableauClass = ButcherTableau


class KurdiEDIRK45_2(RungeKutta):
    """
    Stiffly-accurate fourth-order EDIRK with five stages.
    Taken from
    [here](https://ntrs.nasa.gov/citations/20160005923), second one in eq. (218).
    """

    nodes = np.array([0.0, 4.0 / 3.0, 1.0, 1.0 / 2.0, 1.0])
    weights = np.array([1.0 / 6.0, 0.0, -26.0 / 54.0, 2.0 / 3.0, 35.0 / 54.0])
    matrix = np.zeros((5, 5))
    matrix[0, 0] = 0.0
    matrix[1, :2] = [2.0 / 3.0, 2.0 / 3.0]
    matrix[2, :3] = [3.0 / 8.0, -3.0 / 8.0, 1.0]
    matrix[3, :4] = [35.0 / 160.0, -3.0 / 160.0, 0.0, 48.0 / 160.0]
    matrix[4, :] = [1.0 / 6.0, 0.0, -26.0 / 54.0, 2.0 / 3.0, 35.0 / 54.0]
    ButcherTableauClass = ButcherTableau


class BackwardEulerDAE(RungeKuttaDAE, BackwardEuler):
    pass

class TrapezoidalRuleDAE(RungeKuttaDAE, CrankNicholson):
    pass

class ImplicitMidpointMethodDAE(RungeKuttaDAE, ImplicitMidpointMethod):
    pass

# class ImplicitMidpointMethodIMEXDAE(RungeKuttaIMEXDAE, ImplicitMidpointMethod):
#     pass

class DIRK43_2DAE(RungeKuttaDAE, DIRK43_2):
    pass

# class DIRK43_2IMEXDAE(RungeKuttaIMEXDAE, DIRK43_2):
#     pass

class EDIRK4_DAE(RungeKuttaDAE, EDIRK4):
    pass

class MillerDIRK_DAE(RungeKuttaDAE, MillerDIRK):
    pass

class KurdiEDIRK45DAE(RungeKuttaDAE, KurdiEDIRK45):
    pass

class KurdiEDIRK45_2DAE(RungeKuttaDAE, KurdiEDIRK45_2):
    pass
