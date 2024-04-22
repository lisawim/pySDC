import numpy as np
np.set_printoptions(precision=30)
from scipy.optimize import root

from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta, ButcherTableau, BackwardEuler, CrankNicholson


class RungeKuttaDAE(RungeKutta):
    def __init__(self, params):
        super().__init__(params)
        self.du_init = None
        self.file_name = '/home/lisa/Buw/Programme/Python/Libraries/pySDC/pySDC/projects/DAE/run/data/pysdc.txt'
        file = open(self.file_name, 'w')
        file.close()
        # print(self.QI[1:, 1:])

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep.
        """

        # get current level and problem
        L = self.level
        P = L.prob

        du_init = self.du_init[:] if self.du_init is not None else P.du_exact(L.time)
        # print('du_init:', du_init)
        # print()
        L.f[0] = du_init
        file = open(self.file_name, 'a')
        file.write(f"t={round(L.time, 5)}: Predict u: %s \n" % L.u[0])
        file.write(f"t={round(L.time, 5)}: Predict f: {L.f[0]} \n")
        file.close()
        for m in range(1, self.coll.num_nodes + 1):
            L.u[m] = P.dtype_u(init=P.init, val=0.0)
            L.f[m] = P.dtype_f(init=P.init, val=0.0)

        L.status.unlocked = True
        L.status.updated = True

    def integrate(self):
        """
        Integrates the right-hand side.

        Returns
        -------
        me : list of dtype_u
            Containing the integral as values.
        """

        # get current level and problem
        L = self.level
        P = L.prob

        me = []
        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1] += L.dt * self.QI[m, j] * L.f[j][:]

        return me

    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        assert L.status.sweep <= 1, "RK schemes are direct solvers. Please perform only 1 iteration!"

        M = self.coll.num_nodes
        for m in range(M):
            u_approx = P.dtype_u(L.u[0])
            for j in range(1, m + 1):
                u_approx += L.dt * self.QI[m + 1, j] * L.f[j]
                # print(f"m={m}, j={j}, q={self.QI[m + 1, j]}")
                # file = open(self.file_name, 'a')
                # file.write(f'm={m}, j={j}, q={self.QI[m + 1, j]}: {u_approx}\n')
                # file.close()
            
            file = open(self.file_name, 'a')
            file.write(f'u_approx at time {round(L.time + L.dt * self.coll.nodes[m], 5)}: {u_approx}\n')
            file.close()

            # print(f'Implicit factor: m={m}: q={self.QI[m + 1, m + 1]}')
            def implSystem(unknowns):
                """
                Build implicit system to solve in order to find the unknowns for the derivative
                of u.

                Parameters
                ----------
                unknowns : dtype_u
                    Unknowns of the system.

                Returns
                -------
                sys : dtype_f
                    System to be solved.
                """
                unknowns_mesh = P.dtype_f(unknowns)

                local_u_approx = P.dtype_f(u_approx)

                # defines the "implicit" factor, note that for explicit RK the diagonal element is zero
                local_u_approx += L.dt * self.QI[m + 1, m + 1] * unknowns_mesh
                file = open(self.file_name, 'a')
                # file.write(f'        local_u_approx={local_u_approx}\n')
                # file.write(f'        unknowns_mesh={unknowns_mesh}\n')
                file.close()
                sys = P.eval_f(local_u_approx, unknowns_mesh, L.time + L.dt * self.coll.nodes[m + 1])
                return sys
            file = open(self.file_name, 'a')
            file.write(f'Initial guess at time {round(L.time + L.dt * self.coll.nodes[m + 1], 5)}: {L.f[m]}\n')
            file.close()
            L.f[m + 1][:] = P.solve_system(implSystem, L.f[m], L.time + L.dt * self.coll.nodes[m + 1])
        file = open(self.file_name, 'a')
        file.write(f'\n')
        file.close()
        # Update numerical solution
        L.u[-1][:] = L.u[0][:]
        for j in range(1, M + 1):
            L.u[-1][:] += L.dt * self.QI[-1, j] * L.f[j]#L.u[0][:] + L.dt * self.QI[3, 1] * L.f[1][:] + L.dt * self.QI[3, 2] * L.f[2][:] + L.dt * self.QI[3, 3] * L.f[3][:]

        print('after update_nodes:', L.f[-1])
        self.du_init = P.dtype_f(P.init)
        self.du_init[:] = L.f[-1][:]

        L.status.updated = True

        return None


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


class BackwardEulerDAE(RungeKuttaDAE, BackwardEuler):
    pass


class TrapezoidalRuleDAE(RungeKuttaDAE, CrankNicholson):
    pass
