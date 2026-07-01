from pySDC.core.errors import ParameterError
from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained


class sdc_c_half_explicit(genericImplicitConstrained):
    def __init__(self, params, level):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if "QE" not in params:
            params["QE"] = "EE"

        # call parent's initialization routine
        super().__init__(params, level)

        self.elapsed_time_update_coeffs = 0.0

        # get QE matrix
        self.QE = self.get_Qdelta_explicit(qd_type=self.params.QE)

    def predict(self) -> None:
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # evaluate RHS at left point
        u0 = P.dtype_u(L.u[0])
        u_init = P.dtype_u(P.init)
        u_init.diff[:] = u0.diff[:]
        L.f[0] = P.eval_f(u_init, L.time)

        for m in range(1, self.coll.num_nodes + 1):
            # copy u[0] to all collocation nodes, evaluate RHS
            if self.params.initial_guess == "spread":
                u = P.dtype_u(L.u[0])
                L.u[m].diff[:] = u.diff[0]
                L.u[m].alg[:] = P.dtype_u(P.init).alg[:]
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])
            # copy u[0] and RHS evaluation to all collocation nodes
            elif self.params.initial_guess == "copy":
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.dtype_f(L.f[0])
            # start with zero everywhere
            elif self.params.initial_guess == "zero":
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            # start with random initial guess
            elif self.params.initial_guess == "random":
                L.u[m] = P.dtype_u(init=P.init, val=self.rng.rand(1)[0])
                L.f[m] = P.dtype_f(init=P.init, val=self.rng.rand(1)[0])
            else:
                raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        self.updateVariableCoeffs(L.status.sweep)

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        # gather all terms which are known already (e.g. from the previous iteration)
        # this corresponds to u0 + QF(u^k) - QdF(u^k) + tau

        def h(zz):
            integral = []

            # integrate RHS over all collocation nodes
            for m in range(1, M + 1):
                # new instance of dtype_u, initialize values with 0
                integral.append(P.dtype_u(P.init, val=0.0))
                for j in range(1, M + 1):
                    tau = L.time + L.dt * self.coll.nodes[j - 1]
                    integral[-1].diff[:] += L.dt * self.coll.Qmat[m, j] * P.eval_f_diff(L.u[j], zz, tau)

            # get QF(u^k)
            for m in range(M):
                # get -QdF(u^k)_m
                for j in range(1, M + 1):
                    tau = L.time + L.dt * self.coll.nodes[j - 1]
                    integral[m].diff[:] -= L.dt * self.QE[m + 1, j] * P.eval_f_diff(L.u[j], zz, tau)

                # add initial value
                integral[m].diff[:] += L.u[0].diff[:]
                # add tau if associated
                if L.tau[m] is not None:
                    integral[m][:] += L.tau[m].diff[:]

            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                tau = L.time + L.dt * self.coll.nodes[j - 1]
                rhs.diff[:] += L.dt * self.QE[m + 1, j] * P.eval_f_diff(L.u[j], zz, tau)


        # do the sweep
        for m in range(M):
            # Explicit factor [m+1, m+1] is zero
            # alpha = L.dt * self.QE[m + 1, m + 1]
            L.u[m + 1] = P.solve_system(h, L.u[m + 1].alg[:], L.time + L.dt * self.coll.nodes[m])

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

        # indicate presence of new values at this level
        L.status.updated = True

        return None
    
    @staticmethod
    def f_diff(dt, Q, QE, y, y0, z, t):