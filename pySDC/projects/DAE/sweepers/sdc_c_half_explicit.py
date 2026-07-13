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
        print(params["QE"])
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

        M = self.coll.num_nodes
        QFE = self.get_Qdelta_explicit(qd_type="EE")

        if self.params.initial_guess != "forward_euler":
            raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')

        time_nodes = [L.time] + list(self.coll.nodes)
        for m in range(M):
            rhs = P.dtype_u(P.init)
            rhs.diff[:] = L.u[m].diff[:]

            u0 = P.dtype_u(P.init)
            u0.diff[:] = L.u[m].diff[:]

            alpha = L.dt * QFE[m + 1, m]
            L.u[m].alg[:] = P.solve_system(
                rhs=rhs,
                factor=alpha,
                u0=u0,
                t=L.time + L.dt * time_nodes[m + 1],
                t_c=L.time + L.dt * time_nodes[m],
            )

            L.f[m] = P.eval_f(L.u[m], L.time + L.dt * time_nodes[m])

            L.u[m + 1] = P.dtype_u(P.init)
            L.u[m + 1].diff[:] = L.u[m].diff[:] + alpha * L.f[m].diff[:]

        rhs = P.dtype_u(P.init)
        rhs.diff[:] = L.u[-1].diff[:]

        u0 = P.dtype_u(P.init)
        u0.diff[:] = L.u[-1].diff[:]

        alpha = L.dt * QFE[1, 0]
        L.u[-1].alg[:] = P.solve_system(
            rhs=rhs,
            factor=alpha,
            u0=u0,
            t=(L.time + L.dt) + L.dt * time_nodes[1],
            t_c=L.time + L.dt,
        )
        L.f[-1] = P.eval_f(L.u[-1], L.time + L.dt * time_nodes[-1])

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

        # get QF(u^k)
        integral = self.integrate()
        for m in range(M):
            # get -QdF(u^k)_m
            for j in range(1, M + 1):
                integral[m].diff[:] -= L.dt * self.QE[m + 1, j] * L.f[j].diff[:]

            # add initial value
            integral[m].diff[:] += L.u[0].diff[:]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m][:] += L.tau[m].diff[:]

        # do the sweep
        for m in range(M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m):
                rhs.diff[:] += L.dt * self.QE[m + 1, j] * L.f[j].diff[:]

            # Explicit factor [m+1, m+1] is zero
            alpha = L.dt * self.QE[m + 1, m]
            if m == 1:
                L.u[m + 1].diff[:] = rhs.diff[:]
            else:
                u0 = P.dtype_u(L.u[m])
                # u0.alg[:] = L.u[m].alg[:]
                L.u[m].alg[:] = P.solve_system(
                    rhs=rhs,
                    factor=alpha,
                    u0=u0,
                    t=L.time + L.dt * self.coll.nodes[m],
                    t_c=L.time + L.dt * self.coll.nodes[m - 1],
                )

                # update function values
                L.f[m][:] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m])

                L.u[m + 1].diff[:] = rhs.diff[:] + alpha * L.f[m].diff[:]

        # indicate presence of new values at this level
        L.status.updated = True

        return None
    
    def compute_end_point(self):
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
        """

        L = self.level
        P = L.prob

        M = self.coll.num_nodes

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            rhs = P.dtype_u(P.init)
            rhs.diff[:] = L.u[0].diff[:]
            for m in range(M - 1):
                rhs.diff[:] += L.dt * self.coll.weights[m] * L.f[m + 1].diff[:]

            u0 = P.dtype_u(L.u[M])
            alpha = L.dt * self.coll.weights[-1]
            L.u[-1].alg[:] = P.solve_system(
                rhs=rhs,
                factor=alpha,
                u0=u0,
                t=L.time + L.dt * self.coll.nodes[-1],
                t_c=L.time + L.dt * self.coll.nodes[-2],
            )

            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * L.f[m + 1]
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

        return None
