from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained as sdc_c

from pySDC.core.errors import ProblemError


class imex_sdc_c(sdc_c):
    r"""
    Base sweeper class for solving differential-algebraic equations of the form

    .. math::
        y' = f(y, z),

    .. math::
        0 = g(y, z).

    The SDC scheme applied to semi-explicit DAEs where no quadrature is applied to the constrains reads

    .. math::
       \mathbf{y}^{k+1} = \mathbf{y}_0 + \Delta t (\mathbf{Q} - \mathbf{Q}_\Delta)\otimes \mathbf{I}_{N_d}f(\mathbf{y}^{k}, \mathbf{z}^{k}) +
       \Delta t \mathbf{Q}_\Delta\otimes \mathbf{I}_{N_d}f(\mathbf{y}^{k+1}, \mathbf{z}^{k+1}),

    .. math::
       \mathbf{0} = g(\mathbf{y}^{k+1}, \mathbf{z}^{k+1}).

    Moreover, the differential equations are treated in IMEX fashion, where right-hand side is splitted
    into stiff and non-stiff parts. The stiff part is treated implicitly, while the non-stiff part is treated explicitly.
    """

    def __init__(self, params, level):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if "QI" not in params:
            params["QI"] = "IE"
        if "QE" not in params:
            params["QE"] = "EE"

        super().__init__(params, level)

        # IMEX integration matrices
        self.QI = self.get_Qdelta_implicit(qd_type=self.params.QI)
        self.QE = self.get_Qdelta_explicit(qd_type=self.params.QE)

    def integrate(self):
        """
        Integrates the right-hand side

        Returns:
            list of dtype_u: containing the integral as values
        """

        L = self.level
        P = L.prob

        me = []

        # integrate RHS over all collocation nodes
        for m in range(1, self.coll.num_nodes + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, self.coll.num_nodes + 1):
                me[-1].diff[:] += L.dt * self.coll.Qmat[m, j] * (L.f[j].diff_impl[:] + L.f[j].diff_expl[:])

        return me

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
                integral[m].diff[:] -= L.dt * (self.QI[m + 1, j] * L.f[j].diff_impl[:] + self.QE[m + 1, j] * L.f[j].diff_expl[:])

            # add initial value
            integral[m].diff[:] += L.u[0].diff[:]
            # add tau if associated
            if L.tau[m] is not None:
                integral[m][:] += L.tau[m].diff[:]

        # do the sweep
        for m in range(0, M):
            # build rhs, consisting of the known values from above and new values from previous nodes (at k+1)
            rhs = P.dtype_u(integral[m])
            for j in range(1, m + 1):
                rhs.diff[:] += L.dt * (self.QI[m + 1, j] * L.f[j].diff_impl[:] + self.QE[m + 1, j] * L.f[j].diff_expl[:])

            # implicit solve with prefactor stemming from the diagonal of Qd
            alpha = L.dt * self.QI[m + 1, m + 1]
            L.u[m + 1] = P.solve_system(rhs, alpha, L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

            # update function values
            L.f[m + 1] = P.eval_f(L.u[m + 1], L.time + L.dt * self.coll.nodes[m])

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

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            raise ProblemError("No collocation update possible due to algebraic constraints!")

        return None