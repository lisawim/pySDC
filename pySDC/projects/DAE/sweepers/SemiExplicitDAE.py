import numpy as np
from scipy import optimize

from pySDC.core.Errors import ParameterError
from pySDC.core.Sweeper import sweeper


class SemiExplicitDAE(sweeper):
    r"""
    Custom sweeper class to implement SDC for solving semi-explicit DAEs of the form

    .. math::
        u' = f(u, z, t),

    .. math::
        0 = g(u, z, t)

    with :math:`u(t), u'(t) \in\mathbb{R}^{n_d}` the differential variables and their derivates,
    algebraic variables :math:`z(t) \in\mathbb{R}^{n_a}`, :math:`f(u, z, t) \in \mathbb{R}^{n_d}`,
    and :math:`g(u, z, t) \in \mathbb{R}^{n_a}`. :math:`n = n_d + n_a` is the dimension of the whole
    system of DAEs.

    It solves a collocation problem of the form

    .. math::
        U = f(\vec{U}_0 + \Delta t (\mathbf{Q} \otimes \mathbf{I}_{n_d}) \vec{U}, \vec{z}, \tau),

    .. math::
        0 = g(\vec{U}_0 + \Delta t (\mathbf{Q} \otimes \mathbf{I}_{n_d}) \vec{U}, \vec{z}, \tau),

    where
    
    - :math:`\tau=(\tau_1,..,\tau_M) in \mathbb{R}^M` the vector of collocation nodes,
    - :math:`\vec{U}_0 = (u_0,..,u_0) \in \mathbb{R}^{Mn_d}` the vector of initial condition spread to each node,
    - spectral integration matrix :math:`\mathbf{Q} \in \mathbb{R}^{M \times M}`,
    - :math:`\vec{U}=(U_1,..,U_M) \in \mathbb{R}^{Mn_d}` the vector of unknown derivatives of differential variables
      :math:`U_m \approx U(\tau_m) = u'(\tau_m) \in \mathbb{R}^{n_d}`,
    - :math:`\vec{z}=(z_1,..,z_M) \in \mathbb{R}^{Mn_a}` the vector of unknown algebraic variables
      :math:`z_m \approx z(\tau_m) \in \mathbb{R}^{n_a}`,
    - and identity matrix :math:`\mathbf{I}_{n_d} \in \mathbb{R}^{n_d \times n_d}`.

    This sweeper treates the differential and the algebraic variables differently by only integrating the differential
    components. Solving the nonlinear system, :math:`{U,z}` are the unknowns.

    The sweeper implementation is based on the ideas mentioned in the KDC publication [1]_.

    Parameters
    ----------
    params : dict
        Parameters passed to the sweeper.

    Attributes
    ----------
    QI : np.2darray
        Implicit Euler integration matrix.

    References
    ----------
    .. [1] J. Huang, J. Jun, M. L. Minion. Arbitrary order Krylov deferred correction methods for differential algebraic
       equation. J. Comput. Phys. Vol. 221 No. 2 (2007).

    Note
    ----
    The right-hand side of the problem DAE classes using this sweeper has to be exactly implemented in the way, the
    semi-explicit DAE is defined. Define :math:`\vec{x}=(y, z)^T`, :math:`F(\vec{x})=(f(\vec{x}), g(\vec{x}))`, and the
    matrix

    .. math::
        A = \begin{matrix}
            I & 0 \\
            0 & 0
        \end{matrix}

    then, the problem can be reformulated as

    .. math::
        A\vec{x}' = F(\vec{x}).

    Then, setting :math:`F_{new}(\vec{x}, \vec{x}') = A\vec{x}' - F(\vec{x})` defines a DAE of fully-implicit form

    .. math::
        0 = F_{new}(\vec{x}, \vec{x}').

    Hence, the method ``eval_f`` of problem DAE classes of semi-explicit form implements the right-hand side in the way of
    returning :math:`F(\vec{x})`, whereas ``eval_f`` of problem classes of fully-implicit form return the right-hand side
    :math:`F_{new}(\vec{x}, \vec{x}')`.
    """

    def __init__(self, params):
        """Initialization routine"""

        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super(SemiExplicitDAE, self).__init__(params)

        msg = f"Quadrature type {self.params.quad_type} is not implemented yet. Use 'RADAU-RIGHT' instead!"
        if self.coll.left_is_node:
            raise ParameterError(msg)

        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)

    def predict(self):
        r"""
        Predictor to fill values at nodes before first sweep. It can decides whether the

            - initial condition is spread to each node ('initial_guess' = 'spread'),
            - zero values are spread to each node ('initial_guess' = 'zero'),
            - or random values are spread to each collocation node ('initial_guess' = 'random').

        Default prediction for the sweepers, only copies the values to all collocation nodes. This function
        overrides the base implementation by always initialising ``level.f`` to zero. This is necessary since
        ``level.f`` stores the solution derivative, which is not initially known.
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        assert P.diff_nvars is not None, 'This sweeper can only be used for DAE problems of semi-explicit form!'

        # set initial guess for gradient to zero
        L.f[0] = P.dtype_f(init=P.init, val=0.0)

        for m in range(1, self.coll.num_nodes + 1):
            # copy u[0] to all collocation nodes and set f (the gradient) to zero
            if self.params.initial_guess == 'spread':
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            # start with zero everywhere
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            # start with random initial guess
            elif self.params.initial_guess == 'random':
                L.u[m] = P.dtype_u(init=P.init, val=np.random.rand(1)[0])
                L.f[m] = P.dtype_f(init=P.init, val=np.random.rand(1)[0])
            else:
                raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def integrate(self):
        r"""
        Returns the solution by integrating its gradient (fundamental theorem of calculus) at each collocation node.
        ``level.f`` stores the gradient of solution ``level.u``.

        Returns
        -------
        me : list of lists
            Integral of the gradient at each collocation node.
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        me = []
        for m in range(1, M + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, M + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * L.f[j]
                me[-1][P.diff_nvars :] += L.u[j][P.diff_nvars :]

        return me
    
    def update_nodes(self):
        r"""
        Updates the values of solution ``u`` and their gradient stored in ``f``.
        """

        # get current level and problem description
        L = self.level
        # in the fully implicit case L.prob.eval_f() evaluates the right-hand side of the differential equations
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        # get number of collocation nodes for easier access
        M = self.coll.num_nodes
        u_0 = L.u[0]

        integral = self.integrate()
        # build the rest of the known solution u_0 + del_t(Q - Q_del)U_k
        for m in range(1, M + 1):
            for j in range(1, M + 1):
                integral[m - 1] -= L.dt * self.QI[m, j] * L.f[j]
            # add initial value
            integral[m - 1] += u_0
            integral[m - 1][P.diff_nvars :] = L.u[m][P.diff_nvars :]

        # do the sweep
        for m in range(1, M + 1):
            u_approx = P.dtype_u(integral[m - 1])
            for j in range(1, m):
                u_approx += L.dt * self.QI[m, j] * L.f[j]
            u_approx[P.diff_nvars :] = L.u[m][P.diff_nvars :]

            def implSystem(unknowns):
                """
                Build implicit system to solve in order to find the unknowns.

                Parameters
                ----------
                unknowns : dtype_u
                    Unknowns of the system.

                Returns
                -------
                sys :
                    System to be solved as implicit function.
                """

                unknowns_mesh = P.dtype_f(P.init)
                unknowns_mesh[:] = unknowns
                local_u_approx = u_approx

                # for j in range(P.diff_nvars):
                #     local_u_approx[j] += L.dt * self.QI[m, m] * unknowns_mesh[j]
                local_u_approx += L.dt * self.QI[m, m] * unknowns_mesh
                local_u_approx[P.diff_nvars :] = unknowns_mesh[P.diff_nvars :]
                sys = P.eval_f(local_u_approx, unknowns_mesh[: P.diff_nvars], L.time + L.dt * self.coll.nodes[m - 1])

                return sys

            U0_diff, p0_alg = np.array(L.f[m][: P.diff_nvars]), np.array(L.u[m][P.diff_nvars :]) 
            u0 = np.concatenate((U0_diff, p0_alg))
            # solve = optimize.root(
            #     implSystem,
            #     u0,
            #     method='hybr',
            #     tol=P.newton_tol,
            # )
            # self.work_counters['newton'].niter += solve.maxfev
            u_new = P.solve_system(implSystem, u0, L.time + L.dt * self.coll.nodes[m - 1])

            # ---- update U' and z ----
            L.f[m][: P.diff_nvars] = u_new[: P.diff_nvars]
            L.u[m][P.diff_nvars :] = u_new[P.diff_nvars :]

        # Update solution approximation
        integral = self.integrate()
        for m in range(M):
            L.u[m + 1][: P.diff_nvars] = u_0[: P.diff_nvars] + integral[m][: P.diff_nvars]

        # indicate presence of new values at this level
        L.status.updated = True

        return None
    
    def compute_residual(self, stage=None):
        r"""
        Uses the absolute value of the DAE system

        .. math::
            ||F(t, u, u')||

        for computing the residual.

        Parameters
        ----------
        stage : str, optional
            The current stage of the step the level belongs to.
        """
        
        # get current level and problem description
        L = self.level
        P = L.prob

        # Check if we want to skip the residual computation to gain performance
        # Keep in mind that skipping any residual computation is likely to give incorrect outputs of the residual!
        if stage in self.params.skip_residual_computation:
            L.status.residual = 0.0 if L.status.residual is None else L.status.residual
            return None
        
        res_norm = []
        for m in range(self.coll.num_nodes):
            # use abs function from data type here
            res_norm.append(abs(P.eval_f(L.u[m + 1], L.f[m + 1], L.time + L.dt * self.coll.nodes[m])))

        # find maximal residual over the nodes
        if L.params.residual_type == 'full_abs':
            L.status.residual = max(res_norm)
        elif L.params.residual_type == 'last_abs':
            L.status.residual = res_norm[-1]
        elif L.params.residual_type == 'full_rel':
            L.status.residual = max(res_norm) / abs(L.u[0])
        elif L.params.residual_type == 'last_rel':
            L.status.residual = res_norm[-1] / abs(L.u[0])
        else:
            raise ParameterError(
                f'residual_type = {L.params.residual_type} not implemented, choose '
                f'full_abs, last_abs, full_rel or last_rel instead'
            )

        # indicate that the residual has seen the new values
        L.status.updated = False

        return None
    
    def compute_end_point(self):
        r"""
        Computes the solution ``u`` at the right-hand point. For ``quad_type='RADAU-LEFT'`` a collocation update
        has to be done, which is the full evaluation of the Picard formulation. In cases of
        ``quad_type='RADAU-RIGHT'`` or ``quad_type='LOBATTO'`` the value at last collocation node is the new value
        for the next step.
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[-1])
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * L.f[m + 1]

        return None