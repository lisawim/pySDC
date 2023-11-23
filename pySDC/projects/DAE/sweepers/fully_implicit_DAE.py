import numpy as np

from pySDC.core.Errors import ParameterError
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


class fully_implicit_DAE(generic_implicit):
    r"""
    Custom sweeper class to implement the fully-implicit SDC for solving DAEs. It solves fully-implicit DAE problems
    of the form

    .. math::
        F(t, u, u') = 0.

    It solves a collocation problem of the form

    .. math::
        F(\tau, \vec{U}_0 + \Delta t (\mathbf{Q} \otimes \mathbf{I}_n) \vec{U}, \vec{U}) = 0,

    where

    - :math:`\tau=(\tau_1,..,\tau_M) in \mathbb{R}^M` the vector of collocation nodes,
    - :math:`\vec{U}_0 = (u_0,..,u_0) \in \mathbb{R}^{Mn}` the vector of initial condition spread to each node,
    - spectral integration matrix :math:`\mathbf{Q} \in \mathbb{R}^{M \times M}`,
    - :math:`\vec{U}=(U_1,..,U_M) \in \mathbb{R}^{Mn}` the vector of unknown derivatives
      :math:`U_m \approx U(\tau_m) = u'(\tau_m) \in \mathbb{R}^n`,
    - and identity matrix :math:`\mathbf{I}_n \in \mathbb{R}^{n \times n}`.

    The construction of this sweeper is based on the concepts outlined in [1]_.

    Parameters
    ----------
    params : dict
        Parameters passed to the sweeper.

    Attributes
    ----------
    QI : np.2darray
        Implicit Euler integration matrix.
    du_init : dtype_f
        Stores the initial condition for derivative for each step.

    References
    ----------
    .. [1] J. Huang, J. Jun, M. L. Minion. Arbitrary order Krylov deferred correction methods for differential algebraic equation.
       J. Comput. Phys. Vol. 221 No. 2 (2007).
    """

    def __init__(self, params):
        """Initialization routine"""

        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super(fully_implicit_DAE, self).__init__(params)

        msg = f"Quadrature type {self.params.quad_type} is not implemented yet. Use 'RADAU-RIGHT' instead!"
        # if self.coll.left_is_node and not self.coll.right_is_node:
            # raise ParameterError(msg)

        self.Qmat = np.zeros((self.coll.num_nodes + 1, self.coll.num_nodes + 1))
        # if self.params.quad_type == 'RADAU-LEFT':
        #     if self.coll.num_nodes == 2:
        #         self.Qmat[1, 1:] = [1.0 / 4.0, -1.0 / 4.0]
        #         self.Qmat[2, 1:] = [1.0 / 4.0, 5.0 / 12.0]
        #     elif self.coll.num_nodes == 3:
        #         self.Qmat[1, 1:] = [1.0 / 9.0, (-1.0 - np.sqrt(6.0)) / 18.0, (-1.0 + np.sqrt(6.0)) / 18.0]
        #         self.Qmat[2, 1:] = [1.0 / 9.0, (88.0 + 7.0 * np.sqrt(6.0)) / 360.0, (88.0 - 43.0 * np.sqrt(6.0)) / 360.0]
        #         self.Qmat[3, 1:] = [1.0 / 9.0, (88.0 + 43.0 * np.sqrt(6.0)) / 360.0, (88.0 - 7.0 * np.sqrt(6.0)) / 360.0]
        # else:
        self.Qmat = self.coll.Qmat

        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        self.du_init = None
        print(f"Nodes: {self.coll.nodes}")
        print('Q:', self.Qmat)
        # print('QI:', self.QI)
        print('Weights:', self.coll.weights)

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
                me[-1] += L.dt * self.Qmat[m, j] * L.f[j]

        return me

    def update_nodes(self):
        r"""
        Updates values of ``u`` and ``f`` at collocation nodes. This correspond to a single iteration of the
        preconditioned Richardson iteration in **"ordinary"** SDC.
        """

        # get current level and problem description
        L = self.level
        # in the fully implicit case L.prob.eval_f() evaluates the function F(u, u', t)
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes
        u_0 = L.u[0]

        # get QU^k where U = u'
        # note that for multidimensional functions the required Kronecker product is achieved since
        # e.g. L.f[j] is a mesh object and multiplication with a number distributes over the mesh
        integral = self.integrate()
        # build the rest of the known solution u_0 + del_t(Q - Q_del)U_k
        for m in range(1, M + 1):
            for j in range(1, M + 1):
                integral[m - 1] -= L.dt * self.QI[m, j] * L.f[j]
            # add initial value
            integral[m - 1] += u_0
        # print('Integral:', integral)
        # do the sweep
        #for m in range(1, M + 1):
        for m in range(M):
            # build implicit function, consisting of the known values from above and new values from previous nodes (at k+1)
            u_approx = P.dtype_u(integral[m])
            # add the known components from current sweep del_t*Q_del*U_k+1
            # for j in range(1, m):
            for j in range(1, m + 1):
                # u_approx += L.dt * self.QI[m, j] * L.f[j]
                u_approx += L.dt * self.QI[m + 1, j] * L.f[j]
            # print(f"For m={m}, we have u_approx={u_approx}")
            # params contains U = u'
            def implSystem(params):
                """
                Build implicit system to solve in order to find the unknowns.

                Parameters
                ----------
                params : dtype_u
                    Unknowns of the system.

                Returns
                -------
                sys :
                    System to be solved as implicit function.
                """

                params_mesh = P.dtype_f(P.init)
                params_mesh[:] = params

                # build parameters to pass to implicit function
                local_u_approx = u_approx

                # note that derivatives of algebraic variables are taken into account here too
                # these do not directly affect the output of eval_f but rather indirectly via QI
                # local_u_approx += L.dt * self.QI[m, m] * params_mesh
                local_u_approx += L.dt * self.QI[m + 1, m + 1] * params_mesh
                # print(f"With diagonal element {L.dt * self.QI[m + 1, m + 1]} we have {local_u_approx}")
                # sys = P.eval_f(local_u_approx, params_mesh, L.time + L.dt * self.coll.nodes[m - 1])
                # print(f'Unknown du {params_mesh} and u {local_u_approx}')
                sys = P.eval_f(local_u_approx, params_mesh, L.time + L.dt * self.coll.nodes[m])
                # print(f'System has value {sys}')
                return sys

            # get U_k+1
            # note: not using solve_system here because this solve step is the same for any problem
            # See link for how different methods use the default tol parameter
            # https://github.com/scipy/scipy/blob/8a6f1a0621542f059a532953661cd43b8167fce0/scipy/optimize/_root.py#L220
            # options['xtol'] = P.params.newton_tol
            # options['eps'] = 1e-16

            # which L.f should be used as initial condition? L.f[m] or L.f[m - 1]?
            # For 'RADAU-RIGHT' it does not make any difference, but for 'LOBATTO' we have to use L.f[0],
            # so at least L.f[m - 1] where the initial condition can be found
            # du_new = P.solve_system(implSystem, L.f[m], L.time + L.dt * self.coll.nodes[m - 1])
            # print(f'Initial condition is {L.f[m]}')
            du_new = P.solve_system(implSystem, L.f[m], L.time + L.dt * self.coll.nodes[m])

            # update gradient (recall L.f is being used to store the gradient)
            # L.f[m][:] = du_new
            L.f[m + 1][:] = du_new

        # Update solution approximation
        integral = self.integrate()
        for m in range(M):
            L.u[m + 1] = u_0 + integral[m]
        # print(f'Update of u at time {L.time} is: {L.u}')
        # print(f'Update of f at time {L.time} is: {L.f}')
        # print()

        # store end point for gradient as initial condition for next step
        # self.du_init = None if not self.coll.left_is_node else L.f[-1]
        # print('Initial condition for f for next step:', self.du_init)
        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def predict(self):
        r"""
        Predictor to fill values at nodes before first sweep. It can decide whether the

            - initial condition is spread to each node ('initial_guess' = 'spread'),
            - zero values are spread to each node ('initial_guess' = 'zero'),
            - or random values are spread to each collocation node ('initial_guess' = 'random').

        Default prediction for the sweepers, only copies the values to all collocation nodes. This function
        overrides the base implementation by always initialising ``level.f`` to zero. This is necessary since
        ``level.f`` stores the solution derivative in the fully implicit case, which is not initially known.
        """
        # get current level and problem description
        L = self.level
        P = L.prob

        # set initial guess f[0] for gradient
        L.f[0] = P.dtype_f(init=P.init, val=0.0)
        if self.coll.left_is_node:
            L.f[0][:] = P.du_exact(L.time) if self.du_init is None else self.du_init
            # print(f'Initial condition at time {L.time} should be approx {P.du_exact(L.time)} we have {self.du_init}')
        for m in range(1, self.coll.num_nodes + 1):
            # copy u[0] and f[0] to all collocation nodes
            if self.params.initial_guess == 'spread':
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
                if self.coll.left_is_node:
                    L.f[m][:] = P.du_exact(L.time) if self.du_init is None else self.du_init
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            # start with random initial guess
            elif self.params.initial_guess == 'random':
                L.u[m] = P.dtype_u(init=P.init, val=np.random.rand(1)[0])
                L.f[m] = P.dtype_f(init=P.init, val=np.random.rand(1)[0])
            else:
                raise ParameterError(f'initial_guess option {self.params.initial_guess} not implemented')
        # print(f'Prediction of u at time {L.time} is: {L.u}')
        # print(f'Prediction of f at time {L.time} is: {L.f}')
        # print()
        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

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

        # check if there are new values (e.g. from a sweep)
        # assert L.status.updated

        # compute the residual for each node

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
        # print(f'Residual at time {L.time} is {L.status.residual}')
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
            self.du_init = None if not self.coll.left_is_node else L.f[-1][:]
        else:
            # start with u0 and add integral over the full interval (using coll.weights)
            L.uend = P.dtype_u(L.u[0])
            for m in range(self.coll.num_nodes):
                L.uend += L.dt * self.coll.weights[m] * L.f[m + 1]
            # add up tau correction of the full interval (last entry)
            if L.tau[-1] is not None:
                L.uend += L.tau[-1]

            def update_U(U):
                """
                Function to update U at end of time step.

                Parameters
                ----------
                U : dtype_f
                    Gradient of u.

                Returns
                -------
                sys : dtype_f
                    System to be solved.
                """
                sys = P.eval_f(L.uend, U, L.time + L.dt)
                return sys
            
            duend = P.du_exact(L.time + L.dt)  # P.solve_system(update_U, P.dtype_f(init=P.init, val=0.0), L.time + L.dt)
            self.du_init = duend
            # print('duend:', duend)

        return None