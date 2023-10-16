import numpy as np
from scipy import optimize

from pySDC.core.Errors import ParameterError
from pySDC.core.Sweeper import sweeper


class fully_implicit_DAE(sweeper):
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
        if self.coll.left_is_node:
            raise ParameterError(msg)

        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)

    # TODO: hijacking this function to return solution from its gradient i.e. fundamental theorem of calculus.
    # This works well since (ab)using level.f to store the gradient. Might need to change this for release?
    def integrate(self):
        r"""
        Returns the solution by integrating its gradient (fundamental theorem of calculus) at each collocation node.
        Note that ``level.f`` stores the gradient values in the fully implicit case, rather than the evaluation of
        the right-hand side as in the ODE case.

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

        # integrate gradient over all collocation nodes
        for m in range(1, M + 1):
            # new instance of dtype_u, initialize values with 0
            me.append(P.dtype_u(P.init, val=0.0))
            for j in range(1, M + 1):
                me[-1] += L.dt * self.coll.Qmat[m, j] * L.f[j]

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

        # do the sweep
        for m in range(1, M + 1):
            # build implicit function, consisting of the known values from above and new values from previous nodes (at k+1)
            u_approx = P.dtype_u(integral[m - 1])
            # add the known components from current sweep del_t*Q_del*U_k+1
            for j in range(1, m):
                u_approx += L.dt * self.QI[m, j] * L.f[j]

            # params contains U = u'
            def impl_fn(params):
                # make params into a mesh object
                params_mesh = P.dtype_f(P.init)
                params_mesh[:] = params
                # build parameters to pass to implicit function
                local_u_approx = u_approx
                # note that derivatives of algebraic variables are taken into account here too
                # these do not directly affect the output of eval_f but rather indirectly via QI
                local_u_approx += L.dt * self.QI[m, m] * params_mesh
                return P.eval_f(local_u_approx, params_mesh, L.time + L.dt * self.coll.nodes[m - 1])

            # get U_k+1
            # note: not using solve_system here because this solve step is the same for any problem
            # See link for how different methods use the default tol parameter
            # https://github.com/scipy/scipy/blob/8a6f1a0621542f059a532953661cd43b8167fce0/scipy/optimize/_root.py#L220
            # options['xtol'] = P.params.newton_tol
            # options['eps'] = 1e-16
            opt = optimize.root(
                impl_fn,
                L.f[m],
                method='hybr',
                tol=P.newton_tol
                # callback= lambda x, f: print("solution:", x, " residual: ", f)
            )
            # update gradient (recall L.f is being used to store the gradient)
            L.f[m][:] = opt.x

        # Update solution approximation
        integral = self.integrate()
        for m in range(M):
            L.u[m + 1] = u_0 + integral[m]
        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def predict(self):
        r"""
        Predictor to fill values at nodes before first sweep. It can decides whether the

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
