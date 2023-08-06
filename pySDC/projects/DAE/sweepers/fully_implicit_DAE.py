import numpy as np
from scipy import optimize

from pySDC.core.Errors import ParameterError
from pySDC.core.Sweeper import sweeper


class fully_implicit_DAE(sweeper):
    r"""
    Custom sweeper class to implement the fully-implicit SDC for solving DAEs. It solves DAE problems of the form

    .. math::
        F(u, u', t) = 0.

    It solves a collocation problem of the form

    .. math::
        F(\tau, \vec{U}_0 + \Delta t (\mathbf{Q} \otimes \mathbf{I}_n) \vec{U}, \vec{U}) = 0.

    The sweeper implementation bases on the concepts outlined in the KDC publication [1]_.

    Attributes
    ----------
    QI : np.2darray
        Implicit Euler integration matrix.
    newton_maxiter : float
        Maximum number of iterations the Newton solver should do.
    work_counters : WorkCounter
        Counts iterations, here: the number of Newton iterations.

    References
    ----------
    .. [1] J. Huang, J. Jun, M. L. Minion. Arbitrary order Krylov deferred correction methods for differential algebraic equation.
       J. Comput. Phys. Vol. 221 No. 2 (2007).
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        if 'QI' not in params:
            params['QI'] = 'IE'

        # call parent's initialization routine
        super(fully_implicit_DAE, self).__init__(params)

        msg = f"Quadrature type {self.params.quad_type} is not implemented yet. Use 'RADAU-RIGHT' instead!"
        if self.coll.left_is_node:
            raise ParameterError(msg)

        self.QI = self.get_Qdelta_implicit(coll=self.coll, qd_type=self.params.QI)
        self.newton_maxiter = 100
        self.counter_solve = 0

    # TODO: hijacking this function to return solution from its gradient i.e. fundamental theorem of calculus.
    # This works well since (ab)using level.f to store the gradient. Might need to change this for release?
    def integrate(self):
        """
        Returns the solution by integrating its gradient (fundamental theorem of calculus). Note that level.f
        stores the gradient values in the fully-implicit case, rather than the evaluation of the right-hand side
        as in the ODE case.

        Returns
        -------
        me : dtype_u:
            Integrated numerical solution as mesh.
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
        """
        Updates the values of u and u' at the collocation nodes. This procedure corresponds to a single iteration
        of the preconditioned Richardson iteration in the ordinary SDC.

        Returns
        -------
        None
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

            def F(U):
                """
                Helper function to define an implicit function that can be solved using an iterative solver.

                Parameters
                ----------
                U : dtype_u
                    The current numerical solution as mesh.
                """

                U_mesh = P.dtype_f(P.init)
                U_mesh[:] = U
                local_u_approx = u_approx
                local_u_approx += L.dt * self.QI[m, m] * U_mesh
                return P.eval_f(local_u_approx, U_mesh, L.time + L.dt * self.coll.nodes[m - 1])

            def Fprime(U, dt_jac=1e-9):
                """
                Approximates the Jacobian of F using finite differences.

                Parameters
                ----------
                U : dtype_u
                    Vector for which the Jacobian is computed.
                dt_jac : float, optional
                    Step size for finite differences.

                Returns
                -------
                jac : np.ndarray
                    The Jacobian.
                """

                if not P.jac:
                    N, M = len(U), len(F(U))
                    jac = np.zeros((N, M))
                    e = np.zeros(N)
                    e[0] = 1
                    for k in range(N):
                        jac[:, k] = (1 / dt_jac) * (F(U + dt_jac * e) - F(U))
                        e = np.roll(e, 1)
                else:
                    jac = P.get_Jacobian(L.time + L.dt * self.coll.nodes[m - 1], U)
                return jac
            root_solver = 'hybr'
            root, n = solve_nonlinear_system(root_solver, L.f[m], F, Fprime, P.newton_tol)
            self.counter_solve += n
            # Update of U' (stored in L.f)
            L.f[m][:] = root

        # Update solution approximation
        integral = self.integrate()
        for m in range(M):
            L.u[m + 1] = u_0 + integral[m]
        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep. It can decides whether the

            - initial condition is spread to each node ('initial_guess' = 'spread'),
            - zero values are spread to each node ('initial_guess' = 'zero'),
            - or random values are spread to each collocation node ('initial_guess' = 'random').

        Default prediction for the sweepers, only copies the values to all collocation nodes. This function
        overrides the base implementation by always initialising level.f to zero. This is necessary since
        level.f stores the solution derivative in the fully implicit case, which is not initially known
        """
        # get current level and problem description
        L = self.level
        P = L.prob
        self.counter_solve = 0
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
        Computes the residual of the DAE, which is the absolute value of the implicit function
        :math:`||F(u, u', t)||` as the residual.

        Parameters
        ----------
        stage : str
            The current stage of the step the level belongs to.

        Returns
        -------
        None
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
        """
        Compute u at the right point of the interval

        The value uend computed here is a full evaluation of the Picard formulation unless do_full_update==False

        Returns:
            None
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

def newton(u0, F, Fprime, newton_tol, newton_maxiter):
    """
    Multi-dimensional Newton's method to find the root of the nonlinear system.

    Parameters
    ----------
    u0 : np.ndarray
        Initial condition
    F : callable function
        Nonlinear function where Newton's method is applied at.
    Fprime : callable function
        Jacobian matrix of function F approximated by finite differences.
    newton_tol : float
        Tolerance for Newton to terminate.
    newton_maxiter : int
        Maximum number of iterations that Newton should do.
    """

    n = 0
    while n < newton_maxiter:
        g = F(u0)
        res = np.linalg.norm(g, np.inf)

        if res < newton_tol:
            break

        J_inv = np.linalg.inv(Fprime(u0))

        u0 -= J_inv.dot(g)

        n += 1
    print('Newton took {} iterations with error {}'.format(n, res))
    print()
    root = u0
    return root, n

def solve_nonlinear_system(root_solver, u0, F, Fprime, newton_tol, newton_maxiter=100):
    """
    Function that solves the nonlinear system.

    Parameters
    ----------
    root_solver : str
        Indicates which solver should be used for the nonlinear system.
        Either 'newton' or 'hybr'.
    u0 : np.ndarray
        Initial condition
    F : callable function
        Nonlinear function where Newton's method is applied at.
    Fprime : callable function
        Jacobian matrix of function F approximated by finite differences.
    newton_tol : float
        Tolerance for Newton to terminate.
    newton_maxiter : int
        Maximum number of iterations that Newton should do.
    """

    if root_solver == 'newton':
        root, n = newton(u0, F, Fprime, newton_tol, newton_maxiter=100)
    elif root_solver == 'hybr':
        opt = optimize.root(
            F,
            u0,
            method='hybr',
            tol=newton_tol,
        )
        root = opt.x
        n = opt.nfev
        # print(opt)
    else:
        raise ParameterError("Choose either 'newton' or 'hybr'!")

    return root, n
