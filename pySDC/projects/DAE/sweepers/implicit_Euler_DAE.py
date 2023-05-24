import numpy as np
from scipy import optimize

from pySDC.core.Errors import ParameterError
from pySDC.core.Sweeper import sweeper


class implicit_Euler_DAE(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    Implicit Euler for DAEs Sweeper to solve first order differential equations in fully implicit form
    Primarily implemented to be used with differential algebraic equations
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        assert params['num_nodes'] == 2, 'Implicit Euler (BDF method) only uses the value from t0'
        assert params['quad_type'] == 'LOBATTO', 'quad_type has to be LOBATTO due to both end points corresponding to nodes'

        # call parent's initialization routine
        super(implicit_Euler_DAE, self).__init__(params)

    def update_nodes(self):
        """
        Update the u- and f-values at the end of the time step
        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        # in the fully implicit case L.prob.eval_f() evaluates the function F(u, u', t)
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked

        u0 = L.u[0]

        # implicit function that have to be solved by Newton's method
        def fun(u):
            u_new = P.dtype_u(P.init)
            u_new[:] = u
            return P.eval_f(u_new, (u_new - u0) / L.dt, L.time + L.dt)

        options = dict()
        options['xtol'] = P.newton_tol
        options['eps'] = 1e-7
        opt = optimize.root(
            fun,
            u0,
            method='hybr',
            options=options,
            # callback= lambda x, f: print("solution:", x, " residual: ", f)
        )

        u_new = opt.x
        L.u[2][:] = u_new
        L.f[1][:] = (u_new - u0) / L.dt

        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        This function overrides the base implementation by always initialising level.f to zero
        This is necessary since level.f stores the solution derivative in the fully implicit case, which is not initially known
        """

        L = self.level
        P = L.prob

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

    def compute_residual(self):
        """
        Overrides the base implementation
        Uses the absolute value of the implicit function ||F(u', u, t)|| as the residual
        Returns:
            None
        """

        L = self.level
        P = L.prob

        # compute the residual at the end of the interval
        res_norm = []
        for m in range(self.coll.num_nodes):
            res_norm.append(abs(P.eval_f(L.u[m+1], L.f[m], L.time + L.dt * self.coll.nodes[m])))

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
        Compute u at the right point of the interval. The value uend computed here is the value of the next time step

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # a copy is sufficient
        L.uend = P.dtype_u(L.u[2])

        return None
