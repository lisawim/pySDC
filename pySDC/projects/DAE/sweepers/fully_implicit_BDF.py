import numpy as np
from scipy import optimize

from pySDC.core.Errors import ParameterError
from pySDC.core.Sweeper import sweeper


class fully_implicit_BDF(sweeper):
    """
    Custom sweeper class, implements Sweeper.py

    BDF Sweeper to solve first order differential equations in fully implicit form
    Primarily implemented to be used with differential algebraic equations
    """

    def __init__(self, params):
        """
        Initialization routine for the custom sweeper

        Args:
            params: parameters for the sweeper
        """

        assert params['num_nodes'] == 2, 'Implicit Euler BDF only uses the value from t0!'

        # call parent's initialization routine
        super(fully_implicit_BDF, self).__init__(params)

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single iteration of the preconditioned Richardson iteration >

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
        L.u[1][:] = u_new
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
        res_norm = abs(P.eval_f(L.u[1], L.f[1], L.time + L.dt))

        L.status.residual = res_norm

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

        L = self.level
        P = L.prob

        # check if Mth node is equal to right point and do_coll_update is false, perform a simple copy
        if self.coll.right_is_node and not self.params.do_coll_update:
            # a copy is sufficient
            L.uend = P.dtype_u(L.u[1])
        else:
            L.uend = P.dtype_u(L.u[0])

        return None
