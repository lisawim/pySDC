import numpy as np
from scipy import optimize

from pySDC.core.Errors import ParameterError
from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta
from pySDC.implementations.sweeper_classes.Runge_Kutta import ButcherTableau


class RungeKutta_DAE(RungeKutta):
    """
    Class that implements the Runge Kutta sweeper for solving differential-algebraic equations. The class inherits from RungeKutta class
    implements to solve ordinary differential equations.
    """

    def __init__(self, params):
        print('I am here')

        params['skip_residual_computation'] = ()

        # call parent's initialization routine
        super(RungeKutta_DAE, self).__init__(params)

    def predict(self):
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes.
        This function overrides the base implementation by always initialising level.f to zero.
        This is necessary since level.f stores the solution derivative in the fully implicit case,
        which is not initially known.
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        # set initial guess for gradient to zero
        L.f[0] = P.dtype_f(init=P.init)
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
        print('spread:', L.u)
        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True

    def integrate(self):
        """
        Integrates the right-hand side of the problem.

        Returns
        -------
        me : dtype_u
            Containing the integral as values.
        """

        # get current level and problem description
        L = self.level
        P = L.prob
        M = self.coll.num_nodes

        me = []
        return me

    def update_nodes(self):
        """
        Update the u- and u'-values at the intermediate stages. Note that this node-to-node computation is only possible if both explicit
        and implicit Runge-Kutta methods use a strictly lower triangular matrix A in the Butcher tableau, or a lower triangular matrix A
        in the implicit case, respectively. Otherwise, the other stages also have taken into account into the solving of the nonlinear
        systems of equations.

        Returns
        -------
        None
        """

        L = self.level
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        assert self.coll.implicit, 'Differential-algebraic equations should only treated numerically in an implicit way!'
        assert L.status.sweep <= 1, 'RK schemes are direct solvers. Please perform only one iteration!'

        # get number of collocation nodes for easier access
        M = self.coll.num_nodes

        for m in range(0, M):
            rhs = L.u[0]

            for j in range(1, m + 1):
                rhs += L.dt * self.QI[m + 1, j] * L.f[j]

            def implicit_function(deriv):
                """
                Defines the stages of the Runge-Kutta method.

                Parameters
                ----------
                deriv : dtype_u
                    Values of derivatives which is looking for.

                Returns
                -------
                nonlinear_sys : dtype_f
                    System of nonlinear equation which is want to solve by the Newton solver.
                """

                du = P.dtype_f(P.init)
                du[:] = deriv

                rhs_tmp = rhs
                rhs_tmp += L.dt * self.QI[m + 1, m + 1] * du

                nonlinear_sys = P.eval_f(rhs_tmp, du, L.time + L.dt * self.coll.nodes[m])
                return nonlinear_sys

            # solves the nonlinear system of equations of stage m to find the derivative
            du_sol = optimize.root(
                implicit_function,
                L.f[m],
                method='hybr',
                tol=P.newton_tol,
            )

            L.f[m + 1] = du_sol.x

            # Intermediate values for solution u can be determined by using the stage derivatives
            L.u[m + 1] = L.u[0]
            for j in range(1, m + 1):
                L.u[m + 1] += L.dt * self.QI[m + 1, j] * L.f[j]
        print('update_nodes:', L.u)
        # indicate presence of new values at this level
        L.status.updated = True

        return None

    def compute_residual(self, stage=None):
        """
        Overrides the base implementation
        Uses the absolute value of the implicit function ||F(u', u, t)|| as the residual

        Args:
            stage (str): The current stage of the step the level belongs to

        Returns:
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


class RK1_DAE(RungeKutta_DAE):
    def __init__(self, params):
        implicit = params.get('implicit', False)
        nodes = np.array([0.0])
        weights = np.array([1.0])
        if implicit:
            matrix = np.array(
                [
                    [1.0],
                ]
            )
        else:
            matrix = np.array(
                [
                    [0.0],
                ]
            )
        params['butcher_tableau'] = ButcherTableau(weights, nodes, matrix)
        super(RK1_DAE, self).__init__(params)


class RK4_DAE(RungeKutta_DAE):
    """
    Explicit Runge-Kutta of fourth order: Everybody's darling.
    """

    def __init__(self, params):
        nodes = np.array([0, 0.5, 0.5, 1])
        weights = np.array([1.0, 2.0, 2.0, 1.0]) / 6.0
        matrix = np.zeros((4, 4))
        matrix[1, 0] = 0.5
        matrix[2, 1] = 0.5
        matrix[3, 2] = 1.0
        params['butcher_tableau'] = ButcherTableau(weights, nodes, matrix)
        super(RK4_DAE, self).__init__(params)


class ImplicitEuler_DAE(RungeKutta_DAE):
    """
    Implicit Euler which is a Runge-Kutta method of order one.
    """

    def __init__(self, params):
        nodes = np.array([1.0])
        weights = np.array([1.0])
        matrix = np.array(
                [
                    [1.0],
                ]
            )
        params['butcher_tableau'] = ButcherTableau(weights, nodes, matrix)
        super(ImplicitEuler_DAE, self).__init__(params)
