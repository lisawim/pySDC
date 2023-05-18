import numpy as np
import logging
from scipy import optimize

from pySDC.core.Sweeper import sweeper, _Pars
from pySDC.core.Errors import ParameterError


class StartingFramework(object):
    def __init__(self, k_step, starting_values):
        """
        Initialization routine to get starting values for a k-step BDF method for solving differential-algebraic equations.
        If no starting_values are implemented in order to start BDF, there will be some computed using the
        initial condition u0 implemented in the problem class.

        Attributes:
            k_step (int): order of BDF method, and number of previous steps used to compute the value for new step
            starting_values (np.ndarray): contains the initial values for BDF to start
            num_nodes (int): number of nodes used for one step (which is only one)
        """

        if not isinstance(k_step, int):
            raise ParameterError('Number of steps used for BDF needs to be an integer value!')

        if k_step <= 0:
            raise ParameterError('Number of steps used for BDF needs to be larger than zero!')
        elif k_step > 6:
            raise ParameterError('More than {k_step} steps for BDF leads to an unstable method!')

        if k_step == 1:
            self.starting_values = starting_values if starting_values is not None else []
        else:
            raise ParameterError(
                f'Possibility of computing starting_values for {k_step}-step BDF to start not implemented yet!'
            )

        self.num_nodes = 1

    def get_starting_values(self, k_step):
        """
        Computes starting values for BDF to start using a k_step-th order accurate method

        Args:
            k_step (int): order for the method to compute starting values

        Returns:
            starting_values (np.ndarray): contains the starting values
        """
        raise NotImplementedError('There is not an method implemented to compute starting values for BDF!')


class BDF_DAE(sweeper):
    """
    This class initialises the famous backward differentiation formulas (BDF) for solving differential-algebraic equations (DAEs).
    In order to use this sweeper, for using k-steps the sweeper needs k values to start with solving. Either they can
    be given as parameters, or instead starting values will be generated by an k-accurate method.

    BDF only does one iteration, because it is a multi-step method but not an iterative scheme.

    The following parameters of the Sweeper class will be ignored:

        - num_nodes
        - collocation_class
        - initial_guess
        - QI

    Instead, they are defined in the sense of the BDF method. For example, num_nodes defines only one node, because the method
    only computes a full time step. collocation_class defines the starting framework which includes the order of the method
    chosing the correct coefficient and the starting values. initial_guess and QI won't be used.

    Attributes:
        k_step (int): order of BDF method using k steps
        a (np.ndarray): coefficients used for k-step BDF method
        u_last (np.ndarray): contains the last k_step values for BDF
    """

    def __init__(self, params):
        """
        Initialization routine for the BDF sweeper

        Args:
            params (dict): parameters for the sweeper
        """

        # set up logger
        self.logger = logging.getLogger('sweeper')

        essential_keys = ['k_step', 'starting_values']
        for key in essential_keys:
            if key not in params:
                if key == 'starting_values':
                    params['starting_values'] = None
                else:
                    msg = 'need %s to instantiate step, only got %s' % (key, str(params.keys()))
                    self.logger.error(msg)
                    raise ParameterError(msg)

        for key in ['quad_type', 'num_nodes', 'collocation_class', 'initial_guess', 'QI']:
            if key in params:
                self.logger.warning(f'"{key}" will be ignored by BDF sweeper')

        if params['k_step'] > 1:
            raise NotImplementedError('BDF method for k_step > 1 is not implemented yet!')

        if params['k_step'] > 1:
            if params['starting_values'] is not None:
                if not isinstance(params['starting_values'], list):
                    raise ParameterError('Starting values need to be a list!')

                if len(params['starting_values']) != params['k_step'] - 1:
                    raise ParameterError(
                        f"Number of starting values needs to be equal to {params['k_step'] - 1} (value at t0 not included)!"
                    )

        starting_framework = StartingFramework(params['k_step'], params['starting_values'])
        params['collocation_class'] = starting_framework
        params['num_nodes'] = starting_framework.num_nodes

        self.params = _Pars(params)

        self.a = self.get_BDF_coefficients(params['k_step'])

        self.u0_in_ulast = False
        self.ulast = starting_framework.starting_values

        if len(self.a) != self.params.k_step + 1:
            raise ParameterError(f'Number of coefficients used for BDF needs to be equal to {self.params.k_step} + 1!')

        self.coll = starting_framework

        self.__level = None

        self.parallelizable = False

    def predict(self):
        """
        Initialises the u and the f at the level before start of the sweep of each step

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.dtype_f(init=P.init, val=0.0)

        for m in range(self.coll.num_nodes):
            L.u[m + 1] = P.dtype_u(init=P.init, val=0.0)
            L.f[m + 1] = P.dtype_f(init=P.init, val=0.0)

        # fill the attribute ulast with values from last k_step steps for BDF to start with new step
        self.get_last_values(self.params.k_step, None)

        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True
        return None

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes

        Returns:
            None
        """

        # get current level and problem description
        L = self.level
        # in the fully implicit case L.prob.eval_f() evaluates the function F(u, u', t)
        P = L.prob

        # only if the level has been touched before
        assert L.status.unlocked
        assert L.status.sweep <= 1, "BDF methods only execute one iteration!"

        def implicit_function(u):
            """
            Defines the implicit function that have to be solved by a Newton's method

            Args:
                u (np.ndarray): the sought solution

            Returns:
                f (mesh): implicit function needed for the Newton solver
            """

            u_new = P.dtype_u(P.init)
            u_new[:] = u

            u_interp = self.a[0] * u_new
            for m in range(1, self.params.k_step + 1):
                u_interp += self.a[m] * self.ulast[-m]

            f = P.eval_f(u_new, u_interp / L.dt, L.time + L.dt)
            return f

        # defines the option for the Newton solver
        options = {
            'xtol': P.newton_tol,
            'eps': 1e-7,
        }

        solve = optimize.root(
            implicit_function,
            L.u[0],
            method='hybr',
            options=options,
        )

        u_new = solve.x
        L.u[-1][:] = u_new
        L.f[-1][:] = self.a[0] * u_new
        for m in range(1, self.params.k_step + 1):
            L.f[-1][:] += self.a[m] * self.ulast[-m]
        L.f[-1][:] = L.f[-1][:] / L.dt

        # indicate presence of new values at this level
        L.status.updated = True

        self.get_last_values(self.params.k_step, u_new)

        return None

    def compute_residual(self, stage=None):
        """
        Overrides the base implementation
        Uses the absolute value of the implicit function ||F(u, du, t)|| as the residual

        Returns:
            None
        """

        L = self.level
        P = L.prob

        # compute the residual at the end of the interval
        res_norm = []
        for m in range(self.coll.num_nodes):
            res_norm.append(abs(P.eval_f(L.u[m + 1], L.f[m], L.time + m * L.dt)))

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
        For BDF, solution of the step is stored in last node
        """
        self.level.uend = self.level.u[-1]

    @classmethod
    def get_BDF_coefficients(self, k_step):
        """
        Returns the coefficients for the k-step BDF method. Assume a DAE of the form

            F(u(t), du(t), t) = 0,

        then in general, the BDF method for solving such DAEs is a method of the form:

            F(u_{n}, (a_{0} u_{n} + a_{1} u_{n-1} +..+ a_{k} u_{n-k}) / dt, t_{n}) = 0

        with u_{n} to be sought.
        For BDF, coefficients a are chosen such that k-step BDF is k_step-th order accurate.

        Args:
            k_step (int): order of BDF method, and number of previous steps used to compute the value for new step

        Returns:
            a (np.ndarray): Coefficients of BDF method
        """

        a = {
            1: np.array([1.0, -1.0]),
            2: np.array([3.0, -4.0, 1.0]),
            3: np.array([11.0, -18.0, 9.0, -2.0]),
            4: np.array([25.0, -48.0, 36.0, -16.0, 12.0]),
            5: np.array([137.0, -300.0, -300.0, -200.0, 75.0, -12.0]),
            6: np.array([147.0, -360.0, 450.0, -400.0, 225.0, -72.0, 10.0]),
        }

        return a[k_step]

    def get_last_values(self, k_step, u_new):
        """
        Updates the list ulast consisting of  values of last k-step steps which are needed for BDF to compute the
        solution for the next time step.

        Args:
            k_step (int): order of BDF method, and number of previous steps used to compute the value for new step
            u_new (np.ndarray): new element for the list needed for the next step
        """

        L = self.level

        if not self.u0_in_ulast:
            self.ulast.insert(0, L.u[0])
            self.u0_in_ulast = True

        if u_new is not None:
            self.ulast.append(u_new)
            self.ulast.pop(0)

        assert len(self.ulast) == k_step, f'Expected {k_step} values for ulast, got {len(self.ulast)}!'
