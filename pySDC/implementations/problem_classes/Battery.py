import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class battery(ptype):
    """
    Example implementing the battery drain model as in the description in the PinTSimE project

    Attributes:
        A: system matrix, representing the 2 ODEs
        t_switch: time point of the switch
        nswitches: number of switches
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['ncondensators', 'Vs', 'Rs', 'C', 'R', 'L', 'alpha', 'V_ref']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        problem_params['nvars'] = problem_params['ncondensators'] + 1

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.A = np.zeros((2, 2))
        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if u[1] <= self.params.V_ref[0] or t >= t_switch:
            f.expl[0] = self.params.Vs / self.params.L

        else:
            f.expl[0] = 0

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """
        self.A = np.zeros((2, 2))

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if rhs[1] <= self.params.V_ref[0] or t >= t_switch:
            self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L

        else:
            self.A[1, 1] = -1 / (self.params.C[0] * self.params.R)

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.params.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # cL
        me[1] = self.params.alpha * self.params.V_ref[0]  # vC

        return me

    def get_switching_info(self, u, t):
        """
        Provides information about a discrete event for one subinterval.

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            switch_detected (bool): Indicates if a switch is found or not
            m_guess (np.int): Index of collocation node inside one subinterval of where the discrete event was found
            vC_switch (list): Contains function values of switching condition (for interpolation)
        """

        switch_detected = False
        m_guess = -100

        for m in range(len(u)):
            if u[m][1] - self.params.V_ref[0] <= 0:
                switch_detected = True
                m_guess = m - 1
                break

        vC_switch = [u[m][1] - self.params.V_ref[0] for m in range(1, len(u))] if switch_detected else []

        return switch_detected, m_guess, vC_switch

    def count_switches(self):
        """
        Counts the number of switches. This function is called when a switch is found inside the range of tolerance
        (in switch_estimator.py)
        """

        self.nswitches += 1


class battery_implicit(battery):
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):

        essential_keys = ['newton_maxiter', 'newton_tol', 'ncondensators', 'Vs', 'Rs', 'C', 'R', 'L', 'alpha', 'V_ref']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        problem_params['nvars'] = problem_params['ncondensators'] + 1

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery_implicit, self).__init__(
            problem_params,
            dtype_u=dtype_u,
            dtype_f=dtype_f,
        )

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        non_f = np.zeros(2)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if u[1] <= self.params.V_ref[0] or t >= t_switch:
            self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
            non_f[0] = self.params.Vs

        else:
            self.A[1, 1] = -1 / (self.params.C[0] * self.params.R)
            non_f[0] = 0

        f[:] = self.A.dot(u) + non_f
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        u = self.dtype_u(u0)
        non_f = np.zeros(2)
        self.A = np.zeros((2, 2))

        t_switch = np.inf if self.t_switch is None else self.t_switch

        if rhs[1] <= self.params.V_ref[0] or t >= t_switch:
            self.A[0, 0] = -(self.params.Rs + self.params.R) / self.params.L
            non_f[0] = self.params.Vs

        else:
            self.A[1, 1] = -1 / (self.params.C[0] * self.params.R)
            non_f[0] = 0

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # form function g with g(u) = 0
            g = u - rhs - factor * (self.A.dot(u) + non_f)

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = np.eye(self.params.nvars) - factor * self.A

            # newton update: u1 = u0 - g/dg
            u -= np.linalg.solve(dg, g)

            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me[:] = u[:]

        return me


class battery_n_capacitors(ptype):
    """
    Example implementing the battery drain model with N capacitors, where N is an arbitrary integer greater than 0.
    Attributes:
        nswitches: number of switches
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['ncondensators', 'Vs', 'Rs', 'C', 'R', 'L', 'alpha', 'V_ref']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        n = problem_params['ncondensators']
        problem_params['nvars'] = n + 1

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery_n_capacitors, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        v = np.zeros(n + 1)
        v[0] = 1

        self.A = np.zeros((n + 1, n + 1))
        self.switch_A = {k: np.diag(-1 / (self.params.C[k] * self.params.R) * np.roll(v, k + 1)) for k in range(n)}
        self.switch_A.update({n: np.diag(-(self.params.Rs + self.params.R) / self.params.L * v)})
        self.switch_f = {k: np.zeros(n + 1) for k in range(n)}
        self.switch_f.update({n: self.params.Vs / self.params.L * v})
        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS. No Switch Estimator is used: For N = 3 there are N + 1 = 4 different states of the battery:
            1. u[1] > V_ref[0] and u[2] > V_ref[1] and u[3] > V_ref[2]    -> C1 supplies energy
            2. u[1] <= V_ref[0] and u[2] > V_ref[1] and u[3] > V_ref[2]   -> C2 supplies energy
            3. u[1] <= V_ref[0] and u[2] <= V_ref[1] and u[3] > V_ref[2]  -> C3 supplies energy
            4. u[1] <= V_ref[0] and u[2] <= V_ref[1] and u[3] <= V_ref[2] -> Vs supplies energy
        max_index is initialized to -1. List "switch" contains a True if u[k] <= V_ref[k-1] is satisfied.
            - Is no True there (i.e. max_index = -1), we are in the first case.
            - max_index = k >= 0 means we are in the (k+1)-th case.
              So, the actual RHS has key max_index-1 in the dictionary self.switch_f.
        In case of using the Switch Estimator, we count the number of switches which illustrates in which case of voltage source we are.

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if self.t_switch is not None:
            f.expl[:] = self.switch_f[self.nswitches]

        else:
            # proof all switching conditions and find largest index where it drops below V_ref
            switch = [True if u[k] <= self.params.V_ref[k - 1] else False for k in range(1, len(u))]
            max_index = max([k if switch[k] == True else -1 for k in range(len(switch))])

            if max_index == -1:
                f.expl[:] = self.switch_f[0]

            else:
                f.expl[:] = self.switch_f[max_index + 1]

        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        if self.t_switch is not None:
            self.A = self.switch_A[self.nswitches]

        else:
            # proof all switching conditions and find largest index where it drops below V_ref
            switch = [True if rhs[k] <= self.params.V_ref[k - 1] else False for k in range(1, len(rhs))]
            max_index = max([k if switch[k] == True else -1 for k in range(len(switch))])
            if max_index == -1:
                self.A = self.switch_A[0]

            else:
                self.A = self.switch_A[max_index + 1]

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.params.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # cL
        me[1:] = self.params.alpha * self.params.V_ref  # vC's
        return me

    def get_switching_info(self, u, t):
        """
        Provides information about a discrete event for one subinterval.

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            switch_detected (bool): Indicates if a switch is found or not
            m_guess (np.int): Index of collocation node inside one subinterval of where the discrete event was found
            vC_switch (list): Contains function values of switching condition (for interpolation)
        """

        switch_detected = False
        m_guess = -100
        break_flag = False

        for m in range(len(u)):
            for k in range(1, self.params.nvars):
                if u[m][k] - self.params.V_ref[k - 1] <= 0:
                    switch_detected = True
                    m_guess = m - 1
                    k_detected = k
                    break_flag = True
                    break

            if break_flag:
                break

        vC_switch = (
            [u[m][k_detected] - self.params.V_ref[k_detected - 1] for m in range(1, len(u))] if switch_detected else []
        )

        return switch_detected, m_guess, vC_switch

    def count_switches(self):
        """
        Counts the number of switches. This function is called when a switch is found inside the range of tolerance
        (in switch_estimator.py)
        """

        self.nswitches += 1


class battery_drain_charge(ptype):
    """
    Example implementing the battery drain charge model
    Attributes:
        A: system matrix, representing the 2 ODEs
        t_switch: time point of the switch (discrete event)
        nswitches: number of switches
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=imex_mesh):
        """
        Initialization routine
        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type for solution
            dtype_f: mesh data type for RHS
        """

        problem_params['nvars'] = 2

        # these parameters will be used later, so assert their existence
        essential_keys = ['pvData', 'R_pv', 'C_pv', 'R0', 'C0', 'Rline2', 'Rload', 'Vs', 'V_ref', 'alpha']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(battery_drain_charge, self).__init__(
            init=(problem_params['nvars'], None, np.dtype('float64')),
            dtype_u=dtype_u,
            dtype_f=dtype_f,
            params=problem_params,
        )

        self.A = np.zeros((2, 2))
        self.t_switch = None
        self.nswitches = 0
        self.pv_input = 0.0

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if (t * 60) % 60 == 0:
            # Time t corresponds to a full second
            t_int = int(t)
            self.pv_input = self.params.pvData[t_int + 1]
            I_pv = 

        f.expl[0] = eval_grid_voltage(t) / (self.params.C_pv * self.params.Rline2) - I_pv(t) / self.params.C_pv
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple linear solver for (I-factor*A)u = rhs
        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float): abbrev. for the local stepsize (or any ot>
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (e.g. for time-dependent BCs)
        Returns:
            dtype_u: solution as mesh
        """
        self.A = np.zeros((2, 2))

        if self.t_switch is not None:
            pass
        else:
            if rhs[1] <= self.params.V_ref:
                self.A[0, 0] = -1 / (self.params.C_pv * self.params.Rline2)

            else:
                self.A[0, 0] = - ( 1 / (self.params.C_pv * self.params.Rline2) + 1 / (self.params.C_pv * self.params.R0) )
                self.A[0, 1] = 1 / (self.params.C_pv * self.params.R0)
                self.A[1, 0] = 1 / (self.params.C0 * self.params.R0)
                self.A[1, 1] = - 1 / (self.params.C0 * self.params.R0)

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.params.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t
        Args:
            t (float): current time
        Returns:
            dtype_u: exact solution
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = self.params.alpha * self.params.V_ref  # vC_pv
        me[1] = self.params.alpha * self.params.V_ref  # vC0

        return me

    @staticmethod
    def eval_grid_voltage(t):
        """
        Function evaluates the grid voltage at time t
        Args:
            t (float): current time
        Returns:
            grid_voltage: at time t
        """

        grid_voltage = 0.0
        if t < 0.5 or t > 10.5:
            grid_voltage = self.params.Vs
        else:
            grid_voltage = 0.85 * self.params.Vs

        return grid_voltage

    def get_switching_info(self, u, t):
        """
        Provides information about a discrete event for one subinterval.
        Args:
            u (dtype_u): current values
            t (float): current time
        Returns:
            switch_detected (bool): Indicates if a switch is found or not
            m_guess (np.int): Index of where the discrete event would found
            vC_switch (list): Contains function values of switching condition (for interpolation)
        """

        switch_detected = False
        m_guess = -100

        for m in range(len(u)):
            if self.Scap == 0 and u[m][1] - self.params.V_ref <= 0:
                switch_detected = True
                m_guess = m - 1
                break

            elif self.Scap == 1 and u[m][1] - self.params.V_ref >= 0:
                switch_detected = True
                m_guess = m - 1
                break

        vC_switch = [u[m][1] - self.params.V_ref for m in range(1, len(u))] if switch_detected else []

        return switch_detected, m_guess, vC_switch

    def count_switches(self):
        """
        Counts the number of switches found.
        """

        self.nswitches += 1
