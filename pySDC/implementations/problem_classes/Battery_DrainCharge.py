import numpy as np

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class battery_drain_charge(ptype):
    """
    Example implementing the battery drain charge model
    Attributes:
        A: system matrix, representing the 2 ODEs
        t_switch: time point of the switch (discrete event)
        Scap: state of the switch
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
        essential_keys = ['I_pv', 'R_pv', 'C_pv', 'R0', 'C0', 'Rline2', 'Rload', 'Vs', 'V_ref', 'alpha']
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
        self.count_switches = 0
        self.Scap = 0

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

        vC_switch = []
        if switch_detected:
            for m in range(1, len(u)):
               vC_switch.append(u[m][1] - self.params.V_ref)

        return switch_detected, m_guess, vC_switch

    def set_counter(self):
        """
        Counts the number of switches found.
        """

        self.count_switches += 1
