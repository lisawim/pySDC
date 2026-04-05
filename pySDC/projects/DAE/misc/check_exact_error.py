import numpy as np

from pySDC.core.convergence_controller import ConvergenceController


class CheckExactError(ConvergenceController):
    r"""
    Checks the exact error in each iteration. If numerical solution achieves an certain accuracy, step is converged.
    Note that method ``u_exact`` needs to be implemented for a problem.
    """

    def setup(self, controller, params, description, **kwargs):
        """
        Add a default value for control order to the parameters and check if we are using a Runge-Kutta sweeper

        Args:
            controller (pySDC.Controller): The controller
            params (dict): Parameters for the convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            dict: Updated parameters
        """
        sweeper_type = "SDC"
        if "SweeperMPI" in [me.__name__ for me in description["sweeper_class"].__mro__]:
            sweeper_type = "MPI"
        return {
            "control_order": -78,
            "sweeper_type": sweeper_type,
            **super().setup(controller, params, description, **kwargs),
        }

    def dependencies(self, controller, description, **kwargs):
        """
        Load the convergence controller that stores the solution of the last sweep unless we are doing Runge-Kutta.
        Add the hook for recording the error.

        Args:
            controller (pySDC.Controller): The controller
            description (dict): The description object used to instantiate the controller

        Returns:
            None
        """
        from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate

        controller.add_hook(LogEmbeddedErrorEstimate)
        return None

    def setup_status_variables(self, controller, **kwargs):
        """
        Add the embedded error to the level status

        Args:
            controller (pySDC.Controller): The controller
        """
        self.add_status_variable_to_level('error_embedded_estimate')
        self.add_status_variable_to_level('increment')
        self.add_status_variable_to_level('exact_error')

    def check_exact_error_serial(self, L):
        """
        Estimate the serial embedded error, which may need to be modified for a parallel estimate.

        Depending on the type of sweeper, the lower order solution is stored in a different place.

        Args:
            L (pySDC.level): The level

        Returns:
            dtype_u: The embedded error estimate
        """
        prob = L.prob
        uex = prob.u_exact(L.time + L.dt)
        if self.params.sweeper_type == "RK":
            L.sweep.compute_end_point()
            return abs(uex - L.uend)
        if self.params.sweeper_type == "SDC":
            # order rises by one between sweeps
            return abs(uex - L.u[-1])
        elif self.params.sweeper_type == "MPI":
            comm = L.sweep.comm
            return comm.bcast(abs(uex - L.u[comm.rank + 1]), root=comm.size - 1)
        else:
            raise NotImplementedError(f"Don't know how to check exact error for sweeper type \
\"{self.params.sweeper_type}\"")

    def post_iteration_processing(self, controller, S, **kwargs):
        """
        Estimate the local error here.

        If you are doing MSSDC, this is the global error within the block in Gauss-Seidel mode.
        In Jacobi mode, I haven't thought about what this is.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        if S.status.iter > 0 or self.params.sweeper_type == "RK":
            for L in S.levels:
                L.status.exact_error = max([self.check_exact_error_serial(L), np.finfo(float).eps])
                L.status.error_embedded_estimate = L.status.exact_error * 1
                L.status.increment = L.status.error_embedded_estimate * 1
                self.debug(f'L.status.error_embedded_estimate={L.status.error_embedded_estimate:.5e}', S)

        return None