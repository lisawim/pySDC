import numpy as np
from mpi4py import MPI

from pySDC.core.convergence_controller import ConvergenceController
from pySDC.implementations.convergence_controller_classes.store_uold import StoreUOld

from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta


class EstimateEmbeddedErrorMPIGrouped(ConvergenceController):

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
        sweeper_type = 'SDC'
        sweeper_mro_names = [me.__name__ for me in description['sweeper_class'].__mro__]
        if RungeKutta in description['sweeper_class'].__mro__:
            sweeper_type = 'RK'
        elif 'SweeperMPIGrouped' in sweeper_mro_names:
            sweeper_type = 'MPI-Grouped'
        return {
            "control_order": -79,
            "sweeper_type": sweeper_type,
            "rel_error": False,
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
        if RungeKutta not in description["sweeper_class"].__bases__:
            controller.add_convergence_controller(StoreUOld, description=description)

        from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate

        controller.add_hook(LogEmbeddedErrorEstimate)
        return None

    def estimate_embedded_error_mpi_grouped(self, L):
        """
        Embedded error estimate for grouped MPI sweeper.
        Uses max over collocation nodes of |uold - u| (optionally relative),
        computed locally over owned nodes and reduced globally with MPI.MAX.
        """
        comm = L.sweep.comm

        # grouped sweeper provides local_nodes; fallback: assume 1:1 mapping
        local_nodes = getattr(L.sweep, "local_nodes", [comm.rank])

        loc_max = 0.0
        for m in local_nodes:
            num = abs(L.uold[m + 1] - L.u[m + 1])

            if self.params.rel_error:
                denom = abs(L.u[m + 1])
                # be robust: denom may be dtype-like; convert to float
                denom_f = float(denom)
                val = float(num) / denom_f if denom_f != 0.0 else float(num)
            else:
                val = float(num)

            loc_max = max(loc_max, val)

        # global max so every rank gets the same increment
        return comm.allreduce(loc_max, op=MPI.MAX)

    def estimate_embedded_error_serial(self, L):
        """
        Estimate embedded error depending on sweeper type.
        Returns a scalar (float-like) error estimate.
        """
        if self.params.sweeper_type == "RK":
            L.sweep.compute_end_point()
            if self.params.rel_error:
                return float(abs(L.uend - L.sweep.u_secondary) / abs(L.uend))
            else:
                return float(abs(L.uend - L.sweep.u_secondary))

        elif self.params.sweeper_type == "SDC":
            # order rises by one between sweeps
            if self.params.rel_error:
                return float(abs(L.uold[-1] - L.u[-1]) / abs(L.u[-1]))
            else:
                return float(abs(L.uold[-1] - L.u[-1]))

        elif self.params.sweeper_type == "MPI-Grouped":
            return float(self.estimate_embedded_error_mpi_grouped(L))

        else:
            raise NotImplementedError(
                f"Don't know how to estimate embedded error for sweeper type \"{self.params.sweeper_type}\""
            )

    def setup_status_variables(self, controller, **kwargs):
        """
        Add the embedded error to the level status

        Args:
            controller (pySDC.Controller): The controller
        """
        self.add_status_variable_to_level('error_embedded_estimate')
        self.add_status_variable_to_level('increment')

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
                L.status.error_embedded_estimate = max([self.estimate_embedded_error_serial(L), np.finfo(float).eps])
                L.status.increment = L.status.error_embedded_estimate * 1
                self.debug(f'L.status.error_embedded_estimate={L.status.error_embedded_estimate:.5e}', S)

        return None
