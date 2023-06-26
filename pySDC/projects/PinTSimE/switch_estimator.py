import numpy as np
import scipy as sp

from pySDC.core.Collocation import CollBase
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence


class SwitchEstimator(ConvergenceController):
    """
    Class to predict the time point of the switch and setting a new step size. For the first time, this is a
    nonMPI version, because a MPI version is not yet developed.
    """

    def setup(self, controller, params, description):
        """
        Function sets default variables to handle with the switch at the beginning.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        params : dict
            The parameters passed for this specific convergence controller.
        description : dict
            The description object used to instantiate the controller.

        Returns
        -------
        convergence_controller_params : dict
            The updated params dictionary.
        """

        # for RK4 sweeper, sweep.coll.nodes now consists of values of ButcherTableau
        # for this reason, collocation nodes will be generated here
        coll = CollBase(
            num_nodes=description['sweeper_params']['num_nodes'],
            quad_type=description['sweeper_params']['quad_type'],
        )

        defaults = {
            'control_order': 100,
            'tol': 1e-11,
            'nodes': coll.nodes,
        }
        convergence_controller_params = {**defaults, **params}
        return convergence_controller_params

    def setup_status_variables(self, controller, **kwargs):
        """
        Adds switching specific variables to status variables.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        """

        self.status = Status(['is_zero', 'switch_detected', 't_switch'])

    def reset_status_variables(self, controller, **kwargs):
        """
        Resets status variables.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        """

        self.setup_status_variables(controller, **kwargs)

    def get_new_step_size(self, controller, S, **kwargs):
        """
        Determine a new step size when a switch is found such that the switch happens at the time step.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        L = S.levels[0]

        if CheckConvergence.check_convergence(S):
            self.status.switch_detected, m_guess, state_function = L.prob.get_switching_info(L.u, L.time)

            if self.status.switch_detected:
                t_interp = [L.time + L.dt * self.params.nodes[m] for m in range(len(self.params.nodes))]
                t_interp, state_function = self.adapt_interpolation_info(
                    L.time, L.sweep.coll.left_is_node, t_interp, state_function
                )

                # when the state function is already close to zero the event is already resolved well
                if all(np.isclose(state_function[0], np.zeros(len(t_interp)), atol=1e-11)):
                   self.status.is_zero = True

                # intermediate value theorem states that a root is contained in current step
                if state_function[0] * state_function[-1] < 0 and self.status.is_zero is None:
                    self.status.t_switch = self.get_switch(t_interp, state_function, m_guess)

                    if L.time < self.status.t_switch < L.time + L.dt:
                        dt_switch = self.status.t_switch - L.time

                        if np.isclose(L.time, self.status.t_switch, atol=self.params.tol) or np.isclose(
                            L.time + L.dt, self.status.t_switch, atol=self.params.tol
                        ):
                            self.log(f"Switch located at time {self.status.t_switch:.12f}", S)
                            L.prob.t_switch = self.status.t_switch
                            controller.hooks[0].add_to_stats(
                                process=S.status.slot,
                                time=L.time,
                                level=L.level_index,
                                iter=0,
                                sweep=L.status.sweep,
                                type='switch',
                                value=self.status.t_switch,
                            )

                            L.prob.count_switches()

                        else:
                            self.log(f"Located Switch at time {self.status.t_switch:.12f} is outside the range", S)

                        # when an event is found, step size matching with this event should be preferred
                        dt_planned = L.status.dt_new if L.status.dt_new is not None else L.params.dt
                        if self.status.switch_detected:
                            L.status.dt_new = dt_switch
                        else:
                            L.status.dt_new = min([dt_planned, dt_switch])

                    else:
                        # event occurs on L.time or L.time + L.dt; no restart necessary
                        self.status.switch_detected = False

                else:  # intermediate value theorem is not satisfied
                    self.status.switch_detected = False

    def determine_restart(self, controller, S, **kwargs):
        """
        Check if the step needs to be restarted due to a predicting switch.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        if self.status.switch_detected:
            S.status.restart = True
            S.status.force_done = True

        super().determine_restart(controller, S, **kwargs)

    def post_step_processing(self, controller, S, **kwargs):
        """
        After a step is done, some variables will be prepared for predicting a possibly new switch.
        If no Adaptivity is used, the next time step will be set as the default one from the front end.

        Parameters
        ----------
        controller : pySDC.Controller
            The controller doing all the stuff in a computation.
        S : pySDC.Step
            The current step.
        """

        L = S.levels[0]

        if self.status.t_switch is None:
            L.status.dt_new = L.status.dt_new if L.status.dt_new is not None else L.params.dt_initial
            L.prob.t_switch = None

        super().post_step_processing(controller, S, **kwargs)

    @staticmethod
    def get_switch(t_interp, state_function, m_guess):
        """
        Routine to do the interpolation and root finding stuff.

        Parameters
        ----------
        t_interp : list
            Collocation nodes in a step.
        state_function : list
            Contains values of state function at these collocation nodes.
        m_guess : float
            Index at which the difference drops below zero.

        Returns
        -------
        t_switch : float
           Time point of the founded switch.
        """

        kind = 'quadratic' if len(t_interp) <= 3 else 'cubic'
        p = sp.interpolate.interp1d(t_interp, state_function, kind=kind, bounds_error=False)

        SwitchResults = sp.optimize.root_scalar(
            p,
            method='brentq',
            bracket=[t_interp[0], t_interp[-1]],
            x0=t_interp[m_guess],
            xtol=1e-14,
        )
        t_switch = SwitchResults.root

        return t_switch

    @staticmethod
    def adapt_interpolation_info(t, left_is_node, t_interp, state_function):
        """
        Adapts the x- and y-axis for interpolation. For SDC, it is proven whether the left boundary is a
        collocation node or not. In case it is, the first entry of the state function has to be removed,
        because it would otherwise contain double values on starting time and the first node. Otherwise,
        starting time L.time has to be added to t_interp to also take this value in the interpolation
        into account.

        Parameters
        ----------
        t : float
            Starting time of the step.
        left_is_node : bool
            Indicates whether the left boundary is a collocation node or not.
        t_interp : list
            x-values for interpolation containing collocation nodes.
        state_function : list
            y-values for interpolation containing values of state function.

        Returns
        -------
        t_interp : list
            Adapted x-values for interpolation containing collocation nodes.
        state_function : list
            Adapted y-values for interpolation containing values of state function.
        """

        if not left_is_node:
            t_interp.insert(0, t)
        else:
            del state_function[0]

        return t_interp, state_function
