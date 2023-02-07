import numpy as np
import scipy as sp

from pySDC.core.Collocation import CollBase
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta


class SwitchEstimator(ConvergenceController):
    """
    Class to predict the time point of the switch and setting a new step size

    For the first time, this is a nonMPI version, because a MPI version is not yet developed.
    """

    def setup(self, controller, params, description):
        """
        Function sets default variables to handle with the switch at the beginning.

        Args:
            controller (pySDC.Controller): The controller
            params (dict): The params passed for this specific convergence controller
            description (dict): The description object used to instantiate the controller

        Returns:
            (dict): The updated params dictionary
        """

        sweeper_type = 'RK' if RungeKutta in description['sweeper_class'].__bases__ else 'SDC'

        defaults = {
            'control_order': 100,
            'tol': description['level_params']['dt'],
            'sweeper_type': sweeper_type,
        }
        return {**defaults, **params}

    def setup_status_variables(self, controller, **kwargs):
        """
        Adds switching specific variables to status variables.

        Args:
            controller (pySDC.Controller): The controller
        """

        self.status = Status(['switch_detected', 't_switch'])

    def reset_status_variables(self, controller, **kwargs):
        """
        Resets status variables.

        Args:
            controller (pySDC.Controller): The controller
        """

        self.setup_status_variables(controller, **kwargs)

    def get_new_step_size(self, controller, S):
        """
        Determine a new step size when a switch is found such that the switch happens at the time step.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        L = S.levels[0]

        if self.params.sweeper_type == 'RK':
            tableau_type = 'TableauEmbedded' if L.sweep.coll.__class__.__name__ == 'ButcherTableauEmbedded' else 'Tableau'

        else:
            tableau_type = None

        if S.status.iter == S.params.maxiter:
            t_interp = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(len(L.sweep.coll.nodes))]
            t_interp, u = self.get_extracted_values(tableau_type, t_interp, L.u)

            self.status.switch_detected, m_guess, vC_switch = L.prob.get_switching_info(u, L.time)
            if self.status.switch_detected:
                # only find root if vc_switch[0], vC_switch[-1] have opposite signs (intermediate value theorem)
                if vC_switch[0] * vC_switch[-1] < 0:
                    self.status.t_switch = self.get_switch(t_interp, vC_switch, m_guess)

                    if L.time <= self.status.t_switch <= L.time + L.dt:
                        dt_switch = self.status.t_switch - L.time
                        if not np.isclose(self.status.t_switch - L.time, L.dt, atol=self.params.tol):
                            self.log(
                                f"Located Switch at time {self.status.t_switch:.6f} is outside the range of tol={self.params.tol:.4e}",
                                S,
                            )

                        else:
                            self.log(
                                f"Switch located at time {self.status.t_switch:.6f} inside tol={self.params.tol:.4e}", S
                            )

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

                        dt_planned = L.status.dt_new if L.status.dt_new is not None else L.params.dt

                        # when a switch is found, time step to match with switch should be preferred
                        if self.status.switch_detected:
                            L.status.dt_new = dt_switch

                        else:
                            L.status.dt_new = min([dt_planned, dt_switch])
                        print(L.status.dt_new)
                    else:
                        self.status.switch_detected = False

                else:
                    self.status.switch_detected = False

    def determine_restart(self, controller, S):
        """
        Check if the step needs to be restarted due to a predicting switch.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        if self.status.switch_detected:
            S.status.restart = True
            S.status.force_done = True

        super(SwitchEstimator, self).determine_restart(controller, S)

    def post_step_processing(self, controller, S):
        """
        After a step is done, some variables will be prepared for predicting a possibly new switch.
        If no Adaptivity is used, the next time step will be set as the default one from the front end.

        Args:
            controller (pySDC.Controller): The controller
            S (pySDC.Step): The current step

        Returns:
            None
        """

        L = S.levels[0]

        if self.status.t_switch is None:
            L.status.dt_new = L.status.dt_new if L.status.dt_new is not None else L.params.dt_initial

        super(SwitchEstimator, self).post_step_processing(controller, S)

    @staticmethod
    def get_switch(t_interp, vC_switch, m_guess):
        """
        Routine to do the interpolation and root finding stuff.

        Args:
            t_interp (list): collocation nodes in a step
            vC_switch (list): differences vC - V_ref at these collocation nodes
            m_guess (np.float): Index at which the difference drops below zero

        Returns:
            t_switch (np.float): time point of th switch
        """

        kind = 'quadratic' if len(t_interp) <= 3 else 'cubic'
        p = sp.interpolate.interp1d(t_interp, vC_switch, kind=kind, bounds_error=False)

        SwitchResults = sp.optimize.root_scalar(
            p,
            method='brentq',
            bracket=[t_interp[0], t_interp[m_guess]],
            x0=t_interp[m_guess],
            xtol=1e-10,
        )
        t_switch = SwitchResults.root

        return t_switch

    @staticmethod
    def get_extracted_values(tableau_type, t_interp, u):
        """
        Helper to extract values needed for switch estimation using different sweepers. For RK sweepers, additional
        0 and 1's needs to be removed for the interpolation. When SDC is used as sweeper, only u-value at left should be
        removed, which corresponds to value at first collocation node (Gauss-Lobatto nodes only, needs to be adapted).

        For using RK sweepers, t_interp will be sorted and duplicates will be removed.

        Args:
            tableau_type (str): Type of Butcher Tablaeu (differs between embedded, not embedded, None)
            t_interp (list): list of nodes
            u (dtype_u): function values

        Returns:
            t_interp (list): list of nodes (extracted)
            u (dtype_u): function values (extracted)
        """

        if tableau_type is not None:
            # save values on collocation update
            updates = [t_interp[-1], u[-1]]
            if tableau_type == 'TableauEmbedded':
                t_interp = t_interp[1:-2]
                u = u[1:-2]

            elif tableau_type == 'Tableau':
                t_interp = t_interp[1:-1]
                u = u[1:-1]

            # remove duplicates from nodes
            t_interp_unique = []
            rm_index = []
            for i, t in enumerate(t_interp):
                if t not in t_interp_unique:
                    t_interp_unique.append(t)

                else:
                    rm_index.append(i)

            for i in rm_index:
                u.pop(i)

            # sort nodes and their values
            ind = [item[0] for item in sorted(enumerate(t_interp_unique), key=lambda i:i[1])]
            t_interp = [t_interp_unique[i] for i in ind]
            u = [u[i] for i in ind]

            # remove last values on node 1..
            t_interp.pop(-1)
            u.pop(-1)

            # ..and replace it with update
            t_interp.append(updates[0])
            u.append(updates[1])

        else:
            u = u[1:]

        return t_interp, u
