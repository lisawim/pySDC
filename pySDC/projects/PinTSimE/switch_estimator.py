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

        # for RK4 sweeper, sweep.coll.nodes now consists of values of ButcherTableau
        # for this reason, collocation nodes will be generated here
        coll_local = CollBase(
            num_nodes=description['sweeper_params']['num_nodes'],
            quad_type=description['sweeper_params']['quad_type'],
        )

        sweeper_type = 'RK' if RungeKutta in description['sweeper_class'].__bases__ else 'SDC'

        defaults = {
            'control_order': 100,
            'tol': description['level_params']['dt'],
            'dt_initial': description['level_params']['dt'],
            't_switch': None,
            'switch_detected_step': None,
            'sweeper_type': sweeper_type,
        }
        return {**defaults, **params}

    def setup_status_variables(self, controller, **kwargs):
        """
        Adds switching specific variables to status variables.

        Args:
            controller (pySDC.Controller): The controller
        """

        self.status = Status(['switch_detected'])

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
            print(L.sweep.coll.nodes, L.u)
            u = get_adapted_list(tableau_type, L.u)
            self.status.switch_detected, m_guess, vC_switch = L.prob.get_switching_info(u, L.time)

            t_interp = get_adapted_list(tableau_type, t_interp)
            print(t_interp, vC_switch)
            if self.status.switch_detected:
                # only find root if vc_switch[0], vC_switch[-1] have opposite signs (intermediate value theorem)
                if vC_switch[0] * vC_switch[-1] < 0:
                    self.params.t_switch = get_switch(tableau_type, t_interp, vC_switch, m_guess)

                    if L.time <= self.params.t_switch <= L.time + L.dt:
                        dt_switch = self.params.t_switch - L.time
                        if not np.isclose(self.params.t_switch - L.time, L.dt, atol=self.params.tol):
                            self.log(
                                f"Located Switch at time {self.params.t_switch:.6f} is outside the range of tol={self.params.tol:.4e}",
                                S,
                            )

                        else:
                            self.log(
                                f"Switch located at time {self.params.t_switch:.6f} inside tol={self.params.tol:.4e}", S
                            )

                            L.prob.t_switch = self.params.t_switch
                            controller.hooks[0].add_to_stats(
                                process=S.status.slot,
                                time=L.time,
                                level=L.level_index,
                                iter=0,
                                sweep=L.status.sweep,
                                type='switch',
                                value=self.params.t_switch,
                            )

                            L.prob.count_switches()
                            self.params.switch_detected_step = True

                        dt_planned = L.status.dt_new if L.status.dt_new is not None else L.params.dt

                        # when a switch is found, time step to match with switch should be preferred
                        if self.status.switch_detected:
                            L.status.dt_new = dt_switch

                        else:
                            L.status.dt_new = min([dt_planned, dt_switch])

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

        if self.params.switch_detected_step:
            if L.time + L.dt >= self.params.t_switch:
                L.status.dt_new = L.status.dt_new if L.status.dt_new is not None else self.params.dt_initial
                self.params.t_switch = None
                self.params.switch_detected_step = None

        super(SwitchEstimator, self).post_step_processing(controller, S)


def get_switch(tableau_type, t_interp, vC_switch, m_guess):
    """
    Routine to do the interpolation and root finding stuff.

    Args:
        tableau_type (str): Type of Butcher tableau used for RK sweeper (embedded or not)
        t_interp (list): collocation nodes in a step
        vC_switch (list): differences vC - V_ref at these collocation nodes
        m_guess (np.float): Index at which the difference drops below zero

    Returns:
        t_switch (np.float): time point of th switch
    """

    if tableau_type is not None:
        assume_sorted = False if tableau_type == 'TableauEmbedded' else True

    else:
        assume_sorted=True

    if tableau_type is not None and assume_sorted:
        t_interp, vC_switch = get_unique_list(t_interp, vC_switch)

    elif tableau_type is None and assume_sorted:
        pass

    else:
        raise NotImplementedError('Case of unsorted time points with duplicates is not implemented yet!')

    kind = 'quadratic' if len(t_interp) <= 3 else 'cubic'
    p = sp.interpolate.interp1d(t_interp, vC_switch, kind=kind, bounds_error=False, assume_sorted=assume_sorted)

    bracket = [t_interp[0], t_interp[1]] if m_guess == 0 else [t_interp[0], t_interp[m_guess]]
    SwitchResults = sp.optimize.root_scalar(
        p,
        method='brentq',
        bracket=bracket,
        x0=t_interp[m_guess],
        xtol=1e-10,
    )
    t_switch = SwitchResults.root

    return t_switch

def get_adapted_list(tableau_type, u):
    """
    Helper to extract the additional 0 and 1's from the nodes added in RK sweeper class.

    Args:
        tableau_type (str): Type of Butcher tableau used for RK sweeper (embedded or not)
        u (list): list of some values (could be function values or nodes of the problem)

    Returns:
        u (list): extracted list of u
    """

    # remove additional 0 and 1's from nodes added in RK sweeper class
    if tableau_type is not None:
        if tableau_type == 'TableauEmbedded':
            u = u[1:-2]

        else:
            u = u[1:-1]

    else:
        pass

    return u

def get_unique_list(t_interp, vC_switch):
    """
    Function that returns a list with unique elements. Using a RK sweeper leads to a list with not necessarily unique times.

    Args:
        t_interp (list): list of nodes from a RK sweeper (might have duplicates)
        vC_switch (list): function values at these nodes

    Returns:
        t_interp (list): list of unique times
        vC_switch (list): function values at these unique times
    """

    t_interp_copy = []
    rm_index = []
    for i, t in enumerate(t_interp):
        if t not in t_interp_copy:
            t_interp_copy.append(t)

        else:
            rm_index.append(i)

    for i in rm_index:
        vC_switch.pop(i)

    t_interp = t_interp_copy
    return t_interp, vC_switch
