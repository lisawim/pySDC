import numpy as np
import scipy as sp

from pySDC.core.Collocation import CollBase
from pySDC.core.ConvergenceController import ConvergenceController, Status
from pySDC.implementations.convergence_controller_classes.check_convergence import CheckConvergence

import pySDC.helpers.plot_helper as plt_helper


class SwitchEstimator(ConvergenceController):
    """
    Class to predict the time point of the event and setting a new step size. For the first time, this is a nonMPI version,
    because a MPI version is not yet developed.
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
            'nodes': coll.nodes,
            'count': 0,
        }
        return {**defaults, **params}

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
        Determine a new step size when an event is found such that the event occurs at the time step.

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
                # print(t_interp, state_function)
                # when the state function is already close to zero the event is already resolved well
                if abs(state_function[-1]) <= self.params.tol or abs(state_function[0]) <= self.params.tol:
                    self.log("Is already close enough to one of the end point!", S)
                    self.log_event_time(
                        controller.hooks[0], S.status.slot, L.time, L.level_index, L.status.sweep, t_interp[-1]
                    )
                    L.prob.count_switches()
                    self.status.is_zero = True

                # intermediate value theorem states that a root is contained in current step
                if state_function[0] * state_function[-1] < 0 and self.status.is_zero is None:
                    self.status.t_switch = self.get_switch(t_interp, state_function, m_guess, self.params.count, self.params.dt_FD)
                    self.params.count += 1
                    if L.time < self.status.t_switch < L.time + L.dt:
                        dt_switch = self.status.t_switch - L.time

                        if (
                            abs(self.status.t_switch - L.time) <= self.params.tol
                            or abs((L.time + L.dt) - self.status.t_switch) <= self.params.tol
                        ):
                            self.log(f"Switch located at time {self.status.t_switch:.12f}", S)
                            L.prob.t_switch = self.status.t_switch
                            self.log_event_time(
                                controller.hooks[0],
                                S.status.slot,
                                L.time,
                                L.level_index,
                                L.status.sweep,
                                self.status.t_switch,
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
                        # print('New time step size: {}'.format(L.status.dt_new))
                    else:
                        # event occurs on L.time or L.time + L.dt; no restart necessary
                        boundary = 'left boundary' if self.status.t_switch == L.time else 'right boundary'
                        self.log(f"Estimated switch {self.status.t_switch:.12f} occurs at {boundary}", S)

                        self.log_event_time(
                            controller.hooks[0],
                            S.status.slot,
                            L.time,
                            L.level_index,
                            L.status.sweep,
                            self.status.t_switch,
                        )
                        L.prob.count_switches()
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

        super().post_step_processing(controller, S, **kwargs)

    @staticmethod
    def log_event_time(controller_hooks, process, time, level, sweep, t_switch):
        """
        Logs the event time of an event satisfying an appropriate criterion, e.g., event is already resolved well,
        event time satisfies tolerance.

        Parameters
        ----------
        controller_hooks : pySDC.Controller.hooks
            Controller with access to the hooks.
        process : int
            Process for logging.
        time : float
            Time at which the event time is logged (denotes the current step).
        level : int
            Level at which event is found.
        sweep : int
            Denotes the number of sweep.
        t_switch : float
            Event time founded by switch estimation.
        """

        controller_hooks.add_to_stats(
            process=process,
            time=time,
            level=level,
            iter=0,
            sweep=sweep,
            type='switch',
            value=t_switch,
        )

    @staticmethod
    def get_switch(t_interp, state_function, m_guess, count, dt_FD):
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

        def LagrangePolynomial(t, ti, i):
            """
            Computes the i-th Lagrange Polynomial.

            Parameters
            ----------
            t : float
                Time to evaluate the polynomial.
            ti : list
                Data points.
            i : int
                Index of the Lagrange Polynomial (the index which is skipped in computation).
            """
            poly = 1
            for j in range(len(ti)):
                if j == i:
                    continue
                poly *= (t - ti[j]) / (ti[i] - ti[j])
            return poly
        
        def p(t, ti, yi):
            """
            Computes the interpolation polynomial based on Lagrange interpolation.

            Parameters
            ----------
            t : float
                Time to evaluate the polynomial.
            ti : list
                Data points.
            yi : list
                Values on these data points.
            """
            sum = 0
            for j in range(len(ti)):
                sum += yi[j] * LagrangePolynomial(t, ti, j)
            return sum

        # Interpolator = sp.interpolate.BarycentricInterpolator(t_interp, state_function)
        
        # def p(t, dt_FD):
        #     """
        #     Simplifies the call of the interpolant.

        #     Parameters
        #     ----------
        #     t : float
        #         Time t at which the interpolant is called.

        #     Returns
        #     -------
        #     p(t) : float
        #         The value of the interpolated function at time t.
        #     """
        #     return Interpolator.__call__(t)

        def fprime(t, ti, yi, dt_FD):
            """
            Computes the derivative of the scalar interpolant using finite differences.

            Parameters
            ----------
            t : float
                Time where the derivatives is computed.

            Returns
            -------
            dp : float
                Derivative of interpolation p at time t.
            """
            # dt = 1e-6
            # dp = (p(t + dt_FD) - p(t - dt_FD)) / (2 * dt_FD)  # Dc
            # dp = (p(t + dt_FD) - p(t)) / dt_FD  # Dplus
            # dp = (2 * p(t + dt_FD) + 3 * p(t) - 6 * p(t - dt_FD) + p(t - 2 * dt_FD)) / (6 * dt_FD)  # D3
            dp = (3 * p(t, ti, yi) - 4 * p(t - dt_FD, ti, yi) + p(t - 2 * dt_FD, ti, yi)) / (2 * dt_FD)  # D2
            return dp
        
        # p_int = sp.interpolate.interp1d(t_interp, state_function, 'cubic', bounds_error=False)
        # print('Get_switch:', state_function)
        newton_tol, newton_maxiter = 1e-12, 100
        # t_switch = newton(t_interp[m_guess], p, fprime, dt_FD, newton_tol, newton_maxiter)
        #root = sp.optimize.root(p, t_interp[m_guess], method='hybr', tol=newton_tol)
        results = sp.optimize.root_scalar(p, args=(dt_FD), method='newton', fprime=fprime, xtol=newton_tol, x0=t_interp[m_guess])
        t_switch = results.root
        # print(results)
        #t_switch = root.x[0]

        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
        if t_interp[0] <= 0.5 * np.arcsinh(100) <= t_interp[-1]:
            ax.axvline(
                x=0.5 * np.arcsinh(100),
                linestyle='--',
                linewidth=0.9,
                color='k',
                label='Exact event time',)
        ax.axvline(
            x=t_switch,
            linestyle='--',
            linewidth=0.9,
            color='g',
            label='Founded event time',)
        ax.plot(t_interp, state_function, label='h')
        ax.plot(t_interp, p(t_interp, dt_FD), label='p')
        ax.legend(loc='upper right')
        fig.savefig('data/interpolation/state_function_interp_{}.png'.format(count), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)

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


def newton(x0, p, fprime, dt_FD, newton_tol, newton_maxiter):
    """
    Newton's method fo find the root of interpolant p.

    Parameters
    ----------
    x0 : float
        Initial guess.
    p : callable
        Interpolated function where Newton's method is applied at.
    fprime : callable
        Approximated erivative of p using finite differences.
    newton_tol : float
        Tolerance for termination.
    newton_maxiter : int
        Maximum of iterations the method should execute.

    Returns
    -------
    root : float
        Root of function p.
    """

    n = 0
    while n < newton_maxiter:
        if abs(p(x0)) < newton_tol or np.isnan(p(x0)) and np.isnan(fprime(x0, dt_FD)):
            break
        # print(n, x0, p(x0), fprime(x0))
        x0 -= 1.0 / fprime(x0, dt_FD) * p(x0)

        n += 1

    root = x0
    msg = "Newton's method took {} iterations".format(n)
    print(msg)

    return root


class Lagrange(object):
    """
    This class computes an interpolation polynomial based on Lagrange interpolation.
    """
    def __init__(ti, yi):
        """Initialzation routine"""