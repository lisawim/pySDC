from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE

from pySDC.projects.PinTSimE.battery_model import generate_description, get_recomputed
from pySDC.projects.PinTSimE.discontinuous_test_ODE import controller_run
from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostIter, LogGlobalErrorPostStep
from pySDC.implementations.hooks.log_restarts import LogRestarts
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook, error_hook


class LogEvent(hooks):
    """
    Logs the data for the discontinuous test DAE problem containing one discrete event.
    Note that this logging data is dependent from the problem itself.
    """

    def post_step(self, step, level_number):
        super(LogEvent, self).post_step(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='state_function',
            value=2 * L.uend[0] - 100,
        )


def main():
    """
    Function that executes the main stuff in this file.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = [LogSolution, LogGlobalErrorPostStep, LogEvent, error_hook, LogRestarts, approx_solution_hook]

    problem_class = DiscontinuousTestDAE

    sweeper = fully_implicit_DAE
    nnodes = 3
    quad_type = 'RADAU-RIGHT'
    maxiter = 45

    use_detection = True
    max_restarts = 1
    tol_event = 1e-10
    dt_FD = 1e-10
    alpha = 1.0

    t0 = 3.0
    Tend = 5.4

    dt = 5e-2

    restol = 1e-13
    recomputed = False

    problem_params = dict()
    problem_params['newton_tol'] = 1e-6

    description, controller_params = generate_description(
        dt,
        problem_class,
        sweeper,
        nnodes,
        quad_type,
        hookclass,
        False,
        use_detection,
        problem_params,
        restol,
        maxiter,
        max_restarts,
        tol_event,
        dt_FD,
        alpha,
    )

    stats, t_switch_exact = controller_run(t0, Tend, controller_params, description)


if __name__ == "__main__":
    main()