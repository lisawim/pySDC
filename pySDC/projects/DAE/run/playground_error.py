import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded
from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAEEmbedded

from pySDC.projects.PinTSimE.battery_model import generateDescription, controllerRun

from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimatePostIter

from pySDC.helpers.stats_helper import get_sorted


def nestedListIntoSingleList(res):
    tmp_list = [item for item in res]
    err_iter = []
    for item in tmp_list:
        err_dt = [me[1] for me in item]
        if len(err_dt) > 0:
            err_iter.append([me[1] for me in item])

    err_iter = [item[0] for item in err_iter]
    return err_iter


def main():
    sweepers = [generic_implicit, genericImplicitEmbedded]

    # sweeper params
    M = 3
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'

    # parameters for convergence
    maxiter = 5

    # no special treatment
    use_A = False
    use_SE = False
    max_restarts = 0
    tol_event = 0.0
    alpha = 1.0

    # tolerance for implicit system to be solved
    newton_tol = 1e-12

    e_tol = 1e-15

    t0 = 0.0
    dt = 0.5
    Tend = t0 + dt

    for sweeper in sweepers:
        description, controller_params, controller = generateDescription(
            dt=dt,
            problem=LinearTestDAEEmbedded,
            sweeper=sweeper,
            num_nodes=M,
            quad_type=quad_type,
            QI=QI,
            hook_class=[LogEmbeddedErrorEstimatePostIter],
            use_adaptivity=use_A,
            use_switch_estimator=use_SE,
            problem_params={'newton_tol': newton_tol},
            restol=-1,
            maxiter=maxiter,
            max_restarts=max_restarts,
            tol_event=tol_event,
            alpha=alpha,
            residual_type=None,
            e_tol=e_tol,
        )

        stats, _ = controllerRun(
            description=description,
            controller_params=controller_params,
            controller=controller,
            t0=t0,
            Tend=t0+dt,
            exact_event_time_avail=None,
        )

        type = 'error_embedded_estimate_post_iteration'
        quantityIter = [get_sorted(stats, iter=k, type=type, sortby='time') for k in range(1, maxiter + 1)]

        quantityIter = nestedListIntoSingleList(quantityIter)

        print(quantityIter)

        assert quantityIter[-1] >= e_tol, f"Larger value for {sweeper.__name__} of error expected, got {quantityIter[-1]}"


if __name__ == "__main__":
    main()
