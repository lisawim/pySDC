import pytest
import numpy as np


def get_EOC(dt_list, global_errors):
    """
    Function that computes the experimental order of convergence via comparing the errors of successive steps.

    Args:
        dt_list (list): list of step sizes
        global_errors (list): contains the global error for each step size considered in test_main()

    Returns:
        orders (list): list of order for each time
    """

    orders = []
    thresh = 1e-14
    # loop over two consecutive errors/dt pairs, inspired by @brownbaerchen
    for m in range(1, len(dt_list)):
        order = np.log(global_errors[m] / global_errors[m - 1]) / np.log(dt_list[m] / dt_list[m - 1])
        if global_errors[m] > thresh and global_errors[m - 1] > thresh:
            orders.append(order)

    return orders

@pytest.mark.base
def test_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1, problematic_f
    from pySDC.projects.DAE.sweepers.BDF_DAE import BDF_DAE
    from pySDC.projects.DAE.run.order_check import get_description, controller_run
    from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
    from pySDC.helpers.stats_helper import get_sorted

    hookclass = error_hook

    nvars = [3, 2]
    problem_classes = [simple_dae_1, problematic_f]
    newton_tol = 1e-12

    # for BDF methods, k_step defines the steps for multi-step as well as the order for BDF methods
    k_step = 1
    sweeper = BDF_DAE

    t0 = 0.0
    dt_list = [2 ** (-m) for m in range(2, 12)]

    global_errors = list()
    for problem_class, n in zip(problem_classes, nvars):
        for dt_item in dt_list:
            description, controller_params = get_description(
                dt_item, n, problem_class, newton_tol, hookclass, sweeper, k_step
            )

            if problem_class.__name__ == 'simple_dae_1':
                Tend = 1.0
            else:
                Tend = np.pi

            stats = controller_run(t0, Tend, controller_params, description)

            err = np.array([me[1] for me in get_sorted(stats, type='error_post_step', recomputed=False)])
            global_err_dt = max(err)
            global_errors.append(global_err_dt)

        orders = get_EOC(dt_list, global_errors)
        numerical_order = np.mean(orders)

        assert np.isclose(numerical_order, k_step, atol=1e-1), f"Expected order {k_step}, got {numerical_order}!"

        global_errors = list()


if __name__ == '__main__':
    test_main()
