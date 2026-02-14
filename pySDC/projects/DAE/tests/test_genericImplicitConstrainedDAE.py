import numpy as np
import pytest


def make_setup(dt, hook_class, num_nodes, problem, sweeper_class, QI=None):
    """Make setup for run."""

    level_params = {"dt": dt, "restol": -1, "e_tol": 1e-13}

    problem_params = {"solver_type": "direct"}

    step_params = {"maxiter": 10}

    controller_params = {"logger_level": 30, "hook_class": hook_class}

    sweeper_params = {
        "quad_type": "RADAU-RIGHT",
        "num_nodes": num_nodes,
        "QI": "LU" if QI is None else QI,
        "initial_guess": "spread",
    }

    description = {
        "problem_class": problem,
        "problem_params": problem_params,
        "sweeper_class": sweeper_class,
        "sweeper_params": sweeper_params,
        "level_params": level_params,
        "step_params": step_params,
    }
    return description, controller_params


@pytest.mark.base
def test_integrate_method_integrates_correctly():
    r"""
    In this test the integrate method of the sweeper is tested.
    """

    from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained
    from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAEConstrained
    from pySDC.core.level import Level

    level_params = {"dt": 0.1}

    problem_params = {"solver_type": "direct"}

    sweeper_params = {
        "quad_type": "RADAU-RIGHT",
        "num_nodes": 2,
        "QI": "IE",
        "initial_guess": "spread",
    }

    lvl = Level(
        problem_class=LinearTestDAEConstrained,
        problem_params=problem_params,
        sweeper_class=genericImplicitConstrained,
        sweeper_params=sweeper_params,
        level_params=level_params,
        level_index=0,
    )
    prob = lvl.prob

    sdc_c = genericImplicitConstrained(params=sweeper_params, level=lvl)

    lvl.status.time = 1.0
    lvl.u[0] = prob.u_exact(lvl.time)

    sdc_c.predict()

    for j in range(1, lvl.sweep.coll.num_nodes + 1):
        assert lvl.f[j] is not None

    # integrate RHS over all collocation nodes
    int_ref = []
    for m in range(1, lvl.sweep.coll.num_nodes + 1):
        int_ref.append(prob.dtype_u(prob.init, val=0.0))
        for j in range(1, lvl.sweep.coll.num_nodes + 1):
            int_ref[-1].diff[:] += lvl.dt * lvl.sweep.coll.Qmat[m, j] * lvl.f[j].diff[:]

    int = sdc_c.integrate()
    assert len(int) == lvl.sweep.coll.num_nodes
    for m in range(lvl.sweep.coll.num_nodes):
        assert np.allclose(int[m].diff, int_ref[m].diff)
        assert np.allclose(int[m].alg, int_ref[m].alg)
        assert np.allclose(int[m].alg, 0.0)


@pytest.mark.base
@pytest.mark.parametrize("residual_type", ["full_abs", "last_abs", "full_rel", "last_rel", "else"])
def test_residual_is_correctly_computed(residual_type):
    r"""
    In this test the compute_residual method of the sweeper is tested.
    """

    from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained
    from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAEConstrained
    from pySDC.core.level import Level
    from pySDC.core.errors import ParameterError

    level_params = {"dt": 0.1, "residual_type": residual_type}

    problem_params = {"solver_type": "direct"}

    sweeper_params = {
        "quad_type": "RADAU-RIGHT",
        "num_nodes": 2,
        "QI": "IE",
        "initial_guess": "spread",
    }

    lvl = Level(
        problem_class=LinearTestDAEConstrained,
        problem_params=problem_params,
        sweeper_class=genericImplicitConstrained,
        sweeper_params=sweeper_params,
        level_params=level_params,
        level_index=0,
    )
    prob = lvl.prob

    sdc_c = genericImplicitConstrained(params=sweeper_params, level=lvl)

    lvl.status.time = 1.0
    lvl.u[0] = prob.u_exact(lvl.time)

    sdc_c.predict()

    if residual_type == "else":
        with pytest.raises(ParameterError):
            lvl.sweep.compute_residual()
    else:
        lvl.sweep.compute_residual()

    res_norm_ref, res_norm_ref_full = [], []
    res_ref = lvl.sweep.integrate()
    for m in range(lvl.sweep.coll.num_nodes):
        res_ref[m].diff[:] += lvl.u[0].diff[:] - lvl.u[m + 1].diff[:]

        res_norm_ref.append(abs(res_ref[m]))

        res_norm_ref_full.append(prob.dtype_u(prob.init, val=0.0))
        res_norm_ref_full[-1] += res_ref[m]

    if residual_type == "full_abs":
        assert lvl.status.residual == max(res_norm_ref)
    elif residual_type == "last_abs":
        assert lvl.status.residual == res_norm_ref[-1]
    elif residual_type == "full_rel":
        assert lvl.status.residual == max(res_norm_ref) / abs(lvl.u[0])
    elif residual_type == "last_rel":
        assert lvl.status.residual == res_norm_ref[-1] / abs(lvl.u[0])

    for m in range(lvl.sweep.coll.num_nodes):
        assert np.allclose(res_norm_ref_full[m].alg, 0.0)


@pytest.mark.base
@pytest.mark.parametrize("num_nodes", [2, 3])
def test_compare_results(num_nodes):
    r"""
    Test checks whether the results of the ``genericImplicitConstrainedDAE`` sweeper
    matches with the ``SemiImplicitDAE`` version.
    """

    from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE
    from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained
    from pySDC.projects.DAE.problems.linearTestDAE import SemiImplicitLinearTestDAE, LinearTestDAEConstrained
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    dt = 0.01
    description_sdc_c, controller_params_sdc_c = make_setup(
        dt=dt,
        hook_class=[],
        num_nodes=num_nodes,
        problem=LinearTestDAEConstrained,
        sweeper_class=genericImplicitConstrained,
    )

    description_si_sdc, controller_params_si_sdc = make_setup(
        dt=dt,
        hook_class=[],
        num_nodes=num_nodes,
        problem=SemiImplicitLinearTestDAE,
        sweeper_class=SemiImplicitDAE,
    )

    t0 = 0.0
    Tend = t0 + dt

    controller_sdc_c = controller_nonMPI(
        num_procs=1, controller_params=controller_params_sdc_c, description=description_sdc_c
    )
    controller_si_sdc = controller_nonMPI(
        num_procs=1, controller_params=controller_params_si_sdc, description=description_si_sdc
    )

    P = controller_sdc_c.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend_sdc_c, _ = controller_sdc_c.run(u0=uinit, t0=t0, Tend=Tend)
    uend_si_sdc, _ = controller_si_sdc.run(u0=uinit, t0=t0, Tend=Tend)

    assert np.allclose(uend_sdc_c, uend_si_sdc), "Values at end time does not match!"

    err_sdc_c, err_si_sdc = abs(uend_sdc_c - P.u_exact(Tend)), abs(uend_si_sdc - P.u_exact(Tend))
    assert np.allclose(err_sdc_c, err_si_sdc), "Errors does not match!"


@pytest.mark.base
@pytest.mark.parametrize("QI", ["IE", "LU"])
@pytest.mark.parametrize("num_nodes", [2, 3])
@pytest.mark.parametrize("case", [0, 1])
def test_order_accuracy(case, num_nodes, QI):
    r"""
    In this test, the order of accuracy of the ``SemiImplicitDAE`` sweeper is tested for an index-1 DAE
    and an index-2 DAE of semi-explicit form.
    """

    from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained
    from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAEConstrained
    from pySDC.projects.DAE.problems.simpleDAE import SimpleDAE
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.projects.DAE.misc.hooksDAE import (
        LogGlobalErrorPostStepDifferentialVariable,
        LogGlobalErrorPostStepAlgebraicVariable,
    )
    from pySDC.helpers.stats_helper import get_sorted

    t0 = 0.0
    Tend = 1.0

    ref_order_diff = 2 * num_nodes - 1
    ref_order_alg = ref_order_diff

    dt_list = np.logspace(-1.7, -1.0, num=5)

    errors_diff, errors_alg = np.zeros(len(dt_list)), np.zeros(len(dt_list))
    for i, dt in enumerate(dt_list):
        hook_class = [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable]
        description, controller_params = make_setup(
            dt=dt,
            hook_class=hook_class,
            num_nodes=num_nodes,
            problem=LinearTestDAEConstrained,
            sweeper_class=genericImplicitConstrained,
            QI=QI,
        )

        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)

        _, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

        errors_diff[i] = max(
            np.array(get_sorted(stats, type="e_global_differential_post_step", sortby="time", recomputed=False))[:, 1]
        )
        errors_alg[i] = max(
            np.array(get_sorted(stats, type="e_global_algebraic_post_step", sortby="time", recomputed=False))[:, 1]
        )

    order_diff = np.mean(
        [
            np.log(errors_diff[i] / errors_diff[i - 1]) / np.log(dt_list[i] / dt_list[i - 1])
            for i in range(1, len(dt_list))
        ]
    )
    order_alg = np.mean(
        [
            np.log(errors_alg[i] / errors_alg[i - 1]) / np.log(dt_list[i] / dt_list[i - 1])
            for i in range(1, len(dt_list))
        ]
    )

    assert np.isclose(
        order_diff, ref_order_diff, atol=1e0
    ), f"Expected order {ref_order_diff} in differential variable, got {order_diff}"
    assert np.isclose(
        order_alg, ref_order_alg, atol=1e0
    ), f"Expected order {ref_order_alg} in algebraic variable, got {order_alg}"
