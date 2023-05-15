import pytest
import numpy as np


@pytest.mark.base
def test_predict_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.projects.DAE.sweepers.implicit_Euler_DAE import implicit_Euler_DAE
    from pySDC.projects.DAE.run.simple_dae import get_description
    from pySDC.core.Step import step

    dt = 1e-2
    nvars = 3
    problem_class = simple_dae_1
    newton_tol = 1e-12
    hookclass = None
    sweeper = implicit_Euler_DAE
    quad_type = 'LOBATTO'
    num_nodes = 2

    description, _ = get_description(dt, nvars, problem_class, newton_tol, hookclass, sweeper, quad_type, num_nodes)
    description['sweeper_params']['initial_guess'] = 'zero'

    S = step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.1
    # compute initial value (using the exact function here)
    L.u[0] = P.u_exact(L.time)
    # call prediction function to initialise nodes
    L.sweep.predict()
    # check correct initialisation
    assert np.array_equal(L.f[0], np.zeros(3))
    for i in range(description['sweeper_params']['num_nodes']):
        assert np.array_equal(L.u[i + 1], np.zeros(3))
        assert np.array_equal(L.f[i + 1], np.zeros(3))

    # rerun check for random initialisation
    # expecting that random initialisation does not initialise to zero
    description['sweeper_params']['initial_guess'] = 'random'
    S = step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.1
    # compute initial value (using the exact function here)
    L.u[0] = P.u_exact(L.time)
    L.sweep.predict()
    assert np.array_equal(L.f[0], np.zeros(3))
    for i in range(description['sweeper_params']['num_nodes']):
        assert np.not_equal(L.u[i + 1], np.zeros(3)).any()
        assert np.not_equal(L.f[i + 1], np.zeros(3)).any()


@pytest.mark.base
def test_residual_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.projects.DAE.sweepers.implicit_Euler_DAE import implicit_Euler_DAE
    from pySDC.projects.DAE.run.simple_dae import get_description
    from pySDC.core.Step import step

    dt = 5e-2
    nvars = 3
    problem_class = simple_dae_1
    newton_tol = 1e-12
    hookclass = None
    sweeper = implicit_Euler_DAE
    quad_type = 'LOBATTO'
    num_nodes = 2

    description, _ = get_description(dt, nvars, problem_class, newton_tol, hookclass, sweeper, quad_type, num_nodes)
    description['level_params']['residual_type'] = 'last_rel'

    # last_rel residual test
    S = step(description=description)
    L = S.levels[0]
    P = L.prob
    # set reference values
    u = P.dtype_u(P.init)
    du = P.dtype_u(P.init)
    u[:] = (5, 5, 5)
    du[:] = (0, 0, 0)
    # set initial time in the status of the level
    L.status.time = 0.0
    L.u[0] = u
    # call prediction function to initialise nodes
    L.sweep.predict()
    L.sweep.compute_residual()
    # generate reference norm
    ref_norm = []
    for m in range(description['sweeper_params']['num_nodes']):
        ref_norm.append(abs(P.eval_f(u, du, L.time + L.dt * L.sweep.coll.nodes[m])))
    # check correct residual computation
    assert L.status.residual == ref_norm[-1] / abs(L.u[0]), "ERROR: incorrect norm used"

    description['level_params']['residual_type'] = 'last_abs'

    S = step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.0
    # compute initial value (using the exact function here)
    L.u[0] = u
    # call prediction function to initialise nodes
    L.sweep.predict()
    L.sweep.compute_residual()
    assert L.status.residual == ref_norm[-1], "ERROR: incorrect norm used"


@pytest.mark.base
def test_compute_end_point_main():
    from pySDC.projects.DAE.problems.simple_DAE import simple_dae_1
    from pySDC.projects.DAE.sweepers.implicit_Euler_DAE import implicit_Euler_DAE
    from pySDC.projects.DAE.run.simple_dae import get_description
    from pySDC.core.Step import step

    dt = 1e-1
    nvars = 3
    problem_class = simple_dae_1
    newton_tol = 1e-12
    hookclass = None
    sweeper = implicit_Euler_DAE
    quad_type = 'LOBATTO'
    num_nodes = 2

    description, _ = get_description(dt, nvars, problem_class, newton_tol, hookclass, sweeper, quad_type, num_nodes)
    description['sweeper_params']['initial_guess'] = 'zero'

    # test whether end point is computed correctly
    S = step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.0
    # compute initial value (using the exact function here)
    L.u[0] = P.u_exact(L.time)
    # call prediction function to initialise nodes
    L.sweep.predict()
    # compute end point
    L.sweep.compute_end_point()

    assert np.array_equal(L.uend, L.u[0]), "ERROR: end point not computed correctly"
