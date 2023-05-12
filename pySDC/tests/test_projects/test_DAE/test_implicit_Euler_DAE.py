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
    problem_class = simple_dae
    newton_tol = 1e-12
    hookclass = None

    description, _ = get_description(dt, nvars, problem_class, newton_tol, hookclass)

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
    for i in range(sweeper_params['num_nodes']):
        assert np.array_equal(L.u[i + 1], np.zeros(3))
        assert np.array_equal(L.f[i + 1], np.zeros(3))

    # rerun check for random initialisation
    # expecting that random initialisation does not initialise to zero
    sweeper_params['initial_guess'] = 'random'
    description['sweeper_params'] = sweeper_params

    S = step(description=description)
    L = S.levels[0]
    P = L.prob
    # set initial time in the status of the level
    L.status.time = 0.1
    # compute initial value (using the exact function here)
    L.u[0] = P.u_exact(L.time)
    L.sweep.predict()
    assert np.array_equal(L.f[0], np.zeros(3))
    for i in range(sweeper_params['num_nodes']):
        assert np.not_equal(L.u[i + 1], np.zeros(3)).any()
        assert np.not_equal(L.f[i + 1], np.zeros(3)).any()
