import pytest


@pytest.mark.base
def test_eval_f_computes_correct_right_hand_side():
    import numpy as np

    from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAE

    prob = LinearTestDAE(solver_type="direct")

    lamb_diff = prob.lamb_diff
    lamb_alg = prob.lamb_alg

    t0 = 0.0

    u = prob.u_exact(t0)
    du = prob.du_exact(t0)

    f_ex = du.diff[0] - (lamb_diff * u.diff[0] + lamb_alg * u.alg[0])
    g_ex = lamb_diff * u.diff[0] - lamb_alg * u.alg[0]

    f_check = prob.eval_f(u, du, t0)
    assert np.allclose(f_check.diff[0], f_ex), "Differential equation in eval_f is incorrectly computed!"
    assert np.allclose(f_check.alg[0], g_ex), "Algebraic equation in eval_f is incorrectly computed!"


@pytest.mark.base
@pytest.mark.parametrize("solver_type", ["direct", "newton"])
def test_solve_system(
    solver_type,
):  # TODO: Check for newton if only one iteration is needed to solve the LINEAR implicit system
    """Performs implicit Euler step to check if the solve_system does solve correctly."""
    import numpy as np

    from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAE

    prob = LinearTestDAE(solver_type=solver_type)

    t0 = 0.0
    dt = 1e-3

    u0 = prob.u_exact(t0)
    du_initial_guess = prob.du_exact(t0 + dt)

    rhs = u0.copy()
    factor = dt
    du1 = prob.solve_system(impl_sys=None, rhs=rhs, factor=factor, u0=du_initial_guess, t=t0)

    du1_ex = du_initial_guess.copy()

    assert np.allclose(du1.diff[0], du1_ex.diff[0]), "y is incorrectly computed!"
    assert np.allclose(du1.alg[0], du1_ex.alg[0]), "z is incorrectly computed!"
