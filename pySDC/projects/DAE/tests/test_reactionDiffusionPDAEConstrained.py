import pytest


def source_f_ex(prob, t):
    u = prob.u_ex(t, 0, 0)
    u_t  = prob.u_ex(t, 0, 1)
    u_xx = prob.u_ex(t, 2, 0)
    w_x   = prob.w_ex(t, 1) 
    return u_t - u_xx - u * w_x

def source_g_ex(prob, t):
    v = prob.v_ex(t, 0, 0)
    v_t  = prob.v_ex(t, 0, 1)
    v_xx = prob.v_ex(t, 2, 0)
    w_x   = prob.w_ex(t, 1) 
    return v_t - v_xx + v * w_x


@pytest.mark.base
@pytest.mark.parametrize("spectral", [True, False])
def test_correct_FFT_discretization_source_terms(spectral):
    """Checking whether the source terms are correctly computed in spectral space."""

    import numpy as np

    from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAEConstrained

    prob = ReactionDiffusionPDAEConstrained(bc="periodic", nvars=8, spectral=spectral)

    t0 = 0.0

    f_ex = source_f_ex(prob=prob, t=t0)
    f_hat = prob.src_f_spectral(t0, n=prob.nvars)

    assert np.allclose(np.fft.irfft(f_hat), f_ex, atol=1e-13), "Computation of source f is incorrect!"

    g_ex = source_g_ex(prob=prob, t=t0)
    g_hat = prob.src_g_spectral(t0, n=prob.nvars)

    assert np.allclose(np.fft.irfft(g_hat), g_ex, atol=1e-13), "Computation of source g is incorrect!"


@pytest.mark.base
@pytest.mark.parametrize("spectral", [True, False])
def test_eval_f(spectral):
    import numpy as np

    from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAEConstrained

    prob = ReactionDiffusionPDAEConstrained(bc="periodic", nvars=8, spectral=spectral)

    t0 = 0.0

    u = prob.u_ex(t0, 0, 0)
    u_xx = prob.u_ex(t0, 2, 0)

    v = prob.v_ex(t0, 0, 0)
    v_xx = prob.v_ex(t0, 2, 0)

    w = prob.w_ex(t0, 0)
    w_x = prob.w_ex(t0, 1)
    w_xx = prob.w_ex(t0, 2)

    f_ex = source_f_ex(prob=prob, t=t0)
    g_ex = source_g_ex(prob=prob, t=t0)

    f1_ex = u_xx + u * w_x + f_ex
    f2_ex = v_xx - v * w_x + g_ex
    f3_ex = -u - v - w_xx

    u_check = prob.dtype_u(prob.init)
    u_check.diff[: prob.nvars], u_check.diff[prob.nvars :] = u, v
    u_check.alg[: prob.nvars] = w

    f_check = prob.eval_f(u_check, t0)
    assert np.allclose(f_check.diff[: prob.nvars], f1_ex), "First equation of rhs is incorrectly computed!"
    assert np.allclose(f_check.diff[prob.nvars :], f2_ex), "Second equation of rhs is incorrectly computed!"
    assert np.allclose(f_check.alg[: prob.nvars], f3_ex), "Third equation of rhs is incorrectly computed!"


@pytest.mark.base
@pytest.mark.parametrize("nvars", [8, 64, 256])
@pytest.mark.parametrize("spectral", [True, False])
def test_solve_system(nvars, spectral):
    import numpy as np

    from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAEConstrained

    prob = ReactionDiffusionPDAEConstrained(bc="periodic", nvars=nvars, spectral=spectral)

    t0 = 0.0
    dt = 1e-4

    u0 = prob.u_exact(t0)

    rhs = u0.copy()
    factor = dt
    u1 = prob.solve_system(rhs, factor, u0, t0)

    u1_ex = prob.u_exact(t0 + dt)

    assert np.allclose(u1.diff[: nvars], u1_ex.diff[: nvars]), "u is incorrectly computed!"
    assert np.allclose(u1.diff[nvars :], u1_ex.diff[nvars :]), "v is incorrectly computed!"
    assert np.allclose(u1.alg[: nvars], u1_ex.alg[: nvars]), "w is incorrectly computed!"
