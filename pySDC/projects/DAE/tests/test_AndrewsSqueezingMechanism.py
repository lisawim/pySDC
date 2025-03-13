import numpy as np
import pytest


@pytest.mark.base
def test_eval_f():
    from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import AndrewsSqueezingMechanismDAEConstrained

    prob = AndrewsSqueezingMechanismDAEConstrained()

    t0 = 0.0
    u0 = prob.u_exact(t0)

    f = prob.eval_f(u0, t0)

    f_alg1 = prob.M.dot(u0.alg[0 : 7]) - prob.func + prob.G.T.dot(u0.alg[7 : 13])
    f_alg2 = prob.g

    # Check right-hand side
    assert np.allclose(f.diff[0 : 7], u0.diff[7 : 14], atol=1e-14), f"RHS1 does not match! Error is {abs(f.diff[0 : 7] - u0.diff[7 : 14])}"
    assert np.allclose(f.diff[7 : 14], u0.alg[0 : 7], atol=1e-14), f"RHS2 does not match! Error is {abs(f.diff[7 : 14] - u0.alg[0 : 7])}"
    assert np.allclose(f.alg[0 : 7], f_alg1, atol=1e-14), f"RHS3 does not match! Error is {abs(f.alg[0 : 7] - f_alg1)}"
    assert np.allclose(f.alg[7 : 13], f_alg2, atol=1e-14), f"RHS4 does not match! Error is {abs(f.alg[7 : 13] - f_alg2)}"


@pytest.mark.base
def test_quantities():
    from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import AndrewsSqueezingMechanismDAEConstrained

    prob = AndrewsSqueezingMechanismDAEConstrained()

    t0 = 0.0
    u0 = prob.u_exact(t0)

    # Evaluate f to get matrices and functions
    f = prob.eval_f(u0, t0)

    assert np.allclose(prob.M - prob.M.T, 0, atol=1e-14), f"Matrix M is not symmetric!"

    assert np.allclose(prob.g, 0, atol=1e-14), f"Function g is not zero!"