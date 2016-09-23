import numpy as np
import scipy.sparse as sp

from examples.tutorial.step_1.HeatEquation_1D_FD import heat1d
from pySDC.datatype_classes.mesh import mesh

from pySDC.collocation_classes.gauss_radau_right import CollGaussRadau_Right


def solve_collocation_problem(prob, coll, dt):
    """
    Routine to build and solve the linear collocation problem

    Args:
        prob: a problem instance
        coll: a collocation instance
        dt: time-step size

    Return:
        the analytic error of the solved collocation problem
    """

    # shrink collocation matrix: first line and column deals with initial value, not needed here
    Q = coll.Qmat[1:, 1:]

    # build system matrix M of collocation problem
    M = sp.eye(prob.nvars * coll.num_nodes) - dt * sp.kron(Q, prob.A)

    # get initial value at t0 = 0
    u0 = prob.u_exact(t=0)
    # fill in u0-vector as right-hand side for the collocation problem
    u0_coll = np.kron(np.ones(coll.num_nodes), u0.values)
    # get exact solution at Tend = dt
    uend = prob.u_exact(t=dt)

    # solve collocation problem directly
    u_coll = sp.linalg.spsolve(M, u0_coll)

    # compute error
    err = np.linalg.norm(u_coll[-prob.nvars:] - uend.values, np.inf)

    return err


if __name__ == "__main__":
    """
    A simple test program to create and solve a collocation problem directly
    """

    # initialize problem parameters
    problem_params = {}
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 4  # frequency for the test value
    problem_params['nvars'] = 1023  # number of degrees of freedom

    # instantiate problem
    prob = heat1d(problem_params=problem_params, dtype_u=mesh, dtype_f=mesh)

    # instantiate collocation class, relative to the time interval [0,1]
    coll = CollGaussRadau_Right(num_nodes=3, tleft=0, tright=1)

    # set time-step size (warning: the collocation matrices are relative to [0,1], see above)
    dt = 0.1

    # solve collocation problem
    err = solve_collocation_problem(prob=prob, coll=coll, dt=dt)

    print(err, err <= 4E-04)

    assert err <= 4E-04

