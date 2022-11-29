import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pathlib import Path

from pySDC.core.Step import step
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.PinTSimE.piline_model import setup_mpl

import pySDC.helpers.plot_helper as plt_helper



def compute_stability(K=5, M=3):
    """
    Routine to compute the stability domains of different configurations of a fully-implicit SDC
    Returns:
        float: lambda
        int: number of collocation nodes
        int: number of iterations
        numpy.ndarray: stability numbers
    """
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)

    X, Y = np.meshgrid(x, y)

    lambdas = X + 1j * Y

    # initialize problem parameters
    problem_params = dict()
    problem_params['lambdas'] = np.array([0.0])  # max(lambdas)
    problem_params['u0'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = M

    sweeper_params['do_coll_update'] = True

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1.0

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = testequation0d
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = dict()

    # create folders for different nodes
    Path("M={}".format(sweeper_params['num_nodes'])).mkdir(parents=True, exist_ok=True)

    # instantiate step
    S = step(description=description)

    L = S.levels[0]

    Q = L.sweep.coll.Qmat[1:, 1:]
    nnodes = L.sweep.coll.num_nodes
    dt = L.params.dt

    N, M = np.shape(lambdas)[0], np.shape(lambdas)[1]
    stab = np.zeros((N, M), dtype='complex')
    for i in range(N):
        for j in range(M):
            lambda1 = lambdas[i, j]

            if K != 0:
                Mat_sweep = L.sweep.get_scalar_problems_manysweep_mat(nsweeps=K, lambdas=lambda1)

            else:
                Mat_sweep = np.linalg.inv(np.eye(nnodes) - dt * lambda1 * Q)

            if L.sweep.params.do_coll_update:
                stab_fh = 1.0 + lambda1 * L.sweep.coll.weights.dot(Mat_sweep.dot(np.ones(nnodes)))

            else:
                q = np.zeros(nnodes)
                q[nnodes - 1] = 1.0
                stab_fh = q.dot(Mat_sweep.dot(np.ones(nnodes)))

            stab[i, j] = stab_fh

    plot_stability(x, y, lambdas, sweeper_params['num_nodes'], K, stab)

    return sweeper_params['num_nodes'], K, stab

def plot_stability(x, y, lambdas, num_nodes, K, stab):
    """
    Plotting routine of the stability domains
    Args:
        lambdas (float): lambda
        num_nodes (int): number of collocation nodes
        K (int): number of iterations
        stab (numpy.ndarray): stability numbers
    """

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.plot([min(x), max(x)], [0, 0], 'k--', linewidth=0.5)
    ax.plot([0, 0], [min(y), max(y)], 'k--', linewidth=0.5)
    #ax.contour(x, y, np.absolute(stab), [0.5], colors='g', linestyles='dashed', linewidths=0.5)
    CS = ax.contour(x, y, np.absolute(stab), [0.0, 0.25, 0.5, 0.75, 1.0], cmap='rainbow', linestyles='dashed', linewidths=0.5)
    zc = CS.collections[4]
    plt.setp(zc, linewidth=0.5)
    ax.clabel(CS, fontsize=9, inline=1)
    ax.set_xlabel(r'$Re(\lambda)$', fontsize=8)
    ax.set_ylabel(r'$Im(\lambda)$', fontsize=8)
    fig.savefig("M={}/plot_stability-M{}-K{}.png".format(num_nodes, num_nodes, K), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    compute_stability()
