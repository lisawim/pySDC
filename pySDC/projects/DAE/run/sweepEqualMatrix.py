import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.Step import step
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPPMinion
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit


def getSweeperMatrix(nNodes, dt, QI, Q, A):
    LHS = np.kron(np.identity(nNodes), np.identity(A.shape[0])) - dt * np.kron(QI, A)
    RHS = dt * np.kron(Q - QI, A)
    return LHS, RHS


def getManySweepsMatrix(LHS, RHS, nSweeps):
    r"""
    For a scalar problem, K sweeps of SDC can be written in matrix form.

    Parameters
    ----------
    nSweeps : int
        Number of sweeps.

    Parameters
    ----------
    matSweep : np.2darray
        Matrix to the power ``nSweeps``.
    """

    Pinv = np.linalg.inv(LHS)
    matSweep = np.linalg.matrix_power(Pinv.dot(RHS), nSweeps)
    for k in range(0, nSweeps):
        matSweep += np.linalg.matrix_power(Pinv.dot(RHS), k).dot(Pinv)
    return matSweep


def testSweepEqualMatrix():
    QI = 'IE'
    sweeper = generic_implicit
    problem = LinearTestSPPMinion
    nNodes = 3
    quad_type = 'RADAU-RIGHT'
    nSweeps = 70

    dtValues = np.logspace(-5.0, 0.0, num=40)
    epsValues = [10 ** (-m) for m in range(1, 11)]
    spectralRadius, maxNorm = np.zeros((len(epsValues), len(dtValues))), np.zeros((len(epsValues), len(dtValues)))
    maxNormLastRow = np.zeros((len(epsValues), len(dtValues)))
    for d, dt in enumerate(dtValues):
        for e, eps in enumerate(epsValues):
            print(dt, eps)
            # initialize level parameters
            level_params = {
                'dt': dt,
            }

            problem_params = {
                'newton_tol': 1e-14,
                'eps': eps,
            }

            # initialize sweeper parameters
            sweeper_params = {
                'quad_type': quad_type,
                'num_nodes': nNodes,
                'QI': QI,
                'initial_guess': 'spread',
            }

            step_params = {
                'maxiter': 1,
            }

            # fill description dictionary for easy step instantiation
            description = {
                'problem_class': problem,
                'problem_params': problem_params,
                'sweeper_class': sweeper,
                'sweeper_params': sweeper_params,
                'level_params': level_params,
                'step_params': step_params,
            }

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            L.status.time = 0.0
            u0 = P.u_exact(L.status.time)
            S.init_step(u0)

            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]

            # S.levels[0].sweep.predict()
            # u0full = np.array([L.u[m].flatten() for m in range(1, nNodes + 1)]).flatten()
            # L.sweep.update_nodes()

            # nodes = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(nNodes)]
            # uexfull = np.array([P.u_exact(t).flatten() for t in nodes]).flatten()

            LHS, RHS = getSweeperMatrix(nNodes, dt, QImat, Q, P.A)

            # uMatrix = np.linalg.inv(LHS).dot(u0full + RHS.dot(u0full))
            # uSweep = np.array([L.u[m].flatten() for m in range(1, nNodes + 1)]).flatten()

            # print(f"Numerical error for eps={eps}: {np.linalg.norm(uMatrix - uSweep, np.inf)}")

            K = np.linalg.inv(LHS).dot(RHS)
            lambdas = np.linalg.eigvals(K)
            sR = np.linalg.norm(lambdas, np.inf)
            spectralRadius[e, d] = sR

            # matSweep = getManySweepsMatrix(LHS, RHS, nSweeps)
            maxNorm[e, d] = np.linalg.norm(K, np.inf)
            # maxNorm[e, d] = np.linalg.norm(matSweep, np.inf)

            # values of iteration on last collocation node
            n = K.shape[0]
            eUnit = np.zeros(n)
            eUnit[-P.A.shape[0]:] = 1  # unit vector depends on size of unknowns 
            maxNormLastRow[e, d] = np.linalg.norm(eUnit.T.dot(K), np.inf)
            # maxNormLastRow[e, d] = np.linalg.norm(eUnit.T.dot(matSweep), np.inf)

            # print(f"Spectral radius is {sR}")

    plotQuantity(
        dtValues,
        epsValues,
        spectralRadius,
        description['problem_class'].__name__,
        'Spectral radius',
        1.5,
        f'plotSpectralRadius_QI={QI}_M={nNodes}.png',
    )

    plotQuantity(
        dtValues,
        epsValues,
        maxNorm,
        description['problem_class'].__name__,
        # f'Maximum norm after m={nSweeps} sweeps',
        f'Maximum norm',
        6.0,
        f'plotMaximumNorm_{nSweeps}sweeps_QI={QI}_M={nNodes}.png',
    )

    plotQuantity(
        dtValues,
        epsValues,
        maxNormLastRow,
        description['problem_class'].__name__,
        # rf'Global error transport $||e_N^T K^m||_\infty$ for m={nSweeps}',
        rf'Global error transport $||e_N^T K||_\infty$',
        3.0,
        f'plotMaximumNormLastRow_{nSweeps}sweeps_QI={QI}_M={nNodes}.png',
    )


def plotQuantity(dtValues, epsValues, quantity, prob_cls_name, quantity_label, yLimMax, file_name):
    colors = [
        'lightsalmon',
        'lightcoral',
        'indianred',
        'firebrick',
        'brown',
        'maroon',
        'lightgray',
        'darkgray',
        'gray',
        'dimgray',
    ]
    plt.figure(figsize=(9.5, 9.5))
    for e, eps in enumerate(epsValues):
        plt.semilogx(
            dtValues,
            quantity[e, :],
            color=colors[e],
            marker='*',
            markersize=10.0,
            linewidth=4.0,
            solid_capstyle='round',
            label=rf"$\varepsilon=${eps}",
        )

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(r'$\Delta t$', fontsize=20)
    plt.ylim(0, yLimMax)
    plt.minorticks_off()

    plt.ylabel(quantity_label, fontsize=20)
    plt.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    plt.savefig(f"data/{prob_cls_name}/{file_name}", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    testSweepEqualMatrix()