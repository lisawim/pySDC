import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.step import Step
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP, LinearTestSPPMinion
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


def is_normal_matrix(A):
    """
    Check if a matrix A is normal.
    
    A matrix A is normal if A*A = AA*
    
    Parameters
    ----------
    A : numpy.2darray
        A square matrix.
    
    Returns
    -------
    bool :
        True if A is normal, False otherwise.
    """
    A_star = np.conjugate(A.T)  # Compute the conjugate transpose of A
    return np.allclose(A @ A_star, A_star @ A)


def normality_deviation(A):
    """
    Calculate the Frobenius norm of the commutator of A.
    
    The commutator of A is A*A - AA*.
    The Frobenius norm of the commutator measures how far A is from being normal.
    
    Parameters
    ----------
    A : numpy.ndarray
        A square matrix.
    
    Returns
    -------
    float :
        The Frobenius norm of the commutator A*A - AA*.
    """
    A_star = np.conjugate(A.T)  # Compute the conjugate transpose of A
    commutator = A @ A_star - A_star @ A
    deviation = np.linalg.norm(commutator, 'fro')  # Frobenius norm
    return deviation


def testSweepEqualMatrix():
    QI = 'IE'
    sweeper = generic_implicit
    problem = LinearTestSPP
    nNodes = 3
    quad_type = 'RADAU-RIGHT'
    nSweeps = 70

    t0 = 0.0
    Tend = 1.0
    nSteps = np.array([2, 5, 10, 20, 50, 100, 200])#, 500, 1000])
    dtValues = (Tend - t0) / nSteps
    # dtValues = np.logspace(-2.5, 0.0, num=40)
    epsValues = [10 ** (-m) for m in range(2, 12)]
    spectralRadius, maxNorm = np.zeros((len(epsValues), len(dtValues))), np.zeros((len(epsValues), len(dtValues)))
    maxNormLastRow = np.zeros((len(epsValues), len(dtValues)))
    cond = np.zeros((len(epsValues), len(dtValues)))
    for d, dt in enumerate(dtValues):
        for e, eps in enumerate(epsValues):
            print(dt, eps)
            # initialize level parameters
            level_params = {
                'dt': dt,
            }

            problem_params = {
                'lintol': 1e-12,
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

            S = Step(description=description)

            L = S.levels[0]
            P = L.prob
            N = P.A.shape[0]

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

            # LHS, RHS = getSweeperMatrix(nNodes, dt, QImat, Q, P.A)

            # C = np.kron(np.identity(nNodes), np.identity(P.A.shape[0])) - dt * np.kron(Q, P.A)
            # M = np.linalg.inv(np.kron(np.identity(nNodes), np.identity(P.A.shape[0])) - dt * np.kron(QImat, P.A)).dot(C)
            # sysMatrix = np.kron(np.identity(nNodes), np.identity(P.A.shape[0])) - dt * np.kron(Q, P.A)

            # cond[e, d] = np.linalg.norm(C, np.inf) * np.linalg.norm(np.linalg.inv(C), np.inf)
            # cond[e, d] = np.linalg.norm(M, np.inf) * np.linalg.norm(np.linalg.inv(M), np.inf)
            # cond[e, d] = np.linalg.norm(sysMatrix, np.inf) * np.linalg.norm(np.linalg.inv(sysMatrix), np.inf)

            # uMatrix = np.linalg.inv(LHS).dot(u0full + RHS.dot(u0full))
            # uSweep = np.array([L.u[m].flatten() for m in range(1, nNodes + 1)]).flatten()

            # print(f"Numerical error for eps={eps}: {np.linalg.norm(uMatrix - uSweep, np.inf)}")
            # K = np.linalg.inv(LHS).dot(RHS)
            invMatrix = np.linalg.inv(np.identity(nNodes * N) - dt * np.kron(QImat, P.A))
            rhsMatrix = dt * np.kron(Q - QImat, P.A)
            K = np.matmul(invMatrix, rhsMatrix)

            is_normal = is_normal_matrix(K)
            if not is_normal:
                deviation = normality_deviation(K)
                print(f"Deviation from normality: {deviation}")

            lambdas = np.linalg.eigvals(K)
            sR = max(abs(lambdas))#np.linalg.norm(lambdas, np.inf)
            spectralRadius[e, d] = sR

            # matSweep = getManySweepsMatrix(LHS, RHS, nSweeps)
            # maxNorm[e, d] = np.linalg.norm(K, np.inf)
            # maxNorm[e, d] = np.linalg.norm(matSweep, np.inf)

            # values of iteration on last collocation node
            # n = K.shape[0]
            # eUnit = np.zeros(n)
            # eUnit[-P.A.shape[0]:] = 1  # unit vector depends on size of unknowns 
            # maxNormLastRow[e, d] = np.linalg.norm(eUnit.T.dot(K), np.inf)
            # maxNormLastRow[e, d] = np.linalg.norm(eUnit.T.dot(matSweep), np.inf)

            # print(f"Spectral radius is {sR}")

    plotQuantity(
        dtValues,
        epsValues,
        spectralRadius,
        description['problem_class'].__name__,
        'Spectral radius',
        (0.0, 1.0),#1.0,
        f'plotSpectralRadius_QI={QI}_M={nNodes}.png',
    )

    # plotQuantity(
    #     dtValues,
    #     epsValues,
    #     maxNorm,
    #     description['problem_class'].__name__,
    #     # f'Maximum norm after m={nSweeps} sweeps',
    #     f'Maximum norm',
    #     (0.0, 6.0),
    #     f'plotMaximumNorm_{nSweeps}sweeps_QI={QI}_M={nNodes}.png',
    # )

    # plotQuantity(
    #     dtValues,
    #     epsValues,
    #     maxNormLastRow,
    #     description['problem_class'].__name__,
    #     # rf'Global error transport $||e_N^T K^m||_\infty$ for m={nSweeps}',
    #     rf'Global error transport $||e_N^T K||_\infty$',
    #     (0.0, 3.0),
    #     f'plotMaximumNormLastRow_{nSweeps}sweeps_QI={QI}_M={nNodes}.png',
    # )

    # plotQuantity(
    #     dtValues,
    #     epsValues,
    #     cond,
    #     description['problem_class'].__name__,
    #     # rf'Condition number $\kappa(I-C)$',
    #     r'Condition number $\kappa(I-\Delta t QA)$',
    #     (1e0, 1e14),
    #     f'plotCondition_{nSweeps}sweeps_QI={QI}_M={nNodes}.png',
    #     plot_type='loglog',
    # )


def plotQuantity(dtValues, epsValues, quantity, prob_cls_name, quantity_label, yLim, file_name, plot_type='semilogx'):
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
        if plot_type == 'semilogx':
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
        elif plot_type == 'loglog':
            plt.loglog(
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
    if plot_type == 'loglog':
        plt.yscale('log', base=10)
    if yLim is not None:
        plt.ylim(yLim[0], yLim[-1])
        
    plt.minorticks_off()

    plt.ylabel(quantity_label, fontsize=20)
    plt.legend(frameon=False, fontsize=12, loc='upper left', ncols=2)

    plt.savefig(f"data/{prob_cls_name}/{file_name}", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    testSweepEqualMatrix()