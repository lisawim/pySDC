import mpmath as mp

from pySDC.projects.DAE.plotting.linearTest_analysis import plotQuantityDistribution

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def kron(A, B):
    rows_A, cols_A = A.rows, A.cols
    rows_B, cols_B = B.rows, B.cols
    result = mp.matrix(rows_A * rows_B, cols_A * cols_B)
    
    for i in range(rows_A):
        for j in range(cols_A):
            # Element-wise multiplication and placement
            for k in range(rows_B):
                for l in range(cols_B):
                    result[i * rows_B + k, j * cols_B + l] = A[i, j] * B[k, l]
    
    return result


def getIterationMatrixPrecision(dt, eps, M, Q, QI, problemName, problemType):
    r"""
    Returns iteration matrix for SDC written as Richardson iteration. Here,
    matrices will be returned in arbitrary precision using ``mpmath``.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    eps : float
        Perturbation parameter :math:`\varepsilon` of singular perturbed problems.
    M : int
        Number of quadrature nodes.
    Q : np.2darray
        Spectral quadrature matrix.
    QI : np.2darray
        Lower triangular matrix (for the preconditioner).
    problemName : str
        Name of the problem
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.

    Returns
    -------
    K : mpmath.2darray
        Iteration matrix of linear solver.
    """

    if problemName == 'LINEAR-TEST':
        lamb_diff = mp.mpf(-2.0)
        lamb_alg = mp.mpf(1.0)

        A = mp.matrix([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        Ieps = mp.eye(2)
        Ieps[-1, -1] = mp.mpf(0)

        if problemType == 'SPP':
            A[1, :] *= 1 / mp.mpf(eps)

            P = kron(mp.eye(M), mp.eye(2)) - mp.mpf(dt) * kron(QI, A)
            inv = P**-1
            K = inv * (mp.mpf(dt) * kron(Q - QI, A))

        elif problemType == 'embeddedDAE':
            P = kron(mp.eye(M), Ieps) - mp.mpf(dt) * kron(QI, A)
            inv = P**-1
            K = inv * (mp.mpf(dt) * kron(Q - QI, A))

        elif problemType == 'constrainedDAE':
            A_d = mp.matrix([[lamb_diff, lamb_alg], [0, 0]])
            A_a = mp.matrix([[0, 0], [lamb_diff, -lamb_alg]])

            P = kron(mp.eye(M), Ieps) - mp.mpf(dt) * kron(QI, A_d) + kron(mp.eye(M), A_a)
            inv = P**-1
            K = inv * (mp.mpf(dt) * kron(Q - QI, A_d))

        else:
            raise NotImplementedError(
                f"No iteration matrix for {problemType} of {problemName} problem is not implemented!"
            )

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {problemName}!")

    return K


def main():
    mp.dps = 100

    problemName = 'LINEAR-TEST'

    QI = 'LU'
    kFLEX = 5

    quadType = 'RADAU-RIGHT'
    nrows = 4
    nNodesList = [38]#range(2, 65, nrows)  # range(4, 65, nrows)
    QGenerator = Q_GENERATORS['coll']
    QIGenerator = QDELTA_GENERATORS[QI]

    dt = mp.mpf(1e-2)

    epsList = [mp.power(10, -m) for m in range(1, 12)]

    # Define a dictionary with problem types and their respective parameters
    problems = {
        'SPP': epsList,
        # 'embeddedDAE': [0.0],
        # 'constrainedDAE': [0.0],
    }

    resultsDict = {}

    for nNodes in nNodesList:
        resultsDict[nNodes] = {}

        for problemType, epsValues in problems.items():
            resultsDict[nNodes][problemType] = {}

            for eps in epsValues:
                print(f"\n...Processing for {eps=} with {nNodes} nodes...\n")

                resultsDict[nNodes][problemType][eps] = {}

                coll = QGenerator(nNodes=nNodes, nodeType='LEGENDRE', quadType=quadType)
                Q = coll.Q
                nodes = coll.nodes

                if QI == 'LU':
                    approx = QIGenerator(Q=Q)
                elif QI == 'IE':
                    approx = QIGenerator(nodes=nodes)
                else:
                    approx = QIGenerator(nNodes=nNodes, nodeType='LEGENDRE', quadType=quadType)

                k = kFLEX if QI == 'MIN-SR-FLEX' else None
                QImat = approx.getQDelta(k=k)
                QImat = mp.matrix(QImat.tolist())

                Q = mp.matrix(Q.tolist())

                K = getIterationMatrixPrecision(
                    dt=dt, eps=eps, M=nNodes, Q=Q, QI=QImat, problemName=problemName, problemType=problemType
                )

                eigvals, eigvecsMatrix = mp.eig(K.T)
                resultsDict[nNodes][problemType][eps]['eigvals'] = eigvals

                # S = mp.svd_r(K, compute_uv = False)
                # resultsDict[nNodes][problemType][eps]['singvals'] = S

    plotQuantityDistribution("eigenvalue_distribution_mp_", "Eigenvalue Distribution", problemName, problems, "eigvals", resultsDict)

    # plotQuantityDistribution("singular_value_distribution_mp_", "Singular value Distribution", problemName, problems, "singvals", resultsDict, None, (-0.1, 0.1))


if __name__ == "__main__":
    main()
