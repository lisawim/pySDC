import numpy as np

from pySDC.projects.DAE import getColor, getIterationMatrix, getLabel, getMarker, Plotter

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def getMatrixToBeInversed(dt, eps, nNodes, problemName, problemType, QI):
    if problemName == 'LINEAR-TEST':
        lamb_diff = -2.0
        lamb_alg = 1.0

        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        if problemType == 'SPP':
            A[1, :] *= 1 / eps

            C = np.kron(np.identity(nNodes), np.identity(2)) - np.kron(QI, A)
        else:
            raise NotImplementedError("No matrix implemented!")

        return C


def main():
    problemName = 'LINEAR-TEST'

    QI = 'MIN-SR-FLEX'
    kFLEX = 1

    quadType = 'RADAU-RIGHT'
    nNodesList = range(2, 65, 4)
    QGenerator = Q_GENERATORS['coll']
    QIGenerator = QDELTA_GENERATORS[QI]

    dt = 1e-2

    epsList = [10 ** (-m) for m in range(1, 12)]

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
                resultsDict[nNodes][problemType][eps] = {}

                coll = QGenerator(nNodes=nNodes, nodeType='LEGENDRE', quadType=quadType)
                Q = coll.Q
                print(f"{coll.nodes=}")
                approx = (
                    QIGenerator(Q=Q)
                    if QI == 'LU'
                    else QIGenerator(nNodes=nNodes, nodeType='LEGENDRE', quadType=quadType)
                )
                # k = kFLEX if QI == 'MIN-SR-FLEX' else None
                k = nNodes if QI == 'MIN-SR-FLEX' else None
                QImat = approx.getQDelta(k=k)

                C = getMatrixToBeInversed(dt, eps, nNodes, problemName, problemType, QImat)

                resultsDict[nNodes][problemType][eps]['max_norm'] = np.linalg.norm(C, np.inf)

    plotNorm(problemName, problems, QI, resultsDict)


def plotNorm(problemName, problems, QI, resultsDict):
    NormPlotter = Plotter(nrows=1, ncols=1, orientation='horizontal', figsize=(10, 10))

    norm = {problemType: {eps: [] for eps in epsValues} for problemType, epsValues in problems.items()}
    nNodesList = sorted(resultsDict.keys())

    for nNodes in nNodesList:
        for problemType, epsValues in problems.items():
            for eps in epsValues:
                results = resultsDict[nNodes][problemType][eps]
                norm[problemType][eps].append(results['max_norm'])

    # Plot the data
    for problemType, epsValues in problems.items():
        for i, eps in enumerate(epsValues):
            color, res, problemLabel = getColor(problemType, i), getMarker(problemType), getLabel(problemType)
            label = rf'$\varepsilon=${eps}' + problemLabel
            print(f"For {problemType} with {eps=}: {norm[problemType][eps]}")
            NormPlotter.plot(
                nNodesList,
                norm[problemType][eps],
                subplot_index=0,
                color=color,
                marker=res['marker'],
                markersize=res['markersize'],
                label=label,
                plot_type='semilogy',
            )

    NormPlotter.set_xticks(nNodesList, subplot_index=0)
    NormPlotter.set_xlabel('Number of Nodes', subplot_index=0)

    NormPlotter.set_legend(subplot_index=0, loc='best')
    NormPlotter.set_grid(True, subplot_index=0)

    NormPlotter.set_ylabel('Maximum norm', subplot_index=0)
    # NormPlotter.set_ylim((0.0, 1.0), subplot_index=0)

    filename = "data" + "/" + f"{problemName}" + "/" + f"max_norm_matrix_{QI=}.png"
    NormPlotter.save(filename)


if __name__ == "__main__":
    main()
