import numpy as np

from pySDC.projects.DAE import computeNormalityDeviation, getColor, getCollocationMatrix, getIterationMatrix, getLabel, getMarker, Plotter

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def main():
    problemName = 'LINEAR-TEST'

    QI = 'MIN-SR-S'
    kFLEX = 5

    quadType = 'RADAU-RIGHT'
    nrows = 4
    nNodesList = range(2, 26, nrows)  # range(4, 65, nrows)
    QGenerator = Q_GENERATORS['coll']
    QIGenerator = QDELTA_GENERATORS[QI]

    dt = 5e-1

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
                nodes = coll.nodes

                if QI == 'LU':
                    approx = QIGenerator(Q=Q)
                elif QI == 'IE':
                    approx = QIGenerator(nodes=nodes)
                else:
                    approx = QIGenerator(nNodes=nNodes, nodeType='LEGENDRE', quadType=quadType)

                k = kFLEX if QI == 'MIN-SR-FLEX' else None
                QImat = approx.getQDelta(k=k)

                K = getIterationMatrix(
                    dt=dt, eps=eps, M=nNodes, Q=Q, QI=QImat, problemName=problemName, problemType=problemType
                )

                C = getCollocationMatrix(
                    dt=dt, eps=eps, M=nNodes, Q=QImat, problemName=problemName, problemType=problemType
                )

                deviation = computeNormalityDeviation(K)
                print(f"\n{QI}: For {problemType} with {eps=} using {nNodes} nodes -- Normality of matrix: {deviation}\n")

                resultsDict[nNodes][problemType][eps]['spectral_radius'] = max(abs(np.linalg.eigvals(K)))

                condition_number = np.linalg.norm(C, 2) * np.linalg.norm(np.linalg.inv(C), 2)
                resultsDict[nNodes][problemType][eps]['condition'] = condition_number

                U, S, Vh = np.linalg.svd(np.matmul(K.T, K), full_matrices=True)
                resultsDict[nNodes][problemType][eps]['spectral_radius_sv'] = max(abs(S))

                resultsDict[nNodes][problemType][eps]['max_norm'] = np.linalg.norm(K, np.inf)

                eigvals, eigvecs = np.linalg.eig(K.T)
                resultsDict[nNodes][problemType][eps]['eigvals'] = eigvals
                resultsDict[nNodes][problemType][eps]['eigvecs'] = eigvecs

                resultsDict[nNodes][problemType][eps]['singvals'] = S

    # plotQuantity(
    #     f"condition_{QI=}.png", problemName, problems, QI, 'condition', resultsDict, 'Condition number', None
    # )

    plotQuantity(
        f"spectral_radius_{QI=}_{dt=}.png", problemName, problems, QI, 'spectral_radius', resultsDict, 'Spectral radius', (0.0, 1.8)
    )

    # plotQuantity(
    #     f"spectral_radius_SV_{QI=}.png", problemName, problems, QI, 'spectral_radius_sv', resultsDict, 'Spectral radius SV', None
    # )

    # plotQuantity(
    #     f"max_norm_{QI=}.png", problemName, problems, QI, 'max_norm', resultsDict, 'Maximum norm'
    # )

    # plotQuantityDistribution(f"eigenvalue_distribution_{QI=}_", "Eigenvalue distribution", problemName, problems, "eigvals", resultsDict, (-2.0, 2.0), (-2.0, 2.0))

    # plotQuantityDistribution(f"singular_value_distribution_{QI=}_", "Singular value distribution", problemName, problems, "singvals", resultsDict, None, None)

    # plotEigvals(nrows, problemName, problems, resultsDict)


def main2():
    problemName = 'LINEAR-TEST'

    QIList = ['IE', 'LU', 'MIN-SR-S']
    kFLEX = 5

    quadType = 'RADAU-RIGHT'
    nrows = 4
    nNodesList = range(2, 65, nrows)  # range(4, 65, nrows)
    QGenerator = Q_GENERATORS['coll']

    dt = 1e-2

    # Define a dictionary with problem types and their respective parameters
    problems = {'embeddedDAE': [0.0], 'constrainedDAE': [0.0]}

    linestyle = {"IE": "solid", "LU": "dashed", "MIN-SR-S": "dotted"}

    SpecRadiusPlotter = Plotter(nrows=1, ncols=1, orientation='horizontal', figsize=(10, 10))
    for QI in QIList:
        QIGenerator = QDELTA_GENERATORS[QI]

        for problemType, epsValues in problems.items():

            for i, eps in enumerate(epsValues):
                specRadius = []

                for nNodes in nNodesList:
                    print(f"{QI=} {nNodes=} {problemType=}")
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

                    K = getIterationMatrix(
                        dt=dt, eps=eps, M=nNodes, Q=Q, QI=QImat, problemName=problemName, problemType=problemType
                    )

                    specRadius.append(max(abs(np.linalg.eigvals(K))))

            color, res, problemLabel = getColor(problemType, i), getMarker(problemType), getLabel(problemType)
            SpecRadiusPlotter.plot(
                nNodesList,
                specRadius,
                subplot_index=0,
                color=color,
                marker=res['marker'],
                markersize=res['markersize'],
                linestyle=linestyle[QI],
                label=f"{problemLabel}" if QI == "IE" else None,
            )

            # Dummy
            if problemType == "embeddedDAE":
                SpecRadiusPlotter.plot(
                    2,
                    10,
                    subplot_index=0,
                    color='k',
                    marker=res['marker'],
                    markersize=res['markersize'],
                    linestyle=linestyle[QI],
                    label=f"{QI}",
                )


    SpecRadiusPlotter.set_xticks(nNodesList, subplot_index=0)
    SpecRadiusPlotter.set_xlabel('Number of nodes', subplot_index=0)

    SpecRadiusPlotter.set_legend(subplot_index=0, loc='best')
    SpecRadiusPlotter.set_grid(True, subplot_index=0)

    SpecRadiusPlotter.set_ylabel("Spectral radius", subplot_index=0)
    SpecRadiusPlotter.set_ylim((0.0, 1.2), subplot_index=0)

    filename = "data" + "/" + f"{problemName}" + "/" + f"spectral_radius_DAE_compare_QI.png"
    SpecRadiusPlotter.save(filename)


def plotQuantity(fileNamePlot, problemName, problems, QI, quantityLabel, resultsDict, yLabel, yLim=None):
    SpecRadiusPlotter = Plotter(nrows=1, ncols=1, orientation='horizontal', figsize=(10, 10))

    specRadius = {problemType: {eps: [] for eps in epsValues} for problemType, epsValues in problems.items()}
    nNodesList = sorted(resultsDict.keys())

    for nNodes in nNodesList:
        for problemType, epsValues in problems.items():
            for eps in epsValues:
                results = resultsDict[nNodes][problemType][eps]
                specRadius[problemType][eps].append(results[quantityLabel])

    # Plot the data
    for problemType, epsValues in problems.items():
        for i, eps in enumerate(epsValues):
            color, res, problemLabel = getColor(problemType, i), getMarker(problemType), getLabel(problemType)
            label = rf'$\varepsilon=${eps}' + problemLabel

            SpecRadiusPlotter.plot(
                nNodesList,
                specRadius[problemType][eps],
                subplot_index=0,
                color=color,
                marker=res['marker'],
                markersize=res['markersize'],
                label=label,
            )

    SpecRadiusPlotter.set_xticks(nNodesList, subplot_index=0)
    SpecRadiusPlotter.set_xlabel('Number of nodes', subplot_index=0)

    SpecRadiusPlotter.set_legend(subplot_index=0, loc='upper left')
    SpecRadiusPlotter.set_grid(True, subplot_index=0)

    SpecRadiusPlotter.set_ylabel(yLabel, subplot_index=0)
    if yLim is not None:
        SpecRadiusPlotter.set_ylim(yLim, subplot_index=0)

    filename = "data" + "/" + f"{problemName}" + "/" + fileNamePlot
    SpecRadiusPlotter.save(filename)


def plotQuantityDistribution(fileNamePlot, plotTitle, problemName, problems, quantityLabel, resultsDict, xLim=None, yLim=None):
    nNodesList = sorted(resultsDict.keys())

    a = np.cos(np.linspace(0, 2 * np.pi, 200))
    b = np.sin(np.linspace(0, 2 * np.pi, 200))

    figsize = (10, 10) if quantityLabel == 'eigvals' else (10, 7)

    for nNodes in nNodesList:
        DistributionPlotter = Plotter(nrows=1, ncols=1, figsize=figsize)

        for problemType, epsValues in problems.items():
            sIter = 700
            for i, eps in enumerate(epsValues):
                color, res, problemLabel = getColor(problemType, i), getMarker(problemType), getLabel(problemType)
                label = rf'$\varepsilon=${eps}' + problemLabel

                vals = resultsDict[nNodes][problemType][eps][quantityLabel]

                if isinstance(vals, list):
                    real_parts = [val.real for val in vals]
                    imag_parts = [val.imag for val in vals]
                else:
                    real_parts = vals.real
                    imag_parts = vals.imag

                DistributionPlotter.scatter(
                    real_parts,
                    imag_parts,
                    color=color,
                    marker=res['marker'],
                    label=label,
                    s=sIter,
                )
                sIter -= 50

        if quantityLabel == 'eigvals':
            DistributionPlotter.plot(a, b, color='black')

            DistributionPlotter.add_hline(0.0)
            DistributionPlotter.add_vline(0.0)

            DistributionPlotter.set_aspect('equal')

        if quantityLabel == "singvals":
            DistributionPlotter.set_xscale(scale='log', base=10)

        DistributionPlotter.set_xlabel('Real part')
        DistributionPlotter.set_ylabel('Imaginary part')

        if xLim is not None and yLim is not None:
            DistributionPlotter.set_xlim(xLim)
            DistributionPlotter.set_ylim(yLim)

        DistributionPlotter.set_title(plotTitle + " " + f"{nNodes} nodes", subplot_index=0)

        DistributionPlotter.set_legend(subplot_index=0, loc='upper right', frameon=True, framealpha=0.7)

        filename = "data" + "/" + f"{problemName}" + "/" + fileNamePlot + f"{nNodes=}.png"
        DistributionPlotter.save(filename)


def plotEigvals(nrows, problemName, problems, resultsDict):
    nNodesList = sorted(resultsDict.keys())

    for nNodes in nNodesList:
        for problemType, epsValues in problems.items():
            for eps in epsValues:
                problemLabel = getLabel(problemType)
                label = f"{eps=}" + problemLabel

                eigvals = resultsDict[nNodes][problemType][eps]['eigvals']
                eigvecs = resultsDict[nNodes][problemType][eps]['eigvecs']

                index = range(eigvals.shape[0])

                # Sort eigenvalues and corresponding eigenvectors by the magnitude of eigenvalues
                sorted_indices = np.argsort(np.abs(eigvals))
                sorted_eigvals = eigvals[sorted_indices]
                sorted_eigvecs = eigvecs[:, sorted_indices]

                # Normalize the eigenvectors
                normalized_sorted_eigvecs = sorted_eigvecs / np.linalg.norm(sorted_eigvecs, axis=0)

                neigvals = len(sorted_eigvals)
                ncols = neigvals // nrows

                # Dynamically adjust the figure size: width depends on columns, height on rows
                fig_width = 4 * ncols
                fig_height = 4 * nrows

                EigvecsPlotter = Plotter(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), hspace=0.5, wspace=0.3)

                axs = EigvecsPlotter.axes

                for i in range(axs.flatten().shape[0]):
                    # Real and imaginary parts of the normalized eigenvector
                    real_part = np.real(normalized_sorted_eigvecs[:, i])
                    imag_part = np.imag(normalized_sorted_eigvecs[:, i])

                    EigvecsPlotter.plot(index, real_part, color='k', marker='o', markersize=11.0, linewidth=2.0, subplot_index=i, label="Real" if i == 0 else None)
                    EigvecsPlotter.plot(index, imag_part, color='k', marker='+', markersize=11.0, linewidth=2.0, subplot_index=i, label="Imag" if i == 0 else None)

                    EigvecsPlotter.set_title(f'Eigenvalue {i+1}: {sorted_eigvals[i].real:.2f} + {sorted_eigvals[i].imag:.2f}i', subplot_index=i)

                    EigvecsPlotter.set_xlabel('Index', subplot_index=i)
                    EigvecsPlotter.set_ylabel('Normalized value', subplot_index=i)

                    EigvecsPlotter.set_legend(subplot_index=0, loc='upper left')

                    EigvecsPlotter.set_grid()

                    EigvecsPlotter.set_ylim((-1.2, 1.2), subplot_index=i)

                filename = "data" + "/" + f"{problemName}" + "/" + f"eigenvectors_{nNodes=}_{label}.png"
                EigvecsPlotter.save(filename)


if __name__ == "__main__":
    main()
    # main2()
