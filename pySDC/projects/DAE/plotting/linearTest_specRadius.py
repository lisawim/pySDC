import numpy as np
import matplotlib.cm as cm

from pySDC.projects.DAE import computeNormalityDeviation, getColor, getIterationMatrix, getLabel, getMarker, Plotter

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def getCoefficientMatrix(problemName, eps):
    if problemName == "LINEAR-TEST":
        lamb_diff = -2.0
        lamb_alg = 1.0

        A = np.array([[lamb_diff, lamb_alg], [lamb_diff / eps, -lamb_alg / eps]])
    else:
        raise NotImplementedError()
    return A


def compareEpsilonsAlongStepSizes():
    problemName = 'LINEAR-TEST'

    QI = 'IE'
    kFLEX = 5

    quadType = 'RADAU-RIGHT'
    nrows = 4
    nNodesList = [11, 12]  # range(4, 65, nrows)
    QGenerator = Q_GENERATORS['coll']
    QIGenerator = QDELTA_GENERATORS[QI]

    dtList = np.logspace(-2.5, 0.0, num=11)

    epsList = [10 ** (-m) for m in range(1, 12)]

    # Define a dictionary with problem types and their respective parameters
    problems = {
        'SPP': epsList,
        'embeddedDAE': [0.0],
        'constrainedDAE': [0.0],
    }

    for nNodes in nNodesList:
        SpecRadiusPlotter = Plotter(nrows=1, ncols=1, figsize=(30, 10))

        for problemType, epsValues in problems.items():
            for i, eps in enumerate(epsValues):
                specRadius = []

                for dt in dtList:
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

                color, res, problemLabel = getColor(problemType, i), getMarker(problemType, i), getLabel(problemType)
                label = rf'$\varepsilon=${eps}' + problemLabel

                SpecRadiusPlotter.plot(
                    dtList,
                    specRadius,
                    subplot_index=0,
                    color=color,
                    marker=res['marker'],
                    markersize=res['markersize'],
                    label=label,
                    plot_type="semilogx",
                )


        SpecRadiusPlotter.set_xticks(dtList, subplot_index=0)
        SpecRadiusPlotter.set_xlabel('Time step sizes', subplot_index=0)
        SpecRadiusPlotter.set_xscale(scale="log")

        SpecRadiusPlotter.set_legend(subplot_index=0, loc='best')
        SpecRadiusPlotter.set_grid(True, subplot_index=None)

        SpecRadiusPlotter.set_ylabel("Spectral radius", subplot_index=0)
        SpecRadiusPlotter.set_ylim((0.0, 1.0), subplot_index=0)

        filename = "data" + "/" + f"{problemName}" + "/" + f"spectral_radius_{QI=}_{nNodes=}.png"
        SpecRadiusPlotter.save(filename)


def compareNodesAlongStiffLimitFixedStepSize():
    problemName = 'LINEAR-TEST'

    QI = 'MIN-SR-FLEX'
    kFLEX = 5

    quadType = 'RADAU-RIGHT'
    nrows = 4
    nNodesList = range(2, 65, nrows)  # range(4, 65, nrows)
    QGenerator = Q_GENERATORS['coll']
    QIGenerator = QDELTA_GENERATORS[QI]

    dt = 1e-2

    epsList = [10 ** (-m) for m in range(1, 12)]  # np.logspace(-11.0, -1.0, num=40)

    # Define a dictionary with problem types and their respective parameters
    problems = {
        'SPP': epsList,
        # 'embeddedDAE': [0.0],
        # 'constrainedDAE': [0.0],
    }

    # Colormap setup
    cMap = cm.rainbow
    lenNodes = len(nNodesList)

    SpecRadiusPlotter = Plotter(nrows=1, ncols=1, figsize=(10, 10))
    for i, nNodes in enumerate(nNodesList):
        specRadius, stiffLimit = [], []

        for problemType, epsValues in problems.items():
            for eps in epsValues:
                stiffLimit.append(dt / eps)

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

        color = cMap(i / (lenNodes - 1))
        label = rf'{nNodes} nodes'
        SpecRadiusPlotter.plot(
            stiffLimit,
            specRadius,
            subplot_index=0,
            color=color,
            marker='o',
            # markersize=11.0,
            label=label,
            plot_type="semilogx",
        )

    SpecRadiusPlotter.set_xticks(stiffLimit, subplot_index=0)
    SpecRadiusPlotter.set_xlabel(r"$\Delta t / \varepsilon$", subplot_index=0)
    SpecRadiusPlotter.set_xscale(scale="log")

    SpecRadiusPlotter.set_legend(subplot_index=0, loc='best')
    SpecRadiusPlotter.set_grid(True, subplot_index=None)

    SpecRadiusPlotter.set_ylabel("Spectral radius", subplot_index=0)
    SpecRadiusPlotter.set_ylim((0.0, 1.8), subplot_index=0)

    QILabel = f"{QI=}_{kFLEX=}" if QI == "MIN-SR-FLEX" else f"{QI=}"
    filename = "data" + "/" + f"{problemName}" + "/" + f"spectral_radius_nodes_limit_{QILabel}.png"
    SpecRadiusPlotter.save(filename)


def compareNodesAlongStiffLimitFixedEpsilon():
    """Here, the spectral radius is plotted for fixed epsilon and different step sizes (compared to main())."""
    problemName = 'LINEAR-TEST'

    QI = "LU"
    kFLEX = 5

    quadType = 'RADAU-RIGHT'
    nNodes = 4
    QGenerator = Q_GENERATORS['coll']
    QIGenerator = QDELTA_GENERATORS[QI]

    dtList = np.logspace(-2.5, 0.0, num=11)

    epsList = [10 ** (-m) for m in range(1, 12)]  # np.logspace(-11.0, -1.0, num=40)

    # Define a dictionary with problem types and their respective parameters
    problems = {
        'SPP': epsList,
        # 'embeddedDAE': [0.0],
        # 'constrainedDAE': [0.0],
    }

    SpecRadiusPlotter = Plotter(nrows=1, ncols=1, figsize=(10, 10))
    for problemType, epsValues in problems.items():
        for i, eps in enumerate(epsValues):
            specRadius, stiffLimit = [], []

            for dt in dtList:
                stiffLimit.append(dt / eps)

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
            label = rf'$\varepsilon=${eps}' + problemLabel
            SpecRadiusPlotter.plot(
                stiffLimit,
                specRadius,
                subplot_index=0,
                color=color,
                marker='o',
                # markersize=11.0,
                label=label,
                plot_type="semilogx",
            )

    SpecRadiusPlotter.set_title(f"For {nNodes} collocation nodes", subplot_index=0)
    SpecRadiusPlotter.set_xticks(stiffLimit, subplot_index=0)
    SpecRadiusPlotter.set_xlabel(r"$\Delta t / \varepsilon$", subplot_index=0)
    SpecRadiusPlotter.set_xscale(scale="log")

    SpecRadiusPlotter.set_legend(subplot_index=0, loc='best')
    SpecRadiusPlotter.set_grid(True, subplot_index=None)

    SpecRadiusPlotter.set_ylabel("Spectral radius", subplot_index=0)
    SpecRadiusPlotter.set_ylim((0.0, 1.0), subplot_index=0)

    QILabel = f"{QI=}_{kFLEX=}" if QI == "MIN-SR-FLEX" else f"{QI=}"
    filename = "data" + "/" + f"{problemName}" + "/" + f"spectral_radius_nodes_limit_{QILabel}_{nNodes=}.png"
    SpecRadiusPlotter.save(filename)


if __name__ == "__main__":
    compareEpsilonsAlongStepSizes()
    # compareNodesAlongStiffLimitFixedStepSize()
    # compareNodesAlongStiffLimitFixedEpsilon()
