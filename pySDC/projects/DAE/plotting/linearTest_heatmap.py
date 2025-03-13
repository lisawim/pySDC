import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.DAE import computeNormalityDeviation, getColor, getIterationMatrix, getLabel, getMarker, Plotter

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def main():
    problemName = 'LINEAR-TEST'
    problemType = 'SPP'

    QI = 'LU'
    kFLEX = 5

    quadType = 'RADAU-RIGHT'
    nrows = 4
    nNodesList = [10]  # range(2, 65, nrows)  # range(4, 65, nrows)
    QGenerator = Q_GENERATORS['coll']
    QIGenerator = QDELTA_GENERATORS[QI]

    numsteps = 400
    dtList = np.logspace(-6.0, 0.0, num=numsteps)  # xdim
    epsValues = np.logspace(-12.0, 0.0, numsteps)  # ydim


    for nNodes in nNodesList:
        heatField = np.zeros((len(dtList), len(epsValues)))

        for idy, eps in enumerate(epsValues):

            for idx, dt in enumerate(dtList):
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

                heatField[idx, idy] = max(abs(np.linalg.eigvals(K)))

        plt.figure()
        plt.pcolor(dtList, epsValues, heatField.T, cmap='Reds', vmin=0, vmax=1)
        plt.xlim(min(dtList), max(dtList))
        plt.ylim(min(epsValues), max(epsValues))
        plt.xlabel(r'Time step size $\Delta t$')
        plt.ylabel(r'Perturbation parameter $\varepsilon$')
        plt.xscale('log', base=10)
        plt.yscale('log', base=10)
        cbar = plt.colorbar()
        cbar.set_label('Spectral radius')

        filename = "data" + "/" + f"{problemName}" + "/" + f"spectral_radius_heatmap_{QI=}_{nNodes=}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
