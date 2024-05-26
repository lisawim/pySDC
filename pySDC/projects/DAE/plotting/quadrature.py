import numpy as np
import matplotlib.pyplot as plt

from pySDC.core.Step import step

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPPMinion

from pySDC.projects.DAE.plotting.error_propagation_Minion import generateDescription

from pySDC.helpers.stats_helper import get_sorted


def main():
    problem = LinearTestSPPMinion
    sweeper = generic_implicit

    # sweeper params
    nNodes = [2, 3, 4, 5]
    quad_type = 'RADAU-RIGHT'
    QI = 'IE'

    epsValues = [10 ** (-m) for m in range(4, 11)]

    t0 = 0.0
    dtValues = np.logspace(-5.0, 0.0, num=40)

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

    for M in nNodes:
        fig, ax = plt.subplots(1, 1, figsize=(9.5, 9.5))
        quadRes = np.zeros((len(dtValues), len(epsValues)))

        for e, eps in enumerate(epsValues):

            for d, dt in enumerate(dtValues):
                print(eps, dt)

                description = generateDescription(dt, M, QI, sweeper, quad_type, problem, eps)

                S = step(description=description)

                L = S.levels[0]
                P = L.prob

                L.status.time = t0
                u0 = S.levels[0].prob.u_exact(0.0)
                S.init_step(u0)

                L.sweep.predict()

                Qmat = L.sweep.coll.Qmat

                nodes = L.sweep.coll.nodes
                me = []
                for m in range(1, len(nodes) + 1):
                    tau = L.time + L.dt * nodes[m - 1]
                    z = P.u_exact(tau)[-1]
                    me.append(z - L.u[0][-1])
                    for j in range(1, len(nodes) + 1):
                        me[-1] -= L.dt * Qmat[m, j] * P.eval_f(P.u_exact(L.time + L.dt * nodes[j - 1]), L.time + L.dt * nodes[j - 1])[-1]

                quadRes[d, e] = max(me)

        for e_plot, eps in enumerate(epsValues):
            ax.loglog(
                dtValues,
                quadRes[:, e_plot],
                color=colors[e_plot],
                marker='*',
                markersize=10.0,
                linewidth=4.0,
                linestyle='solid',
                solid_capstyle='round',
                label=rf"$\varepsilon=${eps}",
            )

        ax.set_ylabel('Quadrature result', fontsize=16)

        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'$\Delta t$', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(visible=True)
        ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
        ax.minorticks_off()

        fig.savefig(f"data/{problem.__name__}/QuadratureResult_QI={QI}_{M}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()
