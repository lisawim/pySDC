import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pySDC.core.Step import step
from projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained
from pySDC.projects.DAE.plotting.error_propagation_Minion import generateDescription


def plot():
    r"""
    Function plots the spectral radius for ``simple_dae_1IntegralFormulation``.
    """

    Path("data").mkdir(parents=True, exist_ok=True)
    Path(f"data/simple_dae_1IntegralFormulation").mkdir(parents=True, exist_ok=True)

    dt_list = np.logspace(-3.0, 2.0, num=50)
    sweeper = genericImplicitConstrained
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'IE'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold', 'silver']

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P', 'h']

    a = 10.0

    Nd = 2
    Na = 1
    N = Nd + Na

    E = np.zeros((N, N))
    for m in range(N):
        E[m, m] = 1 if m < Nd else 0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    spectral_radius = np.zeros((len(dt_list), len(M_all)))
    for q, M in enumerate(M_all):
        QIB = np.zeros((M * N, M * N))
        IC = np.zeros((M * N, M * N))
        QQIB = np.zeros((M * N, M * N))
        for e, dt_loop in enumerate(dt_list):
            description = generateDescription(dt_loop, M, QI, sweeper, quad_type)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            Id = np.identity(M)
            dt = L.params.dt
            L.status.time = 0.0
            nodes = [L.time + L.dt * L.sweep.coll.nodes[m] for m in range(M)]

            for m in range(M):
                for j, t in enumerate(nodes):
                    Ad = np.array([[a - 1 / (2 - t), 0, a * (2 - t)], [(1 - a) / (t - 2), -1, a - 1]])
                    Aa = np.array([[t + 2, t**2 - 4, 0]])

                    B = np.concatenate((Ad, np.zeros((Na, N))), axis=0)
                    C = np.concatenate((np.zeros((Nd, N)), Aa), axis=0)

                    QIB[m * N : m * N + N, j * N : j * N + N] = QImat[m, j] * B
                    IC[m * N : m * N + N, j * N : j * N + N] = Id[m, j] * C
                    QQIB[m * N : m * N + N, j * N : j * N + N] = (Q[m, j] - QImat[m, j]) * B

            inv = np.linalg.inv(np.kron(np.identity(M), E) - dt * QIB - IC)
            K = np.matmul(inv, dt * QQIB)

            spectral_radius[e, q] = max(abs(np.linalg.eigvals(K)))

    for q_plot, M_plot in enumerate(M_all):
        ax.semilogx(
            dt_list,
            spectral_radius[:, q_plot],
            color=colors[q_plot],
            marker=marker[q_plot],
            markeredgecolor='k',
            label=rf'$M=${M_plot}',
        )
    ax.set_ylabel(r'$\rho(\mathbf{K}_\varepsilon)$', fontsize=16)
    ax.set_ylim(0.0, 1.4)

    ax.set_xlabel(r'$\Delta t$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/simple_dae_1IntegralFormulation/SR_IterMatrix_QI={QI}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    plot()