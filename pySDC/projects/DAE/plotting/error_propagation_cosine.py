import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.Step import step

from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAEMinion
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.SemiExplicitDAE import SemiExplicitDAE
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.DAE.run.error_propagation_Minion import generateDescription


def plotSRIterMatrixDiffEpsM():
    dt = 0.1
    eps_list = np.logspace(-12.0, 0.0, 100)
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'MIN-SR-S'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    spectral_radius = np.zeros((len(eps_list), len(M_all)))
    for q, M in enumerate(M_all):
        for e, eps_loop in enumerate(eps_list):
            description = generateDescription(dt, M, QI, sweeper, quad_type)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            A = -1 / eps_loop

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(1)) - dt * np.kron(QImat, A))
            K = np.matmul(inv, dt * np.kron(Q - QImat, A))

            eigvals = np.linalg.eigvals(K)
            spectral_radius[e, q] = max(abs(eigvals))

    for q_plot, M_plot in enumerate(M_all):
        ax.semilogx(
            eps_list,
            spectral_radius[:, q_plot],
            color=colors[q_plot],
            marker=marker[q_plot],
            markeredgecolor='k',
            label=rf'$M=${M_plot}',
        )
    ax.set_ylabel(r'$\rho(\mathbf{K}_\varepsilon)$', fontsize=16)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=10, loc='center left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/CosineProblem/eps_embedding/SR_IterMatrixDiffEps_QI={QI}_dt={dt}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotSRIterMatrixDiffEpsDt():
    dt_list = np.logspace(-5.0, 1.0, num=100)
    eps_list = [10 ** (-k) for k in range(1, 13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'LU'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold', 'silver']

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P', 'h']

    for M in M_all:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        spectral_radius = np.zeros((len(dt_list), len(eps_list)))
        for q, dt in enumerate(dt_list):
            for e, eps_loop in enumerate(eps_list):
                description = generateDescription(dt, M, QI, sweeper, quad_type)

                S = step(description=description)

                L = S.levels[0]
                P = L.prob

                u0 = S.levels[0].prob.u_exact(0.0)
                S.init_step(u0)
                QImat = L.sweep.QI[1:, 1:]
                Q = L.sweep.coll.Qmat[1:, 1:]
                dt = L.params.dt

                A = -1 / eps_loop

                inv = np.linalg.inv(np.kron(np.identity(M), np.identity(1)) - dt * np.kron(QImat, A))
                K = np.matmul(inv, dt * np.kron(Q - QImat, A))

                eigvals = np.linalg.eigvals(K)
                spectral_radius[q, e] = max(abs(eigvals))

        for e, eps in enumerate(eps_list):
            ax.semilogx(
                dt_list,
                spectral_radius[:, e],
                color=colors[e],
                marker=marker[e],
                markeredgecolor='k',
                label=rf'$\varepsilon=${eps}',
            )
        ax.set_ylabel(r'$\rho(\mathbf{K}_\varepsilon)$', fontsize=16)
        ax.set_ylim(0.0, 1.5)

        ax.set_xlabel(r'$\Delta t$', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=False, fontsize=10, loc='upper right', ncol=2)
        ax.minorticks_off()

        fig.savefig(f"data/CosineProblem/eps_embedding/SR_IterMatrixDiffEps_QI={QI}_M={M}_{quad_type}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def EVDistributionJacobianDiffEpsM():
    dt = 0.1
    eps_list = [10 ** (-k) for k in range(13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5]#, 6, 7, 8, 9, 10, 11, 12]
    QI_all = ['MIN']
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']
    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']

    for QI in QI_all:
        for eps_index, eps_loop in enumerate(eps_list):
            plt.figure(figsize=(7.5, 7.5))
            for m_plot, M in enumerate(M_all):
                description = generateDescription(dt, M, QI, sweeper, quad_type)

                S = step(description=description)

                L = S.levels[0]
                P = L.prob

                u0 = S.levels[0].prob.u_exact(0.0)
                S.init_step(u0)
                QImat = L.sweep.QI[1:, 1:]
                Q = L.sweep.coll.Qmat[1:, 1:]
                dt = L.params.dt

                A = - 1 / eps_loop
                J = np.kron(np.identity(M), np.identity(1)) - dt * np.kron(QImat, A)

                eigvals = np.linalg.eigvals(J)
                print(eps_loop, max(abs(eigvals)))
                plt.scatter(
                    eigvals.real,
                    eigvals.imag,
                    marker=marker[m_plot],
                    color=colors[m_plot],
                    # facecolors=facecolors,
                    s=200.0,
                    label=rf'$M=${M}',
                )

            # plt.axhline(y=0.0, color='black', linestyle='--')
            # plt.axvline(x=0.0, color='black', linestyle='--')
            # plt.plot(a, b, color="black")
            # plt.gca().set_aspect('equal')
            plt.legend(loc='upper right', fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.xlabel('Real part of eigenvalue', fontsize=16)
            plt.ylabel('Imaginary part of eigenvalue', fontsize=16)
            # plt.xlim(0.0, 0.6)
            # plt.xlim(-1.0, 1.0)
            # plt.ylim(-1.0, 1.0)
            plt.savefig(f"data/CosineProblem/eps_embedding/EVDistributionJacobianDiffEps_{QI}_dt={dt}_eps={eps_loop}.png", dpi=300, bbox_inches='tight')


def plotMaxNormIterMatrixDiffEpsM():
    dt = 0.1
    eps_list = np.logspace(-12.0, 0.0, 100)
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8]#, 9, 10, 11, 12]
    QI = 'IE'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    max_norm = np.zeros((len(eps_list), len(M_all)))
    for q, M in enumerate(M_all):
        for e, eps_loop in enumerate(eps_list):
            description = generateDescription(dt, M, QI, sweeper, quad_type)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            A = -1 / eps_loop

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(1)) - dt * np.kron(QImat, A))
            E = np.matmul(inv, dt * np.kron(Q - QImat, A))

            max_norm[e, q] = np.linalg.norm(E, np.inf)

    for q_plot, M_plot in enumerate(M_all):
        ax.semilogx(
            eps_list,
            max_norm[:, q_plot],
            color=colors[q_plot],
            marker=marker[q_plot],
            markeredgecolor='k',
            label=rf'$M=${M_plot}',
        )
    ax.set_ylabel(r'$||\mathbf{K}_\varepsilon||_\infty$', fontsize=16)
    # ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=10, loc='lower left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/CosineProblem/eps_embedding/MaxNorm_IterMatrixDiffEps_QI={QI}_dt={dt}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotMaxNormIterMatrixDiffEpsDt():
    dt_list = np.logspace(-5.0, 1.0, num=100)
    eps_list = [10 ** (-k) for k in range(1, 13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'LU'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold', 'silver']

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P', 'h']

    for M in M_all:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        max_norm = np.zeros((len(dt_list), len(eps_list)))
        for q, dt in enumerate(dt_list):
            for e, eps_loop in enumerate(eps_list):
                description = generateDescription(dt, M, QI, sweeper, quad_type)

                S = step(description=description)

                L = S.levels[0]
                P = L.prob

                u0 = S.levels[0].prob.u_exact(0.0)
                S.init_step(u0)
                QImat = L.sweep.QI[1:, 1:]
                Q = L.sweep.coll.Qmat[1:, 1:]
                dt = L.params.dt

                A = -1 / eps_loop

                inv = np.linalg.inv(np.kron(np.identity(M), np.identity(1)) - dt * np.kron(QImat, A))
                E = np.matmul(inv, dt * np.kron(Q - QImat, A))

                max_norm[q, e] = np.linalg.norm(E, np.inf)

        for e, eps in enumerate(eps_list):
            ax.semilogx(
                dt_list,
                max_norm[:, e],
                color=colors[e],
                marker=marker[e],
                markeredgecolor='k',
                label=rf'$\varepsilon=${eps}',
            )
        ax.set_ylabel(r'$||\mathbf{K}_\varepsilon||_\infty$', fontsize=16)
        ax.set_ylim(0.0, 3.5)

        ax.set_xlabel(r'$\Delta t$', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=False, fontsize=10, loc='upper right', ncol=2)
        ax.minorticks_off()

        fig.savefig(f"data/CosineProblem/eps_embedding/MaxNorm_IterMatrixDiffEps_QI={QI}_M={M}_{quad_type}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    # plotSRIterMatrixDiffEpsM()
    plotSRIterMatrixDiffEpsDt()
    # EVDistributionJacobianDiffEpsM()
    # plotMaxNormIterMatrixDiffEpsM()
    plotMaxNormIterMatrixDiffEpsDt()