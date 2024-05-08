import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.Step import step

from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.DAE.plotting.error_propagation_Minion import generateDescription


def plotSRIterMatrixDiffEpsQI():
    dt = 0.1
    eps_list = np.logspace(-12.0, 0.0, 100)#np.logspace(-10.0, 0.0, 100)#[0.0, 10 ** (-8), 10 ** (-6), 10 ** (-4), 10 ** (-2), 1.0]
    sweeper = generic_implicit
    M = 5
    quad_type = 'RADAU-RIGHT'
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange']
    lambda_d = 1
    lambda_a = 1

    marker = ['o', '*', 'D', 's', '^', '<']

    I3 = np.identity(2)
    I3[-1, -1] = 0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    spectral_radius = np.zeros((len(eps_list), len(QI_all)))
    for q, QI in enumerate(QI_all):
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

            A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
            K = np.matmul(inv, dt * np.kron(Q - QImat, A))

            eigvals = np.linalg.eigvals(K)
            spectral_radius[e, q] = max(abs(eigvals))

    for q_plot, QI_plot in enumerate(QI_all):
        ax.semilogx(
            eps_list,
            spectral_radius[:, q_plot],
            color=colors[q_plot],
            marker=marker[q_plot],
            markeredgecolor='k',
            label=rf'{QI_plot}',
        )
    ax.set_ylabel(r'$\rho(\mathbf{K}_\varepsilon)$', fontsize=16)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=12, loc='center left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/SR_IterMatrixDiffEps_M={M}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotSRIterMatrixDiffEpsM():
    dt = 0.1
    eps_list = np.logspace(-12.0, 0.0, 100)
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'LU'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']
    lambda_d = 1
    lambda_a = 1

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']

    I3 = np.identity(2)
    I3[-1, -1] = 0

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

            A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
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

    fig.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/SR_IterMatrixDiffEps_QI={QI}_dt={dt}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotSRIterMatrixDiffEpsDt():
    dt_list = np.logspace(-5.0, 1.0, num=100)
    eps_list = [10 ** (-k) for k in range(1, 13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'IE'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold', 'silver']

    lambda_d, lambda_a = 1, 1

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

                A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

                inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
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
        ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
        ax.minorticks_off()

        fig.savefig(f"../run/data/EmbeddedLinearTestDAE/eps_embedding/SR_IterMatrixDiffEps_QI={QI}_M={M}_{quad_type}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def plotSRDAE():
    dt = 0.1
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    quad_type = 'RADAU-RIGHT'
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange']
    lambda_d = 1
    lambda_a = 1

    A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1]])
    E3 = np.identity(4)
    E3[-1, -1] = 0

    marker = ['o', '*', 'D', 's', '^', '<']

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    spectral_radius = np.zeros((len(M_all), len(QI_all)))
    for q, QI in enumerate(QI_all):
        for e, M in enumerate(M_all):
            description = generateDescription(dt, M, QI, sweeper, quad_type)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            Ieps = np.identity(2)
            Ieps[-1, -1] = 0

            A_d = np.array([[lambda_d, lambda_a], [0, 0]])
            A_a = np.array([[0, 0], [lambda_d, -lambda_a]])

            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QImat, A_d) + np.kron(np.identity(M), A_a))
            K = np.matmul(inv, dt * np.kron(Q - QImat, A_d))

            # inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QImat, A))
            # K = np.matmul(inv, dt * np.kron(Q - QImat, A))

            eigvals = np.linalg.eigvals(K)
            spectral_radius[e, q] = max(abs(eigvals))

    for q_plot, QI_plot in enumerate(QI_all):
        ax.plot(
            M_all,
            spectral_radius[:, q_plot],
            color=colors[q_plot],
            marker=marker[q_plot],
            markeredgecolor='k',
            label=rf'{QI_plot}',
        )
    ax.set_ylabel(r'$\rho(\mathbf{K}_0)$', fontsize=16)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(r'Number of nodes $M$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/SR_IterMatrixDAE_M={M}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotErrPropagationMaxNormDiffEps():
    dt = 0.1
    eps_list = np.logspace(-12.0, 0.0, 100)#np.logspace(-10.0, 0.0, 100)#[0.0, 10 ** (-8), 10 ** (-6), 10 ** (-4), 10 ** (-2), 1.0]
    sweeper = generic_implicit
    M = 5
    quad_type = 'RADAU-RIGHT'
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange']
    lambda_d = 1
    lambda_a = 1

    marker = ['o', '*', 'D', 's', '^', '<']

    I3 = np.identity(2)
    I3[-1, -1] = 0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    max_norm = np.zeros((len(eps_list), len(QI_all)))
    for q, QI in enumerate(QI_all):
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

            A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
            E = np.matmul(inv, dt * np.kron(Q - QImat, A))

            max_norm[e, q] = np.linalg.norm(E, np.inf)

    for q_plot, QI_plot in enumerate(QI_all):
        ax.semilogx(
            eps_list,
            max_norm[:, q_plot],
            color=colors[q_plot],
            marker=marker[q_plot],
            markeredgecolor='k',
            label=rf'{QI_plot}',
        )
    ax.set_ylabel(r'$||\mathbf{E}_\varepsilon ||_\infty$', fontsize=16)
    # ax.set_ylim(0.0, 1.5)

    ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=12, loc='center left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/MaxNorm_IterMatrixDiffEps_M={M}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotMaxNormIterMatrixDiffEpsM():
    dt = 1e-10
    eps_list = np.logspace(-12.0, 0.0, 100)
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'LU'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']
    lambda_d = 1
    lambda_a = 1

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']

    I3 = np.identity(2)
    I3[-1, -1] = 0

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

            A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
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

    fig.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/MaxNorm_IterMatrixDiffEps_QI={QI}_dt={dt}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotMaxNormIterMatrixDiffEpsDt():
    dt_list = np.logspace(-5.0, 1.0, num=100)
    eps_list = [10 ** (-k) for k in range(1, 13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'IE'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold', 'silver']

    lambda_d, lambda_a = 1, 1

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

                A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

                inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
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
        ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
        ax.minorticks_off()

        fig.savefig(f"../run/data/EmbeddedLinearTestDAE/eps_embedding/MaxNorm_IterMatrixDiffEps_QI={QI}_M={M}_{quad_type}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def plotErrPropagationMaxNormDAE():
    dt = 0.1
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    quad_type = 'RADAU-RIGHT'
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange']
    lambda_d = 1
    lambda_a = 1

    marker = ['o', '*', 'D', 's', '^', '<']

    I3 = np.identity(2)
    I3[-1, -1] = 0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    max_norm = np.zeros((len(M_all), len(QI_all)))
    for q, QI in enumerate(QI_all):
        for e, M in enumerate(M_all):
            description = generateDescription(dt, M, QI, sweeper, quad_type)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            A_d = np.array([[lambda_d, lambda_a], [0, 0]])
            A_a = np.array([[0, 0], [lambda_d, -lambda_a]])

            inv = np.linalg.inv(np.kron(np.identity(M), I3) - dt * np.kron(QImat, A_d) + np.kron(np.identity(M), A_a))
            E = np.matmul(inv, dt * np.kron(Q - QImat, A_d))

            max_norm[e, q] = np.linalg.norm(E, np.inf)

    for q_plot, QI_plot in enumerate(QI_all):
        ax.plot(
            M_all,
            max_norm[:, q_plot],
            color=colors[q_plot],
            marker=marker[q_plot],
            markeredgecolor='k',
            label=rf'{QI_plot}',
        )
    ax.set_ylabel(r'$||\mathbf{E}_0 ||_\infty$', fontsize=16)
    # ax.set_ylim(0.0, 1.5)

    ax.set_xlabel(r'Number of nodes $M$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/MaxNorm_IterMatrixDAE_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotErrPropagationMaxNormHeatmap():
    sweeper = generic_implicit
    M_all = [3]
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    quad_type = 'RADAU-RIGHT'

    numsteps = 400
    dt_list = np.logspace(-6.0, 0.0, num=numsteps)  # xdim
    eps = np.logspace(-12.0, 0.0, numsteps)  # ydim

    lambda_d = 1
    lambda_a = 1

    for QI in QI_all:
        for M in M_all:
            heat_field = np.zeros((len(dt_list), len(eps)))
            heat_field_eps0 = np.zeros((len(dt_list), len(eps)))
            for idx, x in enumerate(dt_list):
                for idy, y in enumerate(eps):
                    description = generateDescription(x, M, QI, generic_implicit, quad_type)

                    S = step(description=description)

                    L = S.levels[0]
                    P = L.prob

                    u0 = S.levels[0].prob.u_exact(0.0)
                    S.init_step(u0)
                    QImat = L.sweep.QI[1:, 1:]
                    Q = L.sweep.coll.Qmat[1:, 1:]
                    dt = L.params.dt
                    assert dt == x

                    A = np.array([[lambda_d, lambda_a], [lambda_d / y, -lambda_a / y]])

                    inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
                    E = np.matmul(inv, dt * np.kron(Q - QImat, A))

                    heat_field[idx, idy] = np.linalg.norm(E, np.inf)

            plt.figure()
            print(dt_list.shape, eps.shape)
            plt.pcolor(dt_list, eps, heat_field.T, cmap='Reds', vmin=0, vmax=1)
            plt.xlim(min(dt_list), max(dt_list))
            plt.ylim(min(eps), max(eps))
            plt.xlabel(r'Time step size $\Delta t$')
            plt.ylabel(r'Perturbation parameter $\varepsilon$')
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)
            cbar = plt.colorbar()
            cbar.set_label('Spectral radius')

            plt.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/HeatmapMaxNorm_M={M}_{QI}_{quad_type}.png", dpi=300, bbox_inches='tight')


def plotCondNumberDiffEpsM():
    dt = 0.1
    eps_list = np.logspace(-12.0, 0.0, 100)
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'IE'
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']
    lambda_d = 1
    lambda_a = 1

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']

    I3 = np.identity(2)
    I3[-1, -1] = 0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    condition_number = np.zeros((len(eps_list), len(M_all)))
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

            A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
            K = inv

            condition_number[e, q] = np.linalg.norm(K, 2) * np.linalg.norm(np.linalg.inv(K), 2)
            # print('Normality:', np.matmul(K.T, K) - np.matmul(K, K.T))

    for q_plot, M_plot in enumerate(M_all):
        ax.loglog(
            eps_list,
            condition_number[:, q_plot],
            color=colors[q_plot],
            marker=marker[q_plot],
            markeredgecolor='k',
            label=rf'$M=${M_plot}',
        )
    ax.set_ylabel(r'$\kappa(\mathbf{K}_\varepsilon)$', fontsize=16)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=10, loc='center left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/Condition_IterMatrixDiffEps_QI={QI}_dt={dt}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def EVDistributionIterMatrixDiffEpsM():
    dt = 0.1
    eps_list = [10 ** (-k) for k in range(13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5]#, 6, 7, 8, 9, 10, 11, 12]
    QI_all = ['IE']
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']
    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']
    lambda_d = 1
    lambda_a = 1

    a = np.cos(np.linspace(0, 2 * np.pi, 200))
    b = np.sin(np.linspace(0, 2 * np.pi, 200))

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

                A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

                inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
                K = np.matmul(inv, dt * np.kron(Q - QImat, A))

                eigvals = np.linalg.eigvals(K)
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

            plt.axhline(y=0.0, color='black', linestyle='--')
            plt.axvline(x=0.0, color='black', linestyle='--')
            plt.plot(a, b, color="black")
            plt.gca().set_aspect('equal')
            plt.legend(loc='upper right', fontsize=12)
            plt.tick_params(axis='both', which='major', labelsize=16)
            plt.xlabel('Real part of eigenvalue', fontsize=16)
            plt.ylabel('Imaginary part of eigenvalue', fontsize=16)
            # plt.xlim(0.0, 0.6)
            plt.xlim(-1.0, 1.0)
            plt.ylim(-1.0, 1.0)
            plt.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/EVDistributionIterMatrixDiffEps_{QI}_dt={dt}_eps={eps_loop}.png", dpi=300, bbox_inches='tight')


def EVDistributionJacobianDiffEpsM():
    dt = 0.1
    eps_list = [10 ** (-k) for k in range(13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5]#, 6, 7, 8, 9, 10, 11, 12]
    QI_all = ['IE']
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']
    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']
    lambda_d = 1
    lambda_a = 1

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

                A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

                J = np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A)

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
            plt.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/EVDistributionJacobianDiffEps_{QI}_dt={dt}_eps={eps_loop}.png", dpi=300, bbox_inches='tight')


def ConditionODEDiffEps():
    dt = 0.1
    eps_list = [10 ** (-k) for k in range(13)]
    quad_type = 'RADAU-RIGHT'
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange', 'plum', 'salmon', 'forestgreen', 'midnightblue', 'gold']
    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P']
    lambda_d = 1
    lambda_a = 1

    plt.figure(figsize=(7.5, 7.5))
    condition = np.zeros(len(eps_list))
    for eps_index, eps_loop in enumerate(eps_list):

        A = np.array([[lambda_d, lambda_a], [lambda_d / eps_loop, -lambda_a / eps_loop]])

        eigvals = np.linalg.eigvals(A)
        condition[eps_index] = np.linalg.norm(A, 2) * np.linalg.norm(np.linalg.inv(A), 2)

    plt.loglog(
        eps_list,
        condition,
        # marker=marker[m_plot],
        # color=colors[m_plot],
        # facecolors=facecolors,
        # s=200.0,
        # label=rf'$M=${M}',
    )

    # plt.axhline(y=0.0, color='black', linestyle='--')
    # plt.axvline(x=0.0, color='black', linestyle='--')
    # plt.plot(a, b, color="black")
    # plt.gca().set_aspect('equal')
    plt.legend(loc='upper right', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylabel('Condition', fontsize=16)
    plt.xlabel(r'Parameter $\varepsilon$', fontsize=16)
    # plt.xlim(0.0, 0.6)
    # plt.xlim(-1.0, 1.0)
    # plt.ylim(-1.0, 1.0)
    plt.savefig(f"data/EmbeddedLinearTestDAE/eps_embedding/ConditionODEDiffEps_dt={dt}_eps={eps_loop}.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # plotSRIterMatrixDiffEpsQI()
    # plotSRIterMatrixDiffEpsM()
    plotSRIterMatrixDiffEpsDt()
    # plotSRDAE()
    # plotErrPropagationMaxNormDiffEps()
    plotMaxNormIterMatrixDiffEpsDt()
    # plotErrPropagationMaxNormDAE()
    # plotErrPropagationMaxNormHeatmap()
    # plotCondNumberDiffEpsM()
    # EVDistributionIterMatrixDiffEpsM()
    # EVDistributionJacobianDiffEpsM()
    # ConditionODEDiffEps()
    # plotMaxNormIterMatrixDiffEpsM()