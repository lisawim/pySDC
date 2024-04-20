import matplotlib.pyplot as plt
import numpy as np

from pySDC.core.Step import step

from pySDC.projects.DAE.problems.TestDAEs import LinearTestDAEMinion
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.sweepers.SemiExplicitDAE import SemiExplicitDAE
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
# from pySDC.projects.DAE.run.DAE_study import plotStylingStuff


def generateDescription(dt, M, QI, sweeper, quad_type):
    # initialize level parameters
    level_params = {
        'dt': dt,
    }

    problem_params = {
        'newton_tol': 1e-9,
        'method': 'gmres',
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': quad_type,
        'num_nodes': M,
        'QI': QI,
        'initial_guess': 'spread',
    }

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': LinearTestDAEMinion,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': {},
    }

    return description


def plot_SR_different_QDelta():
    # dt_list = np.logspace(-2, 0.0, num=8)
    dt_list = np.logspace(-3, 0.0, num=8)
    sweepers = [fully_implicit_DAE, SemiExplicitDAE]
    M_all = [4, 5, 6, 7]
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    colors, _, _, _ = plotStylingStuff(color_type='M')


    A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1]])
    C = np.array([[0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1]])
    E3 = np.identity(4)
    E3[-1, -1] = 0

    for M in M_all:
        # fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.5))
        fig, ax = plt.subplots(1, 1, figsize=(10.5, 10.5))

        for sweeper in sweepers:
            if sweeper.__name__ == 'fully_implicit_DAE':
                sweeper_cls_name = 'FI-SDC'
                linestyle = '--'
                marker = '*'
            elif sweeper.__name__ == 'SemiExplicitDAE':
                sweeper_cls_name = 'SI-SDC'
                linestyle = '-'
                marker = 'o'
            else:
                raise NotImplementedError

            spectral_radius = np.zeros((len(dt_list), len(QI_all)))
            max_norm = np.zeros((len(dt_list), len(QI_all)))

            for q, QI in enumerate(QI_all):
                # coll = CollBase(num_nodes=M, tleft=0, tright=1, quad_type=quad_type)
                # Sweeper = sweeper(sweeper_params)
                # QImat = Sweeper.get_Qdelta_implicit(coll, QI)
                print(f'For M={M} for {QI} we have')

                for i, dt_loop in enumerate(dt_list):
                    print(f'Time step size dt={dt_loop}')
                    print()
                    description = generateDescription(dt_loop, M, QI, sweeper)

                    S = step(description=description)

                    L = S.levels[0]
                    P = L.prob

                    u0 = S.levels[0].prob.u_exact(0.0)
                    S.init_step(u0)
                    QImat = L.sweep.QI[1:, 1:]
                    Q = L.sweep.coll.Qmat[1:, 1:]
                    dt = L.params.dt

                    if sweeper == fully_implicit_DAE:
                        inv = np.linalg.inv(np.kron(np.identity(M), E3) - dt * np.kron(QImat, A))
                        K = np.matmul(inv, dt * np.kron(Q - QImat, A))

                    elif sweeper == SemiExplicitDAE:
                        inv = np.linalg.inv(np.kron(np.identity(M), E3 + C) - dt * np.kron(QImat, A - C))
                        K = np.matmul(inv, dt * np.kron(Q - QImat, A - C))

                    eigvals = np.linalg.eigvals(K)
                    spectral_radius[i, q] = max(abs(eigvals))
                    # print(sweeper.__name__, 'Spectral radius:')
                    # print(spectral_radius[i, q])
                    # print()
                    max_norm[i, q] = np.linalg.norm(K, np.inf)

            for q_plot, QI_plot in enumerate(QI_all):
                # ax[0].semilogx(
                ax.semilogx(
                    dt_list,
                    spectral_radius[:, q_plot],
                    linestyle=linestyle,
                    color=colors[q_plot + 2],
                    marker=marker,
                    markeredgecolor='k',
                    label=rf'{sweeper_cls_name} - {QI_plot}'
                )
                # ax[1].semilogx(
                #     dt_list,
                #     max_norm[:, q_plot],
                #     linestyle=linestyle,
                #     color=colors[q_plot + 2],
                #     marker=marker,
                #     label=rf'{sweeper_cls_name} - {QI_plot}'
                # )

        # ax[0].set_ylabel(r'$\rho(\mathbf{E})$', fontsize=16)
        ax.set_ylabel(r'$\rho(\mathbf{K})$', fontsize=16)
        # ax[1].set_ylabel(r'$||\mathbf{E}||_\infty$', fontsize=16)
        ax.set_ylim(0.0, 1.0)
        # ax[1].set_ylim(0.0, 25.0)
        for ax_wrapper in [ax]:  # [ax[0], ax[1]]:
            ax_wrapper.set_xlabel(r'Time step size $\Delta t$', fontsize=16)
            ax_wrapper.set_xscale('log', base=10)
            ax_wrapper.tick_params(axis='both', which='major', labelsize=16)
            ax_wrapper.grid(visible=True)
            ax_wrapper.legend(frameon=False, fontsize=12, loc='upper center', ncol=2)
            ax_wrapper.minorticks_off()

        fig.savefig(f"data/LinearTestDAEMinion/Talk/Eigenvalues/ErrorPropagationMatrix_M={M}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_SR_different_M():
    # dt_list = np.logspace(-2, 0.0, num=8)
    dt_list = np.logspace(-3, 0.0, num=8)
    sweepers = [fully_implicit_DAE, SemiExplicitDAE]
    M_all = [4, 5, 6, 7]
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    # colors, _, _, _ = plotStylingStuff(color_type='M')
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick']


    A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1]])
    C = np.array([[0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1]])
    E3 = np.identity(4)
    E3[-1, -1] = 0

    for QI in QI_all:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
        for sweeper in sweepers:
            if sweeper.__name__ == 'fully_implicit_DAE':
                sweeper_cls_name = 'FI-SDC'
                linestyle = '--'
                marker = '*'
            elif sweeper.__name__ == 'SemiExplicitDAE':
                sweeper_cls_name = 'SI-SDC'
                linestyle = '-'
                marker = 'o'
            else:
                raise NotImplementedError

            spectral_radius = np.zeros((len(dt_list), len(M_all)))
            for m, M in enumerate(M_all):
                print(f'For M={M} for {QI} we have')

                for i, dt_loop in enumerate(dt_list):
                    print(f'Time step size dt={dt_loop}')
                    print()
                    description = generateDescription(dt_loop, M, QI, sweeper)

                    S = step(description=description)

                    L = S.levels[0]
                    P = L.prob

                    u0 = S.levels[0].prob.u_exact(0.0)
                    S.init_step(u0)
                    QImat = L.sweep.QI[1:, 1:]
                    Q = L.sweep.coll.Qmat[1:, 1:]
                    dt = L.params.dt

                    if sweeper == fully_implicit_DAE:
                        inv = np.linalg.inv(np.kron(np.identity(M), E3) - dt * np.kron(QImat, A))
                        K = np.matmul(inv, dt * np.kron(Q - QImat, A))

                    elif sweeper == SemiExplicitDAE:
                        inv = np.linalg.inv(np.kron(np.identity(M), E3 + C) - dt * np.kron(QImat, A - C))
                        K = np.matmul(inv, dt * np.kron(Q - QImat, A - C))

                    eigvals = np.linalg.eigvals(K)
                    spectral_radius[i, m] = max(abs(eigvals))

            for m_plot, M_plot in enumerate(M_all):
                ax.semilogx(
                    dt_list,
                    spectral_radius[:, m_plot],
                    linestyle=linestyle,
                    color=colors[m_plot],
                    marker=marker,
                    markeredgecolor='k',
                    label=rf'{sweeper_cls_name} - M={M_plot}'
                )

        ax.set_ylabel(r'$\rho(\mathbf{K})$', fontsize=16)
        ax.set_ylim(0.0, 1.0)

        ax.set_xlabel(r'Time step size $\Delta t$', fontsize=16)
        ax.set_xscale('log', base=10)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=False, fontsize=12, loc='lower center', ncol=2)
        ax.minorticks_off()

        fig.savefig(f"data/LinearTestDAEMinion/Talk/Eigenvalues/ErrorPropagationMatrix_QI={QI}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def plotSPPDifferentEpsilon():
    M_all = [5]  # [4, 5, 6, 7]
    QI_all = ['MIN']  # ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']

    numsteps = 400
    dt_list = np.logspace(-2.0, 0.0, num=numsteps)  # xdim
    eps = np.logspace(-10.0, -2.0, numsteps)  # ydim

    A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1]])

    for QI in QI_all:
        for M in M_all:
            heat_field = np.zeros((len(dt_list), len(eps)))
            heat_field_eps0 = np.zeros((len(dt_list), len(eps)))
            for idx, x in enumerate(dt_list):
                for idy, y in enumerate(eps):
                    Ieps = np.identity(4)
                    Ieps[3, 3] = y

                    I0 = np.identity(4)
                    I0[3, 3] = 0

                    Aeps = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1 / y, 1 / y, 0, 1 / y]])

                    description = generateDescription(x, M, QI, generic_implicit)

                    S = step(description=description)

                    L = S.levels[0]
                    P = L.prob

                    u0 = S.levels[0].prob.u_exact(0.0)
                    S.init_step(u0)
                    QImat = L.sweep.QI[1:, 1:]
                    Q = L.sweep.coll.Qmat[1:, 1:]
                    dt = L.params.dt
                    assert dt == x

                    # inv = np.linalg.inv(np.identity(4 * M) - x * np.kron(QImat, Aeps))
                    # K = np.matmul(inv, dt * np.kron(Q - QImat, Aeps))

                    inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - x * np.kron(QImat, A))
                    K = np.matmul(inv, dt * np.kron(Q - QImat, A))

                    inv = np.linalg.inv(np.kron(np.identity(M), I0) - x * np.kron(QImat, A))
                    K0 = np.matmul(inv, dt * np.kron(Q - QImat, A))

                    eigvals = np.linalg.eigvals(K)
                    spectral_radius = max(abs(eigvals))
                    heat_field[idx, idy] = spectral_radius

                    eigvals0 = np.linalg.eigvals(K0)
                    spectral_radius0 = max(abs(eigvals0))
                    heat_field_eps0[idx, idy] = spectral_radius0
                    # if spectral_radius >= 1:
                        # print(f"dt={x}, eps={y}, sr={spectral_radius}")

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

            plt.savefig(f"data/LinearTestDAEMinion/Talk/SPP/heatmap_SR_SPP_IterMatrix_M={M}_{QI}.png", dpi=300, bbox_inches='tight')

            plt.figure()
            print(dt_list.shape, eps.shape)
            plt.pcolor(dt_list, eps, heat_field_eps0.T, cmap='Reds', vmin=0, vmax=1)
            plt.xlim(min(dt_list), max(dt_list))
            plt.ylim(min(eps), max(eps))
            plt.xlabel(r'Time step size $\Delta t$')
            plt.ylabel(r'Perturbation parameter $\varepsilon=0$')
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)
            cbar = plt.colorbar()
            cbar.set_label('Spectral radius')

            plt.savefig(f"data/LinearTestDAEMinion/Talk/SPP/heatmap_eps0_SR_SPP_IterMatrix_M={M}_{QI}.png", dpi=300, bbox_inches='tight')


def plotSRDifferentEps():
    dt = 0.01
    eps_list = np.logspace(-9.0, 0.0, 100)#[0.0, 10 ** (-8), 10 ** (-6), 10 ** (-4), 10 ** (-2), 1.0]
    sweeper = generic_implicit
    M = 3
    QI_all = ['IE', 'LU', 'IEpar', 'Qpar', 'MIN', 'MIN-SR-S']
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange']

    A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1]])
    E3 = np.identity(4)
    E3[-1, -1] = 0

    marker = ['o', '*', 'D', 's', '^', '<']

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    spectral_radius = np.zeros((len(eps_list), len(QI_all)))
    for q, QI in enumerate(QI_all):
        for e, eps_loop in enumerate(eps_list):
            description = generateDescription(dt, M, QI, sweeper)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            Ieps = np.identity(4)
            Ieps[3, 3] = eps_loop

            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QImat, A))
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
    ax.set_ylabel(r'$\rho(\mathbf{K})$', fontsize=16)
    ax.set_ylim(0.0, 1.5)

    ax.set_xlabel(r'Parameter $\varepsilon$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/SR_IterMatrixDiffEps_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotSRIterMatrixDiffEpsM():
    dt = 0.1
    eps_list = np.logspace(-12.0, 0.0, 100)
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'IE'
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

            A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1 / eps_loop, 1 / eps_loop, 0, 1 / eps_loop]])

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(4)) - dt * np.kron(QImat, A))
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

    fig.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/SR_IterMatrixDiffEps_QI={QI}_dt={dt}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotSplittingIterMatrix():
    M_all = [3, 4, 5, 6]
    QI_all = ['IE', 'LU', 'IEpar', 'Qpar', 'MIN', 'MIN-SR-S']
    eps = 5e-3
    dt = 0.01

    A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1]])
    B = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    C = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1 / eps, 1 / eps, 0, 1 / eps]])

    a = np.cos(np.linspace(0, 2 * np.pi, 200))
    b = np.sin(np.linspace(0, 2 * np.pi, 200))

    for M in M_all:
        for QI in QI_all:
            fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.5))

            description = generateDescription(dt, M, QI, generic_implicit)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(4)) - dt * np.kron(QImat, A))
            KB = np.matmul(inv, dt * np.kron(Q - QImat, B))
            KC = np.matmul(inv, dt * np.kron(Q - QImat, C))

            eigvalsB = np.linalg.eigvals(KB)
            eigvalsC = np.linalg.eigvals(KC)

            ax[0].scatter(
                eigvalsB.real,
                eigvalsB.imag,
                marker='*',
                color='b',
                facecolors=None,
                s=200.0,
                label=r'Part without $\varepsilon$',
            )

            ax[1].scatter(
                eigvalsC.real,
                eigvalsC.imag,
                marker='*',
                color='r',
                facecolors=None,
                s=200.0,
                label=r'Part with $\varepsilon$',
            )

            for ax in [ax[0], ax[1]]:
                ax.axhline(y=0.0, color='black', linestyle='--')
                ax.axvline(x=0.0, color='black', linestyle='--')
                ax.plot(a, b, color="black")
                # ax.gca().set_aspect('equal')
                ax.legend(loc='upper right', fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.set_xlabel('Real part of eigenvalue', fontsize=16)
                ax.set_ylabel('Imaginary part of eigenvalue', fontsize=16)
                ax.set_xlim(-2.0, 2.0)
                ax.set_ylim(-2.0, 2.0)

            fig.savefig(f"data/LinearTestDAEMinion/Talk/Eigenvalues/EV_distribution_splitting_iteration_matrix_M={M}_{QI}_dt={dt}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)


def plotMaxNormIterMatrixDiffEpsM():
    dt = 0.1
    eps_list = np.logspace(-12.0, -3.15, 100)
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

            A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1 / eps_loop, 1 / eps_loop, 0, 1 / eps_loop]])

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(4)) - dt * np.kron(QImat, A))
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
    ax.legend(frameon=False, fontsize=10, loc='center left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/MaxNorm_IterMatrixDiffEps_QI={QI}_dt={dt}_{quad_type}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotErrPropagationMaxNormDiffEps():
    dt = 0.1
    eps_list = np.logspace(-12.0, 0.0, 100)
    sweeper = generic_implicit
    M = 5
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange']

    marker = ['o', '*', 'D', 's', '^', '<']

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    max_norm = np.zeros((len(eps_list), len(QI_all)))
    for q, QI in enumerate(QI_all):
        for e, eps_loop in enumerate(eps_list):
            description = generateDescription(dt, M, QI, sweeper)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1 / eps_loop, 1 / eps_loop, 0, 1 / eps_loop]])

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(4)) - dt * np.kron(QImat, A))
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
    ax.legend(frameon=False, fontsize=12, loc='upper left', ncol=2)
    ax.minorticks_off()

    fig.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/MaxNorm_IterMatrixDiffEps_M={M}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotErrPropagationMaxNormDAE():
    dt = 0.1
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange']
    lambda_d = 1
    lambda_a = 1

    marker = ['o', '*', 'D', 's', '^', '<']

    I3 = np.identity(4)
    I3[-1, -1] = 0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    max_norm = np.zeros((len(M_all), len(QI_all)))
    for q, QI in enumerate(QI_all):
        for e, M in enumerate(M_all):
            description = generateDescription(dt, M, QI, sweeper)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            A_d = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
            A_a = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 1 ]])

            # inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
            # E = np.matmul(inv, dt * np.kron(Q - QImat, A))

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

    fig.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/MaxNorm_IterMatrixDAE.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plotSRIterMatrixDAE():
    dt = 0.1
    eps_list = np.logspace(-5.0, 0.0, 100)#np.logspace(-10.0, 0.0, 100)#[0.0, 10 ** (-8), 10 ** (-6), 10 ** (-4), 10 ** (-2), 1.0]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    QI_all = ['IE', 'LU', 'IEpar', 'MIN', 'MIN-SR-S']
    colors = ['turquoise', 'deepskyblue', 'purple', 'firebrick', 'limegreen', 'orange']
    lambda_d = 1
    lambda_a = 1

    marker = ['o', '*', 'D', 's', '^', '<']

    I3 = np.identity(4)
    I3[-1, -1] = 0

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    spectral_radius = np.zeros((len(M_all), len(QI_all)))
    for q, QI in enumerate(QI_all):
        for e, M in enumerate(M_all):
            description = generateDescription(dt, M, QI, sweeper)

            S = step(description=description)

            L = S.levels[0]
            P = L.prob

            u0 = S.levels[0].prob.u_exact(0.0)
            S.init_step(u0)
            QImat = L.sweep.QI[1:, 1:]
            Q = L.sweep.coll.Qmat[1:, 1:]
            dt = L.params.dt

            A_d = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
            A_a = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 1 ]])

            # inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QImat, A))
            # E = np.matmul(inv, dt * np.kron(Q - QImat, A))

            inv = np.linalg.inv(np.kron(np.identity(M), I3) - dt * np.kron(QImat, A_d) + np.kron(np.identity(M), A_a))
            E = np.matmul(inv, dt * np.kron(Q - QImat, A_d))

            eigvals = np.linalg.eigvals(E)
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

    fig.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/SR_IterMatrixDAE.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def EVDistributionIterMatrixDiffEpsM():
    dt = 0.1
    eps_list = np.logspace(-2.0, 0.0, num=10)#[10 ** (-k) for k in range(13)]
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

                A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1 / eps_loop, 1 / eps_loop, 0, 1 / eps_loop]])

                inv = np.linalg.inv(np.kron(np.identity(M), np.identity(4)) - dt * np.kron(QImat, A))
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
            plt.xlim(-10.0, 10.0)
            plt.ylim(-10.0, 10.0)
            plt.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/EVDistributionIterMatrixDiffEps_{QI}_dt={dt}_eps={eps_loop}.png", dpi=300, bbox_inches='tight')


def EVDistributionJacobianDiffEpsM():
    dt = 0.1
    eps_list = [10 ** (-k) for k in range(13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5]#, 6, 7, 8, 9, 10, 11, 12]
    QI_all = ['IE']
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

                A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1 / eps_loop, 1 / eps_loop, 0, 1 / eps_loop]])

                J = np.kron(np.identity(M), np.identity(4)) - dt * np.kron(QImat, A)

                eigvals = np.linalg.eigvals(J)
                print(f"M={M}:", eps_loop, max(abs(eigvals)))
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
            plt.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/EVDistributionJacobianDiffEps_{QI}_dt={dt}_eps={eps_loop}.png", dpi=300, bbox_inches='tight')


def EVDistributionPreconditioner():
    r"""
    Plots the eigenvalue distribution of (I - dt * QI \otimes A)^(-1) * (I - dt * Q \otimes A) for different epsilon.
    """
    dt = 0.1
    eps_list = [10 ** (-k) for k in range(13)]
    sweeper = generic_implicit
    M_all = [2, 3, 4, 5]#, 6, 7, 8, 9, 10, 11, 12]
    QI_all = ['IE']
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

                A = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0], [1 / eps_loop, 1 / eps_loop, 0, 1 / eps_loop]])

                P = np.kron(np.identity(M), np.identity(4)) - dt * np.kron(QImat, A)
                C = np.kron(np.identity(M), np.identity(4)) - dt * np.kron(Q, A)

                eigvals = np.linalg.eigvals(np.kron(np.identity(M), np.identity(4)) - np.matmul(np.linalg.inv(P), C))
                print(f"M={M}:", eps_loop, max(abs(eigvals)))
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
            plt.savefig(f"data/EmbeddedLinearTestDAEMinion/eps_embedding/EVDistributionPreconditionerDiffEps_{QI}_dt={dt}_eps={eps_loop}.png", dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    # plot_SR_different_QDelta()
    # plot_SR_different_M()
    # plotSPPDifferentEpsilon()
    # plotSRDifferentEps()
    # plotSplittingIterMatrix()
    # plotErrPropagationMaxNormDiffEps()
    # plotErrPropagationMaxNormDAE()
    # plotSRIterMatrixDAE()
    # plotSRIterMatrixDiffEpsM()
    # EVDistributionJacobianDiffEpsM()
    # EVDistributionIterMatrixDiffEpsM()
    # EVDistributionPreconditioner()
    plotMaxNormIterMatrixDiffEpsM()