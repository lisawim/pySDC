import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pySDC.core.Step import step
from pySDC.projects.DAE.sweepers.genericImplicitEmbedded import (
    genericImplicitEmbedded,
    genericImplicitEmbedded2,
)
from pySDC.projects.DAE.plotting.error_propagation_Minion import generateDescription


def getCoefficientMatrix(prob_cls_name):
    r"""
    Returns the coefficient matrix of a DAE problem of name ``prob_cls_name``.
    Further, the matrices corresponding to differential and algebraic part are
    returned.

    Parameters
    ----------
    prob_cls_name : str
        Name of the problem.

    Returns
    -------
    Ad : np.2darray
        Matrix corresponding to differential part.
    Aa : np.2darray
        Matrix corresponding to algebraic part.
    A : np.2darray
        Entire coefficient matrix.
    """

    if prob_cls_name == 'LinearTestDAEIntegralFormulation':
        Ad = np.array([[1, 1]])
        Aa = np.array([[1, -1]])
    elif prob_cls_name == 'LinearTestDAEMinionIntegralFormulation':
        Ad = np.array([[1, 0, -1, 1], [0, -1e4, 0, 0], [1, 0, 0, 0]])
        Aa = np.array([[1, 1, 0, 1]])
    elif prob_cls_name == 'LinearIndexTwoDAEIntegralFormulation':
        Ad = np.array([[1, 0, 0], [2, -1e5, 1]])
        Aa = np.array([[1, 1, 0]])
    else:
        raise NotImplementedError()
    
    A = np.concatenate((Ad, Aa), axis=0)
    return Ad, Aa, A


def plotEmbeddedScheme():
    r"""
    Function plots the norm and determinant of the matrix of the system to be solved on each node for
    all problems. Here, the matrix is considered that arises when in the embedded scheme :math:`\varepsilon`
    is set to 0.
    """

    problems = {
        'LinearTestDAEIntegralFormulation',
        'LinearTestDAEMinionIntegralFormulation',
        'LinearIndexTwoDAEIntegralFormulation',
        # 'simple_dae_1IntegralFormulation',
    }

    Path("data").mkdir(parents=True, exist_ok=True)
    for prob_cls_name in problems:
        Path(f"data/{prob_cls_name}").mkdir(parents=True, exist_ok=True)

    dt_list = np.logspace(-3.0, 0.5, num=50)
    sweeper = genericImplicitEmbedded
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'IE'
    quad_type = 'RADAU-RIGHT'
    colors = [
        'turquoise',
        'deepskyblue',
        'purple',
        'firebrick',
        'limegreen',
        'orange',
        'plum',
        'salmon',
        'forestgreen',
        'midnightblue',
        'gold',
        'silver',
    ]

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P', 'h']

    determinant = np.zeros((len(dt_list), len(M_all)))
    for prob_cls_name in problems:
        fig, ax = plt.subplots(figsize=(9.5, 9.5))

        Ad, Aa, A = getCoefficientMatrix(prob_cls_name)

        Nd = Ad.shape[0]
        Na = Aa.shape[0]
        N = Nd + Na

        Ed, Ea = np.zeros((N, N)), np.zeros((N, N))
        for m in range(N):
            Ed[m, m] = 1 if m < Nd else 0
            Ea[m, m] = 1 if m >= Nd else 0

        for q, M in enumerate(M_all):
            for e, dt_loop in enumerate(dt_list):
                description = generateDescription(dt_loop, M, QI, sweeper, quad_type)

                S = step(description=description)

                L = S.levels[0]
                P = L.prob

                u0 = S.levels[0].prob.u_exact(0.0)
                S.init_step(u0)
                QImat = L.sweep.QI[1:, 1:]
                dt = L.params.dt

                J = np.kron(np.identity(M), Ed) - dt * np.kron(QImat, A)
                # detProd = 1
                # for m in range(M):
                #     print(J[m * N : m * N + N, m * N : m * N + N])
                #     print(np.linalg.det(J[m * N : m * N + N, m * N : m * N + N]))
                #     detProd *= np.linalg.det(J[m * N : m * N + N, m * N : m * N + N])
                # print(detProd)
                # print(np.linalg.det(J))
                determinant[e, q] = np.linalg.det(J)

        for q_plot, M_plot in enumerate(M_all):
            ax.loglog(
                dt_list,
                determinant[:, q_plot],
                color=colors[q_plot],
                marker=marker[q_plot],
                markeredgecolor='k',
            )
        
        ax.set_ylim(1e-15, 1e1)
        ax.set_ylabel(r'$\det(J)$', fontsize=16)

        ax.set_xscale('log', base=10)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'$\Delta t$', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(visible=True)
        ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
        ax.minorticks_off()

        fig.savefig(f"data/{prob_cls_name}/DeterminantJacobianEmbeddedScheme_QI={QI}_{quad_type}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def plotEmbeddedScheme2():
    r"""
    Function plots the norm and determinant of the matrix of the system to be solved on each node for
    all problems. Here, the matrix is considered that arises when we replace the discretization of the
    algebraic constraints arising in the embedded scheme by :math:`0 = g(y^{k+1}, z^{k+1})`.
    """

    problems = {
        'LinearTestDAEIntegralFormulation',
        'LinearTestDAEMinionIntegralFormulation',
        'LinearIndexTwoDAEIntegralFormulation',
        # 'simple_dae_1IntegralFormulation',
    }

    Path("data").mkdir(parents=True, exist_ok=True)
    for prob_cls_name in problems:
        Path(f"data/{prob_cls_name}").mkdir(parents=True, exist_ok=True)

    dt_list = np.logspace(-3.0, 2.0, num=50)
    sweeper = genericImplicitEmbedded2
    M_all = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    QI = 'IE'
    quad_type = 'RADAU-RIGHT'
    colors = [
        'turquoise',
        'deepskyblue',
        'purple',
        'firebrick',
        'limegreen',
        'orange',
        'plum',
        'salmon',
        'forestgreen',
        'midnightblue',
        'gold',
        'silver',
    ]

    marker = ['o', '*', 'D', 's', '^', '<', '>', 'd', '8', 'p', 'P', 'h']

    determinant = np.zeros((len(dt_list), len(M_all)))
    for prob_cls_name in problems:
        fig, ax = plt.subplots(figsize=(9.5, 9.5))

        Ad, Aa, A = getCoefficientMatrix(prob_cls_name)

        Nd = Ad.shape[0]
        Na = Aa.shape[0]
        N = Nd + Na

        B = np.concatenate((Ad, np.zeros((Na, N))), axis=0)
        C = np.concatenate((np.zeros((Nd, N)), Aa), axis=0)

        Ed, Ea = np.zeros((N, N)), np.zeros((N, N))
        for m in range(N):
            Ed[m, m] = 1 if m < Nd else 0
            Ea[m, m] = 1 if m >= Nd else 0

        for q, M in enumerate(M_all):
            for e, dt_loop in enumerate(dt_list):
                description = generateDescription(dt_loop, M, QI, sweeper, quad_type)

                S = step(description=description)

                L = S.levels[0]
                P = L.prob

                u0 = S.levels[0].prob.u_exact(0.0)
                S.init_step(u0)
                QImat = L.sweep.QI[1:, 1:]
                dt = L.params.dt
                # print(B)
                J = np.kron(np.identity(M), Ed) - dt * np.kron(QImat, B) - np.kron(np.identity(M), C)
                # detProd = 1
                # for m in range(M):
                #     print(J[m * N : m * N + N, m * N : m * N + N])
                #     print(np.linalg.det(J[m * N : m * N + N, m * N : m * N + N]))
                #     detProd *= np.linalg.det(J[m * N : m * N + N, m * N : m * N + N])
                # print(detProd)
                # print(np.linalg.det(J))
                determinant[e, q] = np.linalg.det(J)

        for q_plot, M_plot in enumerate(M_all):
            ax.semilogx(
                dt_list,
                determinant[:, q_plot],
                color=colors[q_plot],
                marker=marker[q_plot],
                markeredgecolor='k',
            )
        
        ax.set_ylabel(r'$\det(J)$', fontsize=16)

        ax.set_xscale('log', base=10)
        ax.set_yscale('symlog', linthresh=1e-4)
        ax.set_xlabel(r'$\Delta t$', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(visible=True)
        ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=2)
        ax.minorticks_off()
        ax.set_ylim(-1e11, 1e11)

        fig.savefig(f"data/{prob_cls_name}/DeterminantJacobianEmbeddedScheme2_QI={QI}_{quad_type}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    plotEmbeddedScheme()
    plotEmbeddedScheme2()
