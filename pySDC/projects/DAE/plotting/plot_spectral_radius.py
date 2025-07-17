import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def u_exact(t, lamb_diff=-2.0, lamb_alg=1.0):
    return np.array([
        np.exp(2 * lamb_diff * t),
        lamb_diff / lamb_alg * np.exp(2 * lamb_diff * t)
    ])

def get_iteration_matrices(dt, num_nodes, Qmat, QImat, problem_name, sweeper_type):
    r"""
    Returns iteration matrix for different SDC methods.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    num_nodes : int
        Number of quadrature nodes.
    Qmat : np.2darray
        Spectral quadrature matrix.
    QImat : np.2darray
        Lower triangular matrix (for the preconditioner).
    problem_name : str
        Name of the problem.
    sweeper_type : str
        Type of the sweeper.

    Returns
    -------
    K : numpy.2darray
        Iteration matrix.
    """

    if problem_name != "LINEAR-TEST":
        raise NotImplementedError(f"No iteration matrix implemented for problem {problem_name}.")

    lamb_diff, lamb_alg = -2.0, 1.0

    Ieps = np.identity(2)
    Ieps[-1, -1] = 0
    I_M = np.identity(num_nodes)

    def kronI(A):
        return np.kron(I_M, A)

    if sweeper_type in ["embeddedDAE", "fullyImplicitDAE"]:
        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        LHS = kronI(Ieps) - dt * np.kron(QImat, A)
        RHS = dt * np.kron(Qmat - QImat, A)

    elif sweeper_type == "constrainedDAE":
        A_diff = np.array([[lamb_diff, lamb_alg], [0, 0]])
        A_alg = np.array([[0, 0], [lamb_diff, -lamb_alg]])

        LHS = kronI(Ieps) - dt * np.kron(QImat, A_diff) + np.kron(I_M, A_alg)
        RHS = dt * np.kron(Qmat - QImat, A_diff)

    elif sweeper_type == "semiImplicitDAE":
        A_diff2 = np.array([[lamb_diff, 0], [lamb_diff, 0]])
        A_alg2 = np.array([[0, lamb_alg], [0, -lamb_alg]])

        LHS = kronI(Ieps) - dt * np.kron(QImat, A_diff2) - np.kron(I_M, A_alg2)
        RHS = dt * np.kron(Qmat - QImat, A_diff2)

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {sweeper_type}!")

    return LHS, RHS


def get_rhs_collocation_problem(num_nodes, problem_name, RHS, sweeper_type, t0):
    if problem_name != "LINEAR-TEST":
        raise NotImplementedError(f"No iteration matrix implemented for problem {problem_name}.")
    
    Ieps = np.identity(2)
    Ieps[-1, -1] = 0
    I_M = np.identity(num_nodes)

    if sweeper_type == "constrainedDAE":
        u0 = u_exact(t0)
        u0_full = np.kron(np.ones(num_nodes), u0)

    elif sweeper_type in ["fullyImplicitDAE", "semiImplicitDAE"]:
        u0_full = np.kron(np.ones(num_nodes), np.zeros(2))

    b = np.kron(I_M, Ieps) @ u0_full + RHS @ u0_full

    return b, u0_full

def compute_Q_coefficients(num_nodes_list=range(2, 22, 2)):
    r"""
    Computes the Q-coefficients and store it in a dictionary.

    Parameters
    ----------
    num_nodes_list : list
        List containing different number of nodes.

    Returns
    -------
    Q_coefficients : dict
        Contains nodes and the spectral integration matrix.
    """

    QGenerator = Q_GENERATORS["coll"]

    Q_coefficients = {}

    if isinstance(num_nodes_list, int):
        num_nodes_list = [num_nodes_list]

    for num_nodes in num_nodes_list:
        Q_coefficients[num_nodes] = {}

        coll = QGenerator(nNodes=num_nodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
        Q = coll.Q
        nodes = coll.nodes
        weights = coll.weights

        Q_coefficients[num_nodes]["matrix"] = Q
        Q_coefficients[num_nodes]["nodes"] = nodes
        Q_coefficients[num_nodes]["weights"] = weights

    return Q_coefficients

def compute_QI_coefficients(Q_coefficients: dict, QI_list: list):
    r"""
    Computes the QI-coefficients and store it in a dictionary.

    Parameters
    ----------
    Q_coefficients : dict
        Contains Q-coefficients.
    QI_list : list
        Contains names of different types of :math:`Q_\Delta`.

    Returns
    -------
    QI_coefficients : dict
        Contains :math:`Q_\Delta` matrix.
    """

    num_nodes_list = list(Q_coefficients.keys())
    QI_coefficients = {}

    for QI in QI_list:
        QI_coefficients[QI]= {}

        QIGenerator = QDELTA_GENERATORS[QI]

        for num_nodes in num_nodes_list:
            QI_coefficients[QI][num_nodes] = {}

            Q = Q_coefficients[num_nodes]["matrix"]
            nodes = Q_coefficients[num_nodes]["nodes"]

            approx = QIGenerator(Q=Q, nNodes=num_nodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT", nodes=nodes)

            QImat = approx.getQDelta()

            QI_coefficients[QI][num_nodes]["matrix"] = QImat

    return QI_coefficients

def plot_spectral_radius(
        problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing"
    ):
    figsize = figsize_by_journal(journal, scale=0.71, ratio=0.6)

    sweeper_type = "constrainedDAE"
    QI_list = ["IE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]

    dt_list = choose_time_step_sizes(problem_name)
    dt_fix = 1e-1
    num_nodes_list = range(2, 17)
    num_nodes_fix = 6

    Q_coefficients = compute_Q_coefficients(num_nodes_list)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    my_setup_mpl(fontsize=8)
    colors, markers, _ = my_plot_style_config()
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for q, QI in enumerate(QI_list):
        print(f"Running for {sweeper_type} with {QI}..")
        key = f"constrainedDAE_{QI}"

        spectral_radii_fix, spectral_radii = [], []
        if (sweeper_type, QI) != ("fullyImplicitDAE", "Picard"):
            for dt in dt_list:
                Qmat = Q_coefficients[num_nodes_fix]["matrix"]
                QImat = QI_coefficients[QI][num_nodes_fix]["matrix"]

                LHS, RHS = get_iteration_matrices(dt, num_nodes_fix, Qmat, QImat, problem_name, sweeper_type)
                K = np.linalg.inv(LHS) @ RHS

                spectral_radius_fix = max(abs(np.linalg.eigvals(K)))
                spectral_radii_fix.append(spectral_radius_fix)

            for num_nodes in num_nodes_list:
                Qmat = Q_coefficients[num_nodes]["matrix"]
                QImat = QI_coefficients[QI][num_nodes]["matrix"]

                LHS, RHS = get_iteration_matrices(dt_fix, num_nodes, Qmat, QImat, problem_name, sweeper_type)
                K = np.linalg.inv(LHS) @ RHS

                spectral_radius = max(abs(np.linalg.eigvals(K)))
                # U, S, Vh = np.linalg.svd(K, full_matrices=True)
                # spectral_radius = max(abs(S))
                spectral_radii.append(spectral_radius)

            axs[0].semilogx(
                dt_list,
                spectral_radii_fix,
                marker=markers[key],
                color=colors[key],
                label=f"{QI}",
            )

            axs[1].plot(
                num_nodes_list,
                spectral_radii,
                marker=markers[key],
                color=colors[key],
                label=f"{QI}",
            )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_ylabel("spectral radius")

        ax.set_ylim((0.0, 0.3))

    axs[0].set_xlabel(r"$\Delta t$")
    axs[1].set_xlabel("number of collocation nodes")

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    plot_name = "Fig2.eps"  # f"spectral_radius_{sweeper_type}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_spectral_radius()
