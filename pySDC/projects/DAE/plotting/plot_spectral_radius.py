import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def format_tick(x, pos):
    if abs(x) < 1e-12:
        return "0.0"
    return f"{x:.3f}"


def u_exact(t, lamb_diff=-2.0, lamb_alg=1.0):
    return np.array([np.exp(2 * lamb_diff * t), lamb_diff / lamb_alg * np.exp(2 * lamb_diff * t)])


def setup_mpl_spectrum(fontsize=16):
    my_setup_mpl(fontsize=fontsize)

    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams["lines.markersize"] = 8.0


def compute_normality_deviation(A):
    """
    Calculate the maximum norm of the commutator of A.

    The commutator of A is A*A - AA*.
    The relative maximum norm of the commutator measures how far A is from being normal.

    Parameters
    ----------
    A : numpy.ndarray
        A square matrix.

    Returns
    -------
    deviation : float
        The relative maximum norm of the commutator A*A - AA*.
    """

    A_star = np.conjugate(A.T)
    commutator = A @ A_star - A_star @ A
    deviation = np.linalg.norm(commutator, np.inf)
    denominator = np.linalg.norm(A, np.inf)
    return deviation / denominator


def get_iteration_matrices_ordered_by_nodes(dt, num_nodes, Qmat, QImat, problem_name, sweeper_type):
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

        LHS = kronI(Ieps) - dt * np.kron(QImat, A_diff) + kronI(A_alg)
        RHS = dt * np.kron(Qmat - QImat, A_diff)

    elif sweeper_type == "semiImplicitDAE":
        A_diff2 = np.array([[lamb_diff, 0], [lamb_diff, 0]])
        A_alg2 = np.array([[0, lamb_alg], [0, -lamb_alg]])

        LHS = kronI(Ieps) - dt * np.kron(QImat, A_diff2) - np.kron(I_M, A_alg2)
        RHS = dt * np.kron(Qmat - QImat, A_diff2)

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {sweeper_type}!")

    return LHS, RHS


def get_iteration_matrices_ordered_by_vars(dt, num_nodes, Qmat, QImat, problem_name, sweeper_type):
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
        return np.kron(A, I_M)

    if sweeper_type in ["embeddedDAE", "fullyImplicitDAE"]:
        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        LHS = kronI(Ieps) - dt * np.kron(A, QImat)
        RHS = dt * np.kron(A, Qmat - QImat)

    elif sweeper_type == "constrainedDAE":
        A_diff = np.array([[lamb_diff, lamb_alg], [0, 0]])
        A_alg = np.array([[0, 0], [lamb_diff, -lamb_alg]])

        LHS = kronI(Ieps) - dt * np.kron(A_diff, QImat) + kronI(A_alg)
        RHS = dt * np.kron(A_diff, Qmat - QImat)

    elif sweeper_type == "semiImplicitDAE":
        A_diff2 = np.array([[lamb_diff, 0], [lamb_diff, 0]])
        A_alg2 = np.array([[0, lamb_alg], [0, -lamb_alg]])

        LHS = kronI(Ieps) - dt * np.kron(A_diff2, QImat) - kronI(A_alg2)
        RHS = dt * np.kron(A_diff2, Qmat - QImat)

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {sweeper_type}!")

    K = np.linalg.inv(LHS) @ RHS
    return K


def axes_limits_evd():
    axes_limits = {
        2: {"x_lim": (-0.2, 0.08), "y_lim": (-0.11, 0.11)},
        3: {"x_lim": (-0.15, 0.04), "y_lim": (-0.1, 0.1)},
        4: {"x_lim": (-0.11, 0.097), "y_lim": (-0.066, 0.066)},
        5: {"x_lim": (-0.09, 0.087), "y_lim": (-0.056, 0.056)},
        6: {"x_lim": (-0.08, 0.078), "y_lim": (-0.046, 0.046)},
        7: {"x_lim": (-0.07, 0.071), "y_lim": (-0.043, 0.043)},
        8: {"x_lim": (-0.07, 0.066), "y_lim": (-0.038, 0.038)},
        9: {"x_lim": (-0.065, 0.062), "y_lim": (-0.032, 0.032)},
        10: {"x_lim": (-0.06, 0.058), "y_lim": (-0.028, 0.028)},
        11: {"x_lim": (-0.06, 0.054), "y_lim": (-0.026, 0.026)},
        12: {"x_lim": (-0.055, 0.051), "y_lim": (-0.025, 0.025)},
        13: {"x_lim": (-0.041, 0.048), "y_lim": (-0.023, 0.023)},
        14: {"x_lim": (-0.039, 0.046), "y_lim": (-0.02, 0.02)},
        15: {"x_lim": (-0.0248, 0.044), "y_lim": (-0.02, 0.02)},
        16: {"x_lim": (-0.0248, 0.042), "y_lim": (-0.02, 0.02)},
    }
    return axes_limits


def setup_blocks(K, num_nodes, problem_name):
    if problem_name != "LINEAR-TEST":
        raise NotImplementedError(f"No iteration matrix implemented for problem {problem_name}.")

    return {
        "yy": K[:num_nodes, :num_nodes],
        "yz": K[:num_nodes, num_nodes:],
        "zy": K[num_nodes:, :num_nodes],
        "zz": K[num_nodes:, num_nodes:],
    }


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
        QI_coefficients[QI] = {}

        QIGenerator = QDELTA_GENERATORS[QI]

        for num_nodes in num_nodes_list:
            QI_coefficients[QI][num_nodes] = {}

            Q = Q_coefficients[num_nodes]["matrix"]
            nodes = Q_coefficients[num_nodes]["nodes"]

            approx = QIGenerator(Q=Q, nNodes=num_nodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT", nodes=nodes)

            QImat = approx.getQDelta()

            QI_coefficients[QI][num_nodes]["matrix"] = QImat

    return QI_coefficients


# def plot_spectral_radius(
#         problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing", format="eps"
#     ):
#     figsize = figsize_by_journal(journal, scale=0.71, ratio=0.6)

#     sweeper_type = "constrainedDAE"#"semiImplicitDAE"#"constrainedDAE"
#     QI_list = ["IE", "EE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]

#     dt_list, _ = choose_time_step_sizes(problem_name)
#     dt_fix = 1e-1
#     num_nodes_list = range(2, 17)
#     num_nodes_fix = 6

#     Q_coefficients = compute_Q_coefficients(num_nodes_list)

#     QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

#     my_setup_mpl(fontsize=7)
#     colors, markers, _ = my_plot_style_config()
#     fig, axs = plt.subplots(1, 2, figsize=figsize)
#     for q, QI in enumerate(QI_list):
#         print(f"Running for {sweeper_type} with {QI}..")
#         key = f"constrainedDAE_{QI}"

#         spectral_radii_fix, spectral_radii = [], []
#         if (sweeper_type, QI) != ("fullyImplicitDAE", "Picard"):
#             for dt in dt_list:
#                 Qmat = Q_coefficients[num_nodes_fix]["matrix"]
#                 QImat = QI_coefficients[QI][num_nodes_fix]["matrix"]

#                 LHS, RHS = get_iteration_matrices_ordered_by_nodes(dt, num_nodes_fix, Qmat, QImat, problem_name, sweeper_type)
#                 K = np.linalg.inv(LHS) @ RHS

#                 deviation = compute_normality_deviation(K)
#                 print(f"{QI} - {sweeper_type} for {num_nodes_fix} nodes with {dt=}: Derivation from normality is: {deviation}\n")

#                 spectral_radius_fix = max(abs(np.linalg.eigvals(K)))
#                 spectral_radii_fix.append(spectral_radius_fix)

#             for num_nodes in num_nodes_list:
#                 Qmat = Q_coefficients[num_nodes]["matrix"]
#                 QImat = QI_coefficients[QI][num_nodes]["matrix"]

#                 LHS, RHS = get_iteration_matrices_ordered_by_nodes(dt_fix, num_nodes, Qmat, QImat, problem_name, sweeper_type)
#                 K = np.linalg.inv(LHS) @ RHS

#                 deviation = compute_normality_deviation(K)
#                 print(f"{QI} - {sweeper_type} for {num_nodes} nodes with {dt_fix=}: Derivation from normality is: {deviation}\n")

#                 spectral_radius = max(abs(np.linalg.eigvals(K)))
#                 spectral_radii.append(spectral_radius)

#             axs[0].semilogx(
#                 dt_list,
#                 spectral_radii_fix,
#                 marker=markers[key],
#                 color=colors[key],
#                 label=f"{QI}",
#             )

#             axs[1].plot(
#                 num_nodes_list,
#                 spectral_radii,
#                 marker=markers[key],
#                 color=colors[key],
#                 label=f"{QI}",
#             )

#     for ax in axs:
#         ax.tick_params(axis="both", which="minor", bottom=False, left=False)
#         ax.set_ylabel(r"spectral radius $\rho(\mathbf{K})$")

#         # ax.set_ylim((0.0, 0.3))
#         # ax.set_ylim((0.0, 1.0))

#         ax.grid(linewidth=0.5)

#     # axs[0].set_ylim((0.0, 0.25))

#     axs[0].set_xlabel(r"time step size $\Delta t$")
#     axs[1].set_xlabel(r"number of collocation nodes $M$")

#     handles, labels = axs[0].get_legend_handles_labels()

#     fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

#     plot_name = f"Fig2.{format}"
#     filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
#     file_path = Path(filename)
#     file_path.parent.mkdir(parents=True, exist_ok=True)

#     fig.savefig(filename, dpi=400, bbox_inches="tight")
#     plt.close(fig)


def plot_spectral_radius_and_evd(problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing", format="eps"):
    figsize = figsize_by_journal(journal, scale=1.9, ratio=0.6)

    sweeper_type = "constrainedDAE"
    QI_list = ["EE", "IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard"]

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_fix = 1e-1
    num_nodes = 6

    Q_coefficients = compute_Q_coefficients(num_nodes)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    setup_mpl_spectrum(fontsize=18)
    colors, markers, _ = my_plot_style_config()

    gridspec_kw = {"width_ratios": [2, 1, 1], "wspace": 0.5, "hspace": 0.6}
    fig, axs = plt.subplots(3, 3, figsize=figsize, gridspec_kw=gridspec_kw)

    outer_gs = plt.GridSpec(3, 3, width_ratios=[2, 1, 1], wspace=0.5)[:3, 0]
    big_ax = fig.add_subplot(outer_gs)

    small_axs_flatten = axs[:, 1:].flatten()

    axes_limits = axes_limits_evd()
    x_lim, y_lim = axes_limits[num_nodes]["x_lim"], axes_limits[num_nodes]["y_lim"]

    for q, QI in enumerate(QI_list):
        print(f"Running for {sweeper_type} with {QI}..")
        key = f"constrainedDAE_{QI}"

        spectral_radii = []
        if (sweeper_type, QI) != ("fullyImplicitDAE", "Picard"):
            for dt in dt_list:
                Qmat = Q_coefficients[num_nodes]["matrix"]
                QImat = QI_coefficients[QI][num_nodes]["matrix"]

                LHS, RHS = get_iteration_matrices_ordered_by_nodes(
                    dt, num_nodes, Qmat, QImat, problem_name, sweeper_type
                )
                K = np.linalg.inv(LHS) @ RHS

                deviation = compute_normality_deviation(K)
                print(
                    f"{QI} - {sweeper_type} for {num_nodes} nodes with {dt=}: Derivation from normality is: {deviation}\n"
                )

                spectral_radius = max(abs(np.linalg.eigvals(K)))
                spectral_radii.append(spectral_radius)

            big_ax.semilogx(
                dt_list,
                spectral_radii,
                marker=markers[key],
                color=colors[key],
                label=f"{QI}",
            )

            Qmat = Q_coefficients[num_nodes]["matrix"]
            QImat = QI_coefficients[QI][num_nodes]["matrix"]

            LHS, RHS = get_iteration_matrices_ordered_by_nodes(
                dt_fix, num_nodes, Qmat, QImat, problem_name, sweeper_type
            )
            K = np.linalg.inv(LHS) @ RHS

            lambdas = np.linalg.eigvals(K)

            small_axs_flatten[q].scatter(
                lambdas.real,
                lambdas.imag,
                color=colors[key],
                marker=markers[key],
                s=70,
                edgecolor="black",
                alpha=0.7,
                linewidth=0.7,
                label=f"{QI}",
            )

            # small_axs_flatten[q].set_xlabel(r"real part $\Re(\lambda)$")
            # small_axs_flatten[q].set_ylabel(r"imag. part $\Im(\lambda)$")
            small_axs_flatten[q].set_xlabel(r"$\Re(\lambda)$")
            small_axs_flatten[q].set_ylabel(r"$\Im(\lambda)$")

            small_axs_flatten[q].grid(linewidth=0.5)
            small_axs_flatten[q].set_axisbelow(True)

            small_axs_flatten[q].set_xlim(x_lim)
            small_axs_flatten[q].set_ylim(y_lim)

            # Reduce digits in ticks numbers to two and make the ticks not overlap
            small_axs_flatten[q].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            small_axs_flatten[q].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            small_axs_flatten[q].xaxis.set_major_locator(plt.MaxNLocator(2))

    big_ax.tick_params(axis="both", which="minor", bottom=False, left=False)

    big_ax.grid(linewidth=0.5)

    big_ax.set_xlabel(r"time step size $\Delta t$")
    big_ax.set_ylabel(r"spectral radius $\rho(\mathbf{K})$")

    handles, labels = big_ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.125), ncol=3)

    for ax in axs[:, 0]:
        ax.remove()

    fig.subplots_adjust(bottom=0.22)

    plot_name = f"Fig2.{format}"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400)
    plt.close(fig)


def plot_error_norm(problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing", format="eps"):
    figsize = figsize_by_journal(journal, scale=0.71, ratio=0.6)

    sweeper_type = "constrainedDAE"
    QI_list = ["IE", "EE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_fix = 1e-1
    num_nodes_list = range(2, 17)
    num_nodes_fix = 6

    Q_coefficients = compute_Q_coefficients(num_nodes_list)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    my_setup_mpl(fontsize=7)
    colors, markers, _ = my_plot_style_config()
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for q, QI in enumerate(QI_list):
        print(f"Running for {sweeper_type} with {QI}..")
        key = f"constrainedDAE_{QI}"

        error_norms_fix, error_norms = [], []
        if (sweeper_type, QI) != ("fullyImplicitDAE", "Picard"):
            for dt in dt_list:
                Qmat = Q_coefficients[num_nodes_fix]["matrix"]
                QImat = QI_coefficients[QI][num_nodes_fix]["matrix"]

                LHS, RHS = get_iteration_matrices_ordered_by_nodes(
                    dt, num_nodes_fix, Qmat, QImat, problem_name, sweeper_type
                )
                K = np.linalg.inv(LHS) @ RHS

                error_norm_fix = np.linalg.norm(K, np.inf)
                print(f"{QI} - {sweeper_type} for {num_nodes_fix} nodes with {dt=}: Error norm: {error_norm_fix}\n")

                error_norms_fix.append(error_norm_fix)

            for num_nodes in num_nodes_list:
                Qmat = Q_coefficients[num_nodes]["matrix"]
                QImat = QI_coefficients[QI][num_nodes]["matrix"]

                LHS, RHS = get_iteration_matrices_ordered_by_nodes(
                    dt_fix, num_nodes, Qmat, QImat, problem_name, sweeper_type
                )
                K = np.linalg.inv(LHS) @ RHS

                error_norm = np.linalg.norm(K, np.inf)
                print(f"{QI} - {sweeper_type} for {num_nodes} nodes with {dt_fix=}: Error norm: {error_norm}\n")

                error_norms.append(error_norm)

            axs[0].semilogx(
                dt_list,
                error_norms_fix,
                marker=markers[key],
                color=colors[key],
                label=f"{QI}",
            )

            axs[1].plot(
                num_nodes_list,
                error_norms,
                marker=markers[key],
                color=colors[key],
                label=f"{QI}",
            )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_ylabel(r"error norm $||\mathbf{K}||$")

        # ax.set_ylim((0.0, 0.3))
        # ax.set_ylim((0.0, 1.0))

        ax.grid(linewidth=0.5)

    # axs[0].set_ylim((0.0, 0.25))

    axs[0].set_xlabel(r"time step size $\Delta t$")
    axs[1].set_xlabel(r"number of collocation nodes $M$")

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    plot_name = f"Fig2a.{format}"
    filename = "data" + "/" + f"{problem_name}" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_evd(problem_name="LINEAR-TEST", journal="Springer_Scientific_Computing", format="eps"):
    figsize = figsize_by_journal(journal, scale=1.0, ratio=0.66)

    sweeper_type = "constrainedDAE"
    QI_list = ["IE", "EE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_fix = 1e-1
    num_nodes_list = range(2, 17)
    num_nodes_fix = 6

    Q_coefficients = compute_Q_coefficients(num_nodes_list)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    axes_limits = axes_limits_evd()

    my_setup_mpl(fontsize=12)
    colors, markers, _ = my_plot_style_config()

    a = np.cos(np.linspace(0, 2 * np.pi, 200))
    b = np.sin(np.linspace(0, 2 * np.pi, 200))

    for d, dt in enumerate(dt_list):
        fig, axs = plt.subplots(2, 3, figsize=figsize)
        ax_flatten = axs.flatten()

        for q, QI in enumerate(QI_list):
            key = f"constrainedDAE_{QI}"

            ax_flatten[q].plot(a, b, color="black")

            ax_flatten[q].set_title(QI)

            Qmat = Q_coefficients[num_nodes_fix]["matrix"]
            QImat = QI_coefficients[QI][num_nodes_fix]["matrix"]

            LHS, RHS = get_iteration_matrices_ordered_by_nodes(
                dt, num_nodes_fix, Qmat, QImat, problem_name, sweeper_type
            )
            K = np.linalg.inv(LHS) @ RHS

            lambdas = np.linalg.eigvals(K)

            ax_flatten[q].scatter(
                lambdas.real,
                lambdas.imag,
                color=colors[key],
                marker=markers[key],
                s=70,
                edgecolor="black",
                alpha=0.7,
                linewidth=0.7,
                label=f"{QI}",
            )

            ax_flatten[q].set_xlabel(r"real part $\Re(\lambda)$")
            ax_flatten[q].set_ylabel(r"imaginary part $\Im(\lambda)$")

            ax_flatten[q].grid(linewidth=0.5)
            ax_flatten[q].set_axisbelow(True)

        plot_name = f"{d+1}_evd_{sweeper_type}_{dt=}.{format}"
        filename = "data" + "/" + f"{problem_name}" + "/" + "evd" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

    for num_nodes in num_nodes_list:
        fig, axs = plt.subplots(2, 3, figsize=figsize)
        ax_flatten = axs.flatten()

        x_lim, y_lim = axes_limits[num_nodes]["x_lim"], axes_limits[num_nodes]["y_lim"]

        for q, QI in enumerate(QI_list):
            key = f"constrainedDAE_{QI}"

            ax_flatten[q].plot(a, b, color="black")

            ax_flatten[q].set_title(QI)

            Qmat = Q_coefficients[num_nodes]["matrix"]
            QImat = QI_coefficients[QI][num_nodes]["matrix"]

            LHS, RHS = get_iteration_matrices_ordered_by_nodes(
                dt_fix, num_nodes, Qmat, QImat, problem_name, sweeper_type
            )
            K = np.linalg.inv(LHS) @ RHS

            lambdas = np.linalg.eigvals(K)
            lambda_max = max(abs(lambdas))

            ax_flatten[q].scatter(
                lambdas.real,
                lambdas.imag,
                color=colors[key],
                marker=markers[key],
                s=70,
                edgecolor="black",
                alpha=0.7,
                linewidth=0.7,
                label=f"{QI}",
            )

            # circle = plt.Circle((0.0, 0.0), lambda_max, fill=False, linestyle='--')
            # ax_flatten[q].add_patch(circle)

            ax_flatten[q].set_xlabel(r"real part $\Re(\lambda)$")
            ax_flatten[q].set_ylabel(r"imaginary part $\Im(\lambda)$")

            ax_flatten[q].grid(linewidth=0.5)
            ax_flatten[q].set_axisbelow(True)

            ax_flatten[q].set_xlim(x_lim)
            ax_flatten[q].set_ylim(y_lim)

            # Reduce digits in ticks numbers to two and make the ticks not overlap
            ax_flatten[q].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax_flatten[q].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax_flatten[q].xaxis.set_major_locator(plt.MaxNLocator(2))

        plot_name = f"evd_{sweeper_type}_{num_nodes=}.{format}"
        filename = "data" + "/" + f"{problem_name}" + "/" + "evd" + "/" + plot_name
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    plot_spectral_radius_and_evd(format="eps")
    # plot_error_norm(format="png")
    # plot_evd(format="png")
