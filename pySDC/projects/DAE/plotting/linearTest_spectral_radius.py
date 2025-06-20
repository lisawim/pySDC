import numpy as np

from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.projects.DAE import computeNormalityDeviation, getColor, getLabel, get_linestyle, getMarker, Plotter

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def get_iteration_matrices(dt, eps, M, Qmat, QImat, problem_name, problem_type):
    r"""
    Returns iteration matrix for different SDC methods.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    eps : float
        Perturbation parameter :math:`\varepsilon` of singular perturbed problems.
    M : int
        Number of quadrature nodes.
    Qmat : np.2darray
        Spectral quadrature matrix.
    QImat : np.2darray
        Lower triangular matrix (for the preconditioner).
    problem_name : str
        Name of the problem.
    problem_type : str
        Type of the problem.

    Returns
    -------
    K : numpy.2darray
        Iteration matrix.
    """

    if problem_name == "LINEAR-TEST":
        lamb_diff = -2.0
        lamb_alg = 1.0

        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        Ieps = np.identity(2)
        Ieps[-1, -1] = eps

        if problem_type in ["SPP", "SPP-yp", "embeddedDAE"]:
            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QImat, A))
            K = np.matmul(inv, dt * np.kron(Qmat - QImat, A))

        elif problem_type == "constrainedDAE":
            A_d_eq = np.array([[lamb_diff, lamb_alg], [0, 0]])
            A_a_eq = np.array([[0, 0], [lamb_diff, -lamb_alg]])

            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QImat, A_d_eq) + np.kron(np.identity(M), A_a_eq))
            K = np.matmul(inv, dt * np.kron(Qmat - QImat, A_d_eq))

        elif problem_type == "fullyImplicitDAE":
            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QImat, A))
            K = np.matmul(inv, dt * np.kron(Qmat - QImat, A))
            # K_tilde = np.matmul(dt * np.kron(Qmat, np.identity(2)), np.matmul(inv, dt * np.kron(Qmat - QImat, A)))
            # K = K_tilde

        elif problem_type == "semiImplicitDAE":
            A_d_var = np.array([[lamb_diff, 0], [lamb_diff, 0]])
            A_a_var = np.array([[0, lamb_alg], [0, -lamb_alg]])

            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QImat, A_d_var) - np.kron(np.identity(M), A_a_var))
            K = np.matmul(inv, dt * np.kron(Qmat - QImat, A_d_var))

        else:
            raise NotImplementedError(f"No iteration matrix implemented for {problem_type}!")
        
    elif problem_name == "PROTHERO-ROBINSON":
        if problem_type in ["SPP", "SPP-yp"]:
            inv = np.linalg.inv(eps * np.identity(M) + dt * QImat)
            K = np.matmul(inv, -dt * (Qmat - QImat))

        elif problem_type == "embeddedDAE":
            K = np.identity(M) - np.matmul(np.linalg.inv(QImat), Qmat)

        elif problem_type == "constrainedDAE":
            K = np.zeros((M, M))

        elif problem_type == "fullyImplicitDAE":
            K = np.identity(M) - np.matmul(np.linalg.inv(QImat), Qmat)

        elif problem_type == "semiImplicitDAE":
            K = np.zeros((M, M))

        else:
            raise NotImplementedError(f"No iteration matrix implemented for {problem_type}!")

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {problem_name}!")
    
    return K


def get_nodes_range_of_convergence(problem_name, node_every=1):
    r"""
    Returns range of nodes where problem converges.

    Parameters
    ----------
    problem_name : str
        Name of problem.
    node_every : int, optional
        If ``node_every=2`` range contains every second node in range.

    Returns
    -------
    num_nodes_list : np.1darray
        Range of nodes.
    """

    if problem_name == "LINEAR-TEST":
        num_nodes_list = range(2, 16 + node_every, node_every)
    elif problem_name in ["MICHAELIS-MENTEN", "PROTHERO-ROBINSON"]:
        num_nodes_list = range(2, 10 + node_every, node_every)
    else:
        raise NotImplementedError()

    return num_nodes_list


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

    QGenerator = Q_GENERATORS['coll']

    Q_coefficients = {}

    if isinstance(num_nodes_list, int):
        num_nodes_list = [num_nodes_list]

    for num_nodes in num_nodes_list:
        print(f"\n Generate collocation matrix for {num_nodes} nodes\n")

        Q_coefficients[num_nodes] = {}

        coll = QGenerator(nNodes=num_nodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
        Q = coll.Q
        nodes = coll.nodes
        weights = coll.weights

        Q_coefficients[num_nodes]["matrix"] = Q
        Q_coefficients[num_nodes]["nodes"] = nodes
        Q_coefficients[num_nodes]["weights"] = weights

    return Q_coefficients

def compute_QI_coefficients(Q_coefficients: dict, QI_list: list, k_flex=None):
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
            print(f"\n Generate {QI} for {num_nodes} nodes\n")

            k_flex = k_flex if k_flex is not None else None

            QI_coefficients[QI][num_nodes] = {}

            Q = Q_coefficients[num_nodes]['matrix']
            nodes = Q_coefficients[num_nodes]['nodes']

            approx = QIGenerator(Q=Q, nNodes=num_nodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT", nodes=nodes)

            QImat = approx.getQDelta(k=k_flex)

            QI_coefficients[QI][num_nodes]['matrix'] = QImat

    return QI_coefficients

def finalize_plot(dt, k, k_flex, num_nodes_list, problem_name, QI_list, spectral_radii_plotter):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    dt : float
        Time step size.
    k : int
        Case number.
    num_nodes_list : list
        List contains different number of collocation nodes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    spectral_radii_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17}

    for q, QI in enumerate(QI_list):
        spectral_radii_plotter.set_xticks(num_nodes_list[0::2], subplot_index=q)
        spectral_radii_plotter.set_xlabel('number of nodes', subplot_index=q)

        spectral_radii_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

    spectral_radii_plotter.set_ylabel('spectral radius', subplot_index=None)

    spectral_radii_plotter.set_ylim((-0.03, 1.12), subplot_index=None)
    spectral_radii_plotter.set_ylim((-0.03, 10.0), subplot_index=3)

    spectral_radii_plotter.adjust_layout(num_subplots=len(QI_list))

    spectral_radii_plotter.set_grid()

    bbox_pos = bbox_position[k]
    spectral_radii_plotter.set_shared_legend(loc='lower center', bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"spectral_radius_{dt=}_case{k}_{k_flex=}.png"
    spectral_radii_plotter.save(filename)

if __name__ == "__main__":
    problem_name = "LINEAR-TEST"
    # problem_name = "PROTHERO-ROBINSON"

    QI_list = ["IE", "LU", "MIN-SR-S"]  # , 'MIN-SR-FLEX']
    num_nodes_list = get_nodes_range_of_convergence(problem_name, node_every=1)

    k_flex = 1

    dt = 1e-2
    case = 6

    problems = get_problem_cases(k=case, problem_name=problem_name)

    Q_coefficients = compute_Q_coefficients(num_nodes_list)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list, k_flex)

    spectral_radii_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                spectral_radii = []

                for num_nodes in num_nodes_list:

                    Qmat = Q_coefficients[num_nodes]["matrix"]
                    QImat = QI_coefficients[QI][num_nodes]["matrix"]

                    K = get_iteration_matrices(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type)

                    spectral_radius = max(abs(np.linalg.eigvals(K)))
                    spectral_radii.append(spectral_radius)

                    derivation = computeNormalityDeviation(K)
                    print(f"{QI}-{problem_type} for {num_nodes} nodes: Derivation from normality is: {derivation}\n")

                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)

                marker, markersize = res["marker"], res["markersize"]
                plot_result(
                    spectral_radii_plotter,
                    num_nodes_list,
                    spectral_radii,
                    q,
                    color,
                    marker,
                    markersize,
                    linestyle,
                    problem_label,
                    plot_type="plot",
                )

    finalize_plot(dt, case, k_flex, num_nodes_list, problem_name, QI_list, spectral_radii_plotter)
