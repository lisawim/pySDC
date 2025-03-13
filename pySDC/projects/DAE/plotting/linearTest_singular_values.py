import numpy as np

from pySDC.projects.DAE.run.error import get_problem_cases
from pySDC.projects.DAE.plotting.linearTest_spectral_radius import (
    compute_Q_coefficients, compute_QI_coefficients, get_iteration_matrices
)
from pySDC.projects.DAE import getColor, getLabel, Plotter


def get_Jacobians(dt, eps, M, QImat, problem_name, problem_type):
    r"""
    Returns Jacobians for different SDC methods.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    eps : float
        Perturbation parameter :math:`\varepsilon` of singular perturbed problems.
    M : int
        Number of quadrature nodes.
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

    # Define the problem parameters
    lamb_diff, lamb_alg = -2.0, 1.0

    A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

    Ieps = np.identity(2)
    Ieps[-1, -1] = eps

    IM = np.identity(M)

    if problem_type in ["SPP", "SPP-yp", "embeddedDAE", "fullyImplicitDAE"]:
        J = np.kron(IM, Ieps) - dt * np.kron(QImat, A)

    elif problem_type == "constrainedDAE":
        A_d_eq = np.array([[lamb_diff, lamb_alg], [0, 0]])
        A_a_eq = np.array([[0, 0], [lamb_diff, -lamb_alg]])

        J = np.kron(IM, Ieps) - dt * np.kron(QImat, A_d_eq) + np.kron(IM, A_a_eq)

    elif problem_type == "semiImplicitDAE":
        A_d_var = np.array([[lamb_diff, 0], [lamb_diff, 0]])
        A_a_var = np.array([[0, lamb_alg], [0, -lamb_alg]])

        J = np.kron(IM, Ieps) - dt * np.kron(QImat, A_d_var) - np.kron(IM, A_a_var)

    else:
        raise NotImplementedError("No Jacobian implemented for {}!".format(problem_type))
    return J


def get_x_indices(k: int):
    r"""
    Returns indices for x-axis.

    Parameters
    ----------
    k : int
        Case number.

    Returns
    -------
    x_indices : dict
        Indices for x-axis for different problem types.
    """

    eps_list = list(range(1, 13))

    mapping = {
        1: {"SPP": eps_list, "embeddedDAE": [eps_list[-1]], "constrainedDAE": [eps_list[-1] + 1]},
        2: {"SPP-yp": eps_list, "fullyImplicitDAE": [eps_list[-1] + 1], "semiImplicitDAE": [eps_list[-1] + 2]},
        3: {"SPP": list(range(1, 23, 2)), "SPP-yp": list(range(2, 24, 2))},
        4: {"constrainedDAE": [1], "embeddedDAE": [2], "fullyImplicitDAE": [3], "semiImplicitDAE": [4]},
        5: {"fullyImplicitDAE": [1], "semiImplicitDAE": [2]},
        6: {"SPP": eps_list, "constrainedDAE": [eps_list[-1]], "embeddedDAE": [eps_list[-1] + 1],
            "fullyImplicitDAE": [eps_list[-1] + 2], "semiImplicitDAE": [eps_list[-1] + 3]},
    }
    
    if case in mapping:
        return mapping[case]
    else:
        raise NotImplementedError("No indices for x-axis implemented for case {}!".format(case))


def scatter_result(plotter: Plotter, x, y, color, plot_values_for, problem_label, subplot_indices_QI, threshold=1e-6):
    r"""
    Plots the results using a scatter plot.

    Parameters
    ----------
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    x : list or numpy.1darray
        Values for x-axis.
    y : list or numpy.1darray
        Values for y-axis.
    color : str
        Problem-specific color.
    plot_values_for : str
        Contains information for what data is plotted.
    problem_label : str
        Label for plot.
    subplot_indices_QI : tuple
        Contains subplot indices where quantity is plotted in.

    Returns
    -------
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Updated plotter class.
    """

    if plot_values_for == "iteration_matrix":
        x = np.array(x)
        y = np.array(y)

        # # Separate the data
        x_upper = x[y > threshold]
        x_lower = x[(y <= threshold) & (y > 0)]

        y_upper = y[y > threshold]
        y_lower = y[(y <= threshold) & (y > 0)]

        # Plot y-data with larger values first on primary axis
        for x_u, y_u in zip(x_upper, y_upper):
            plotter.scatter(
                x_u,
                y_u,
                color=color,
                marker='.',
                label=problem_label,
                s=450,
                edgecolor="black",
                alpha=1.0,
                subplot_index=subplot_indices_QI[0],
            )

        for x_l, y_l in zip(x_lower, y_lower):
            plotter.scatter(
                x_l,
                y_l,
                color=color,
                marker='.',
                label=problem_label,
                s=450,
                edgecolor="black",
                alpha=1.0,
                subplot_index=subplot_indices_QI[1],
            )
    else:
        plotter.scatter(
            x,
            y,
            color=color,
            marker='.',
            label=problem_label,
            s=450,
            edgecolor="black",
            alpha=1.0,
            subplot_index=subplot_indices_QI[0],
        )

    return plotter


def finalize_plot(dt, k, num_nodes, plot_values_for, problem_name, QI_list, singular_values_plotter, subplot_indices):
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
    plot_values_for : str
        Values are plotted either for iteration matrix or Jacobian.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    singular_values_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    # Dictionary for legend bounding box positions.
    bbox_positions = {
        "iteration_matrix": {1: -0.22, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.25},
        "jacobian": {1: -0.41, 2: -0.36, 3: -0.44, 4: -0.24, 5: -0.24, 6: -0.44},
    }

    if plot_values_for == "iteration_matrix":
        y_ticks_upper = [10 ** (m) for m in range(-4, 2)]
        y_labels_upper = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$", r"$10^1$"]

        y_ticks_lower = [10 ** (m) for m in range(-20, -12)]
        y_labels_lower = [r"$10^{-20}$", r"$10^{-19}$", r"$10^{-18}$", r"$10^{-17}$", r"$10^{-16}$", r"$10^{-15}$", r"$10^{-14}$", r"$10^{-13}$"]

    for q, QI in enumerate(QI_list):
        sub_id = subplot_indices[q]

        # Set titles
        singular_values_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=sub_id[0], fontsize=26)

        if plot_values_for == "iteration_matrix":
            singular_values_plotter.set_yticks(y_ticks_upper, y_labels_upper, subplot_index=sub_id[0], fontsize=22)
            singular_values_plotter.set_yticks(y_ticks_lower, y_labels_lower, subplot_index=sub_id[1], fontsize=22)

            # Set y-limits
            singular_values_plotter.set_ylim((3e-4, 1e1), scale="log", subplot_index=sub_id[0])
            singular_values_plotter.set_ylim((1.5e-20, 7e-16), scale="log", subplot_index=sub_id[1])

            singular_values_plotter.set_xlabel(r"Parameter $\varepsilon$", fontsize=28, subplot_index=sub_id[1])

        elif plot_values_for == "jacobian":
            singular_values_plotter.set_ylim((1.5e-4, 3.5e0), scale="log", subplot_index=q)

            singular_values_plotter.set_xlabel(r"Parameter $\varepsilon$", fontsize=28, subplot_index=q)

    singular_values_plotter.set_ylabel('Singular value', fontsize=28)

    singular_values_plotter.set_xticks(clear_labels=True)

    bbox_pos = bbox_positions[plot_values_for].get(case, -0.3)
    singular_values_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=28)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"singular_values_{plot_values_for}_{num_nodes=}_{dt=}_case{k}.png"
    singular_values_plotter.save(filename)


if __name__ == "__main__":
    problem_name = "LINEAR-TEST"

    QI_list = ["IE", "LU", "MIN-SR-S"]  # ["IE", "LU", "MIN-SR-S", "MIN-SR-FLEX"]

    num_nodes_list = range(2, 22, 2)

    dt = 1e-2

    case = 6

    plot_values_for = "iteration_matrix"  # "jacobian"

    problems = get_problem_cases(k=case)

    Q_coefficients = compute_Q_coefficients(num_nodes_list)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    x_indices = get_x_indices(k=case)

    subplot_indices = {0: (0, 3), 1: (1, 4), 2: (2, 5)}

    for num_nodes in num_nodes_list:
        if plot_values_for == "iteration_matrix":
            singular_values_plotter = Plotter(nrows=2, ncols=3, figsize=(18, 12), hspace=0.05)
        elif plot_values_for == "jacobian":
            singular_values_plotter = Plotter(nrows=1, ncols=3, figsize=(18, 6), hspace=0.05)

        for q, QI in enumerate(QI_list):

            for problem_type, eps_values in problems.items():
                for i, eps in enumerate(eps_values):
                    Qmat = Q_coefficients[num_nodes]["matrix"]
                    QImat = QI_coefficients[QI][num_nodes]["matrix"]

                    if plot_values_for == "iteration_matrix":
                        A = get_iteration_matrices(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type)
                    elif plot_values_for == "jacobian":
                        A = get_Jacobians(dt, eps, num_nodes, QImat, problem_name, problem_type)
                    else:
                        raise NotImplementedError(f"Plot type {plot_values_for} not implemented.")

                    U, S, Vh = np.linalg.svd(A, full_matrices=True)

                    ind = x_indices[problem_type][i]
                    eps_x = [ind] * len(S)

                    # if plot_values_for == "jacobian":
                    condition_number = np.max(S) / np.min(S)  # np.linalg.norm(A, 2) * np.linalg.norm(np.linalg.inv(A), 2)
                    print(f"{QI} with {num_nodes} nodes: For {problem_type} with {eps=} condition is: {condition_number}\n")

                    color = getColor(problem_type, i, QI)
                    problem_label = getLabel(problem_type, eps, QI)

                    sub_id = subplot_indices[q]

                    singular_values_plotter = scatter_result(singular_values_plotter, eps_x, S, color, plot_values_for, problem_label, sub_id)

        finalize_plot(dt, case, num_nodes, plot_values_for, problem_name, QI_list, singular_values_plotter, subplot_indices)
