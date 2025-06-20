import numpy as np

from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.projects.DAE.plotting.linearTest_spectral_radius import (
    compute_Q_coefficients, compute_QI_coefficients, get_nodes_range_of_convergence
)
from pySDC.projects.DAE import getColor, getLabel, get_linestyle, getMarker, Plotter


def get_sweeper_mats(dt, eps, num_nodes, Q, QI, problem_name, problem_type):
        """
        Returns matrices for doing the sweeps applied to Prothero-Robinson.

        Parameters
        ----------
        dt : float
            Time step size.
        eps : float
            Perturbation parameter of Prothero-Robinson problem.
        num_nodes : int
            Number of collocation nodes.
        Q : np.2darray
            Spectral integration matrix.
        QI : np.2darray
            Lower-triangular or diagonal matrix.
        problem_name : str
            Name of the problem.
        problem_type : str
            Type of problem (if we solve an ODE (SPP) or a DAE).

        Returns
        -------
        L : np.2darray
            Left-hand side matrix of SDC formulation.
        R : np.2darray
            Right-hand side matric of SDC formulation.
        """

        if problem_name == "LINEAR-TEST":
            lamb_diff = -2.0
            lamb_alg = 1.0

            A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

            Ieps = np.identity(2)
            Ieps[-1, -1] = eps

            if problem_type in ["SPP", "embeddedDAE"]:
                L = np.kron(np.identity(num_nodes), Ieps) - dt * np.kron(QI, A)
                R = dt * np.kron(Q - QI, A)

            elif problem_type == "constrainedDAE":
                A_d_eq = np.array([[lamb_diff, lamb_alg], [0, 0]])
                A_a_eq = np.array([[0, 0], [lamb_diff, -lamb_alg]])

                L = np.kron(np.identity(num_nodes), Ieps) - dt * np.kron(QI, A_d_eq) + np.kron(np.identity(num_nodes), A_a_eq)
                R = dt * np.kron(Q - QI, A_d_eq)

        elif problem_name == "PROTHERO-ROBINSON":
            if problem_type == "SPP":
                L = np.identity(num_nodes) + (dt / eps) *  QI
                R = (-dt / eps) * (Q - QI)

            elif problem_type == "constrainedDAE":
                L = np.identity(num_nodes)
                R = np.zeros((num_nodes, num_nodes))

            elif problem_type == "embeddedDAE":
                L = dt * QI
                R = -dt * (Q - QI)

        elif problem_name == "DAHLQUIST-PROTHERO-ROBINSON":
            if problem_type == "SPP":
                L = np.identity(num_nodes) + (dt / eps) *  QI
                R = (-dt / eps) * (Q - QI)

            elif problem_type == "constrainedDAE":
                L = np.identity(num_nodes)
                R = np.zeros((num_nodes, num_nodes))

            elif problem_type == "embeddedDAE":
                L = dt * QI
                R = -dt * (Q - QI)

        return L, R


def perform_sweeps(L, R, num_sweeps, problem_name, problem_type):
        r"""
        ``nsweeps`` sweeps in matrix form are performed for Prothero-Robinson. It can be presented in a series.

        Parameters
        ----------
        num_sweeps : int
            Number of sweeps.
        L : np.2darray
            Left-hand side matrix of SDC formulation.
        R : np.2darray
            Right-hand side matric of SDC formulation.
        problem_name : str
            Name of problem.
        problem_type : str
            Type of problem.

        Returns
        -------
        Mat_sweep : np.2darray
            Matrix sweeps.
        """

        L_inv = np.linalg.inv(L)

        Mat_sweep = np.linalg.matrix_power(L_inv.dot(R), num_sweeps)

        if problem_name == "LINEAR-TEST" and problem_type in ["SPP", "embeddedDAE"]:
            Ieps = np.identity(2)
            Ieps[-1, -1] = 0

            M = np.kron(np.identity(num_nodes), Ieps)

            for k in range(0, num_sweeps):
                Mat_sweep += np.linalg.matrix_power(L_inv.dot(R), k).dot(L_inv.dot(M))

        else:
            for k in range(0, num_sweeps):
                Mat_sweep += np.linalg.matrix_power(L_inv.dot(R), k).dot(L_inv)

        return Mat_sweep


def perform_collocation_update(do_coll_update, dt, eps, Mat_sweep, num_nodes, problem_name, problem_type, weights):
    r"""
    Performs the collocation update after a fixed number of sweeps.

    Parameters
    ----------
    do_coll_update : bool
        If True, collocation update is computed.
    dt : float
        Time step size.
    eps : float
        Perturbation parameter.
    Mat_sweep : np.2darray
        Matrix sweeps.
    num_nodes : int
        Number of collocation nodes.
    problem_name : str
        Name of the problem.
    problem_type : str
        Type of problem (whether we solve the SPP or the DAE).
    weights : np.1darray
        Weights of the collocation method.

    Returns
    -------
    stabi_function : float
        Stability function.
    """

    if problem_name == "LINEAR-TEST":
        lamb_diff = -2.0
        lamb_alg = 1.0

        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        Ieps = np.identity(2)
        Ieps[-1, -1] = eps

        q = np.zeros(2 * num_nodes)
        q[2 * num_nodes - 2] = 1.0
        q[2 * num_nodes - 1] = 1.0

        if problem_type in ["SPP", "embeddedDAE"]:
            if do_coll_update:
                b = Mat_sweep.dot(np.ones(2 * num_nodes))

                stabi_function = np.array([1.0, 0.0])
                for j in range(num_nodes):
                    stabi_function[:] += dt * weights[j] * A.dot(b[2 * j : 2 * (j + 1)])

            else:
                stabi_function = q.dot(Mat_sweep.dot(np.ones(2 * num_nodes)))

        elif problem_type == "constrainedDAE":
            stabi_function = np.kron(np.zeros(num_nodes), Ieps)

    elif problem_name == "PROTHERO-ROBINSON":
        q = np.zeros(num_nodes)
        q[num_nodes - 1] = 1.0

        if problem_type == "SPP":
            stabi_function = 1 - dt / eps * weights.dot(Mat_sweep.dot(np.ones(num_nodes))) if do_coll_update else q.dot(Mat_sweep.dot(np.ones(num_nodes)))

        elif problem_type == "embeddedDAE":
            stabi_function = 1 - dt * weights.dot(Mat_sweep.dot(np.ones(num_nodes))) if do_coll_update else q.dot(Mat_sweep.dot(np.ones(num_nodes)))

        elif problem_type == "constrainedDAE":
            stabi_function = np.zeros(num_nodes)

        else:
            raise NotImplementedError()

    return stabi_function


def finalize_plot(dt, k, num_nodes, num_sweeps_list, problem_name, QI_list, stabi_plotter):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    dt : float
        Time step size.
    k : int
        Case number.
    num_nodes : int
        Number of collocation nodes.
    num_sweeps_list : list
        Different numbers of performed sweeps.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    stabi_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17}

    for q, QI in enumerate(QI_list):
        stabi_plotter.set_xlabel("number of sweeps", subplot_index=q)

        stabi_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

    stabi_plotter.set_ylabel("absolute value of stability function", subplot_index=None)

    # stabi_plotter.set_ylim((-0.03, 1.12), subplot_index=None)

    stabi_plotter.adjust_layout(num_subplots=len(QI_list))

    stabi_plotter.set_grid()

    bbox_pos = bbox_position[k]
    stabi_plotter.set_shared_legend(loc='lower center', bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"stability_function_{dt=}_case{k}_{num_nodes=}.png"
    stabi_plotter.save(filename)


if __name__ == "__main__":
    problem_name = "PROTHERO-ROBINSON"
    # problem_name = "LINEAR-TEST"

    QI_list = ["IE", "LU", "MIN-SR-S"]  # , 'MIN-SR-FLEX']
    num_nodes_list = np.arange(2, 21)#get_nodes_range_of_convergence(problem_name, node_every=1)
    num_sweeps_list = np.arange(1, 81)
    do_coll_update = True

    dt = 1e-1
    case = 1

    # problems = get_problem_cases(k=case, problem_name=problem_name)
    problems = {"constrainedDAE": [0.0], "embeddedDAE": [0.0]}
    # problems = {"SPP": [10 ** (-m) for m in range(1, 12)]}

    Q_coefficients = compute_Q_coefficients(num_nodes_list)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    for num_nodes in num_nodes_list:
        stabi_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

        for q, QI in enumerate(QI_list):
            for problem_type, eps_values in problems.items():
                for i, eps in enumerate(eps_values):
                    abs_stabi = []

                    for num_sweeps in num_sweeps_list:
                        Qmat = Q_coefficients[num_nodes]["matrix"]
                        QImat = QI_coefficients[QI][num_nodes]["matrix"]
                        weights = Q_coefficients[num_nodes]["weights"]

                        L, R = get_sweeper_mats(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type)

                        Mat_sweep = perform_sweeps(L, R, num_sweeps, problem_name, problem_type)

                        stabi_function = perform_collocation_update(do_coll_update, dt, eps, Mat_sweep, num_nodes, problem_name, problem_type, weights)

                        val = np.linalg.norm(stabi_function, np.inf) if np.size(stabi_function) > 1 else np.absolute(stabi_function)
                        abs_stabi.append(val)

                    color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                    problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)

                    marker, markersize = res["marker"], res["markersize"]
                    stabi_plotter = plot_result(
                        stabi_plotter,
                        num_sweeps_list,
                        abs_stabi,
                        q,
                        color,
                        marker,
                        markersize,
                        linestyle,
                        problem_label,
                        plot_type="plot",
                        markevery=4,
                    )

        finalize_plot(dt, case, num_nodes, num_sweeps_list, problem_name, QI_list, stabi_plotter)
