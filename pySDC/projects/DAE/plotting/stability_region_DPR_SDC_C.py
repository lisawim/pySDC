import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.projects.DAE.plotting.linearTest_spectral_radius import (
    compute_Q_coefficients, compute_QI_coefficients, get_nodes_range_of_convergence
)
from pySDC.projects.DAE import getColor, getLabel, get_linestyle, getMarker, Plotter


def get_sweeper_mats(dt, eps, num_nodes, Q, QI, problem_name, problem_type, **kwargs):
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

        I_M = np.identity(num_nodes)

        lamb = kwargs.get("lamb")

        if problem_type == "constrainedDAE":
            if problem_name == "DPR":
                L = I_M - lamb * dt * QI
                R = lamb * dt * (Q - QI)

            elif problem_name == "LINEAR-TEST":
                lamb_diff = lamb  # -2.0
                L = I_M - 2 * lamb_diff * dt * QI
                R = 2 * lamb_diff * dt * (Q - QI)

        return L, R


def perform_sweeps(eps, num_nodes, L, R, num_sweeps, problem_name, problem_type):
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

    lamb = kwargs.get("lamb")

    q = np.zeros(num_nodes)
    q[num_nodes - 1] = 1.0

    if problem_name == "DPR":
        if do_coll_update:
            stabi_function = (1 + lamb * weights.dot(Mat_sweep.dot(np.ones(num_nodes))))

        else:
            stabi_function = q.dot(Mat_sweep.dot(np.ones(num_nodes)))

    elif problem_name == "LINEAR-TEST":
        if do_coll_update:
            lamb_diff = lamb  # -2.0
            stabi_function = (1 + 2 * lamb_diff * weights.dot(Mat_sweep.dot(np.ones(num_nodes))))

        else:
            stabi_function = q.dot(Mat_sweep.dot(np.ones(num_nodes)))

    return stabi_function


def finalize_plot(num_nodes, num_sweeps, problem_name, problem_type, QI_list, stabi_region_plotter):
    for q, QI in enumerate(QI_list):
        stabi_region_plotter.set_xlabel(r"Real part $\lambda$", subplot_index=q)
        stabi_region_plotter.set_ylabel(r"Imaginary part $\lambda$", subplot_index=q)

        stabi_region_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

    stabi_region_plotter.set_grid(linestyle=':')

    stabi_region_plotter.set_aspect()

    stabi_region_plotter.adjust_layout(num_subplots=len(QI_list))

    filename = "data" + "/" + f"{problem_name}" + "/" + f"stabi_region_{problem_type}_{num_nodes=}_{num_sweeps=}.png"
    stabi_region_plotter.save(filename)


if __name__ == "__main__":
    # problem_name = "DPR"
    problem_name = "LINEAR-TEST"

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = 4
    num_sweeps = 4
    do_coll_update = True

    dt = 1e0
    case = 1

    # problems = get_problem_cases(k=case, problem_name=problem_name)
    problems = {"constrainedDAE": [0.0]}
    # problems = {"SPP": [10 ** (-m) for m in range(1, 12)]}

    Q_coefficients = compute_Q_coefficients(num_nodes)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    resolution = 400
    x = np.linspace(-20.0, 15.0, num=resolution)
    y = np.linspace(-20.0, 20.0, num=resolution)

    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y  # Komplexes Lambda

    stabi_region_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for problem_type, eps_values in problems.items():
        for i, eps in enumerate(eps_values):
            for q, QI in enumerate(QI_list):
                print(f"\n{QI}: Compute stability function for {problem_type} with {eps=} using {dt=}...\n")

                Qmat = Q_coefficients[num_nodes]["matrix"]
                QImat = QI_coefficients[QI][num_nodes]["matrix"]
                weights = Q_coefficients[num_nodes]["weights"]

                rho = np.zeros_like(X)

                for i in range(resolution):
                    for j in range(resolution):
                        lamb = Z[i, j]

                        kwargs = {"lamb": lamb}

                        L, R = get_sweeper_mats(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type, **kwargs)

                        Mat_sweep = perform_sweeps(eps, num_nodes, L, R, num_sweeps, problem_name, problem_type)

                        stabi_function = perform_collocation_update(do_coll_update, dt, eps, Mat_sweep, num_nodes, problem_name, problem_type, weights)

                        rho[i, j] = np.max(np.abs(np.linalg.eigvals(stabi_function))) if np.size(stabi_function) > 1 else np.abs(stabi_function)

                stabi_region_plotter.add_hline(0.0, subplot_index=q, color="black", lw=0.5)
                stabi_region_plotter.add_vline(0.0, subplot_index=q, color="black", lw=0.5)

                stabi_region_plotter.contourf(X, Y, rho, subplot_index=q, levels=[0, 1], colors="gray")
                stabi_region_plotter.contour(X, Y, rho, subplot_index=q, levels=[1], colors="black", linewidths=1)

                # plt.figure(figsize=(6, 6))
                # plt.contourf(X, Y, rho, levels=[0, 1], colors=["#add8e6"])  # hellblau für stabil
                # plt.contour(X, Y, rho, levels=[1], colors="black")

                # # Achsen und Gitter
                # plt.axhline(0, color='gray', lw=0.5)
                # plt.axvline(0, color='gray', lw=0.5)
                # plt.xlabel(r'$\mathrm{Re}(\lambda)$')
                # plt.ylabel(r'$\mathrm{Im}(\lambda)$')
                # plt.title(f'Stabilitätsgebiet für SDC-C mit {QI}')
                # plt.grid(True, linestyle=':')
                # plt.gca().set_aspect('equal')

    finalize_plot(num_nodes, num_sweeps, problem_name, problem_type, QI_list, stabi_region_plotter)
