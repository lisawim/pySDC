import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.DAE.plotting.linearTest_spectral_radius import (
    compute_Q_coefficients,
    compute_QI_coefficients,
    get_nodes_range_of_convergence,
    get_iteration_matrices,
)

def u_exact(t, lamb_diff=-2.0, lamb_alg=1.0):
        r"""
        Routine for the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        u_ex : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Exact solution.
        """

        return np.array(
            [
                np.exp(2 * lamb_diff * t),
                lamb_diff / lamb_alg * np.exp(2 * lamb_diff * t)
            ]
        )

def du_exact(t, lamb_diff=-2.0, lamb_alg=1.0):
        r"""
        Routine for the exact solution at time :math:`t`.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        u_ex : pySDC.projects.DAE.misc.meshDAE.MeshDAE
            Exact solution.
        """

        return np.array(
            [
                2 * lamb_diff * np.exp(2 * lamb_diff * t),
                (2 * lamb_diff ** 2) / lamb_alg * np.exp(2 * lamb_diff * t)
            ]
        )

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

        I_M = np.identity(num_nodes)
        Ieps = np.identity(2)
        Ieps[-1, -1] = eps

        if problem_name == "LINEAR-TEST":
            lamb_diff = -2.0
            lamb_alg = 1.0

            A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

            if problem_type in ["SPP", "embeddedDAE", "fullyImplicitDAE"]:
                L = np.kron(I_M, Ieps) - dt * np.kron(QI, A)
                R = dt * np.kron(Q - QI, A)

            elif problem_type == "constrainedDAE":
                lamb_diff = -2.0
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

        if problem_name == "LINEAR-TEST":
            lamb_diff = -2.0
            lamb_alg = 1.0

            A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

            if problem_type in ["SPP", "embeddedDAE"]:
                I_M = np.identity(num_nodes)
                Ieps = np.identity(2)
                Ieps[-1, -1] = eps
                I_Meps = np.kron(I_M, Ieps)

                Mat_sweep = np.linalg.matrix_power(L_inv.dot(R), num_sweeps).dot(I_Meps)
                for k in range(0, num_sweeps):
                    Mat_sweep += (np.linalg.matrix_power(L_inv.dot(R), k).dot(L_inv)).dot(I_Meps)

            elif problem_type == "fullyImplicitDAE":
                I_M = np.identity(num_nodes)
                I_MA = np.kron(I_M, A)

                Mat_sweep = np.linalg.matrix_power(L_inv.dot(R), num_sweeps).dot(I_MA)
                for k in range(0, num_sweeps):
                    Mat_sweep += (np.linalg.matrix_power(L_inv.dot(R), k).dot(L_inv)).dot(I_MA)             

        return Mat_sweep


if __name__ == "__main__":
    problem_name = "LINEAR-TEST"

    lamb_diff = -2.0
    lamb_alg = 1.0

    A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

    QI_list = ["MIN-SR-NS"]
    num_nodes = 30
    num_sweeps_list = range(1, 120 + 1)

    Q_coefficients = compute_Q_coefficients(num_nodes)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list, None)

    t0 = 0.0
    dt = 1e-2

    u1_ex = u_exact(t0 + dt)
    du1_ex = du_exact(t0 + dt)

    problems = {"embeddedDAE": [0.0], "fullyImplicitDAE": [0.0]}

    u0_full = np.array([u_exact(t0) for m in range(num_nodes)]).flatten()
    du0_full = np.zeros_like(u0_full)  # provisional solution for du is zero

    fig, axs = plt.subplots(2, 1, figsize=(6, 12))
    for problem_type, eps_values in problems.items():
        for i, eps in enumerate(eps_values):
            for q, QI in enumerate(QI_list):
                print(f"\n{QI}: Run experiment for {problem_type} for {num_nodes} nodes using {dt=}...\n")

                Qmat = Q_coefficients[num_nodes]["matrix"]
                QImat = QI_coefficients[QI][num_nodes]["matrix"]
                weights = Q_coefficients[num_nodes]["weights"]

                L, R = get_sweeper_mats(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type)

                errors, errors_diff, errors_alg = [], [], []
                for num_sweeps in num_sweeps_list:
                    L_inv = np.linalg.inv(L)
                    I_M = np.identity(num_nodes)

                    if problem_type in ["SPP", "embeddedDAE"]:
                        Ieps = np.identity(2)
                        Ieps[-1, -1] = eps
                        I_Meps = np.kron(I_M, Ieps)

                        Mat_sweep = np.linalg.matrix_power(L_inv.dot(R), num_sweeps).dot(I_Meps).dot(u0_full)
                        for k in range(0, num_sweeps):
                            Mat_sweep += (np.linalg.matrix_power(L_inv.dot(R), k).dot(L_inv)).dot(I_Meps).dot(u0_full)

                    elif problem_type == "fullyImplicitDAE":
                        I_MA = np.kron(I_M, A)

                        Mat_sweep = np.linalg.matrix_power(L_inv.dot(R), num_sweeps).dot(du0_full)
                        for k in range(0, num_sweeps):
                            Mat_sweep += (np.linalg.matrix_power(L_inv.dot(R), k).dot(L_inv)).dot(I_MA.dot(u0_full))
                    # Mat_sweep = perform_sweeps(eps, num_nodes, L, R, num_sweeps, problem_name, problem_type)

                    u1 = Mat_sweep

                    if problem_type == "embeddedDAE":
                        err = max(abs(u1_ex - u1[-2 :]))
                        err_diff = abs(u1_ex[0] - u1[-2])
                        err_alg = abs(u1_ex[1] - u1[-1])
                    elif problem_type == "fullyImplicitDAE":
                        err = max(abs(du1_ex - u1[-2 :]))
                        err_diff = abs(du1_ex[0] - u1[-2])
                        err_alg = abs(du1_ex[1] - u1[-1])
                    else:
                        raise NotImplementedError
                    
                    errors.append(err)
                    errors_diff.append(err_diff)
                    errors_alg.append(err_alg)
                    # print(f"Error after {num_sweeps} sweeps: {err}")
                    # print(f"Diff. error after {num_sweeps} sweeps: {err_diff}")
                    # print(f"Alg. error after {num_sweeps} sweeps: {err_alg} \n")
                    # print(errors_diff)
                    # print(errors_alg)
                    # print()

                axs[0].semilogy(num_sweeps_list, errors_diff, label=f"{problem_type}-{QI}")
                axs[1].semilogy(num_sweeps_list, errors_alg)

    for ax in axs:
        ax.set_ylim((1e-15, 1e3))

        ax.set_xlabel("iteration")

        ax.set_yscale("log", base=10)

    axs[0].set_ylabel("Diff. error")
    axs[1].set_ylabel("Alg. error")

    axs[0].legend(loc="upper right")

    filename = "data" + "/" + f"LINEAR-TEST" + "/" + f"error_propagation_{dt=}_{num_nodes=}.png"
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)
