import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.DAE.plotting.linearTest_spectral_radius import (
    compute_Q_coefficients, compute_QI_coefficients
)


def field_of_values(A, num_points=3000):
    """
    Approximiert das Field of Values (numerical range) einer Matrix A.
    
    Parameters
    ----------
    A : (n, n) complex or real ndarray
        Die Eingabematrix.
    num_points : int
        Anzahl der Richtungsvektoren auf dem Einheitskreis.

    Returns
    -------
    w : complex ndarray
        Approximierte Werte x*Ax für ||x||=1.
    """
    # n = A.shape[0]
    # fov_values = np.zeros(num_points, dtype=complex)
    
    # for i, theta in enumerate(np.linspace(0, 2 * np.pi, num_points, endpoint=False)):
    #     x = np.exp(1j * theta) * np.ones(n) / np.sqrt(n)  # Einheitlicher Richtungsvektor
    #     x = x / np.linalg.norm(x)
    #     fov_values[i] = np.vdot(x, A @ x)

    n = A.shape[0]
    fov_values = np.zeros(num_points, dtype=complex)
    for i in range(num_points):
        x = np.random.randn(n) + 1j * np.random.randn(n)
        x /= np.linalg.norm(x)
        fov_values[i] = np.vdot(x, A @ x)

    return fov_values


import numpy as np

def random_unit_vector(n):
    """Generate a random unit vector of dimension n with real entries."""
    v = np.random.randn(n)  # Generate n Gaussian-distributed random numbers
    v /= np.linalg.norm(v)  # Normalize to unit length
    return v


if __name__ == "__main__":
    num_nodes_list = range(2, 31)
    QI_list = ["MIN-SR-NS"]#["IE", "LU", "MIN-SR-S", "MIN-SR-NS"]
    marker = ["^", "s", "o", "D"]

    dt = 1e-1

    lamb_diff = -2.0
    lamb_alg = 1.0

    A = np.array([
        [lamb_diff, lamb_alg],
        [lamb_diff, -lamb_alg]
    ])

    Ieps = np.identity(2)
    Ieps[-1, -1] = 0.0

    Q_coefficients = compute_Q_coefficients(num_nodes_list)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    num = 1000#3000

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # for QI in QI_list:
    #     print(f"\n{QI}:\n")
    #     fov_std = []

    #     for num_nodes in num_nodes_list:
    #         list_of_unit_vectors = [random_unit_vector(2 * num_nodes) for _ in range(num)]

    #         Qmat = Q_coefficients[num_nodes]["matrix"]
    #         QImat = QI_coefficients[QI][num_nodes]["matrix"]

    #         inv = np.linalg.inv(np.kron(np.identity(num_nodes), Ieps) - dt * np.kron(QImat, A))
    #         K_stiff = np.matmul(inv, dt * np.kron(Qmat - QImat, A))

    #         fov = [x.T.dot(K_stiff.dot(x)) for x in list_of_unit_vectors]
    #         fov_std.append(np.std(fov))

    #         print(f"Standardabweichung: {np.std(fov)}")
    #         print()

    #     ax.plot(num_nodes_list, fov_std, marker="o", label=f"{QI}")

    # ax.set_xlabel("number of nodes")
    # ax.set_ylabel("standard deviation of field of values")
    # ax.set_xticks(num_nodes_list[::4])
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)

    # plt.tight_layout()

    # filename = "data" + "/" + "LINEAR-TEST" + "/" + f"standard_deviation_fov_fullyImplicitDAE.png"
    # fig.savefig(filename, dpi=400, bbox_inches="tight")

    a = np.cos(np.linspace(0, 2 * np.pi, 200))
    b = np.sin(np.linspace(0, 2 * np.pi, 200))

    for num_nodes in num_nodes_list:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for q, QI in enumerate(QI_list):
            Qmat = Q_coefficients[num_nodes]["matrix"]
            QImat = QI_coefficients[QI][num_nodes]["matrix"]

            inv = np.linalg.inv(np.kron(np.identity(num_nodes), Ieps) - dt * np.kron(QImat, A))
            K_stiff = np.matmul(inv, dt * np.kron(Qmat - QImat, A))

            fov = field_of_values(K_stiff, num_points=num)
            print(fov)
            lambdas = np.linalg.eigvals(K_stiff)

            for k in range(1, num_nodes + 1):
                spectral_radius = max(abs(np.linalg.eigvals(np.linalg.matrix_power(K_stiff, k))))
                print(f"Spectral radius with power {k}: {spectral_radius}")
            print()

            ax.scatter(fov.real, fov.imag, s=1, alpha=0.5, label=f"FoV - FI-SDC-{QI}")
            ax.scatter(lambdas.real, lambdas.imag, marker=marker[q], label=f"FI-SDC-{QI}")

        ax.plot(a, b, color="black")

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_aspect("equal")

        ax.set_xlim((-30.0, 2.0))
        ax.set_ylim((-5.0, 5.0))

        ax.set_xlabel("Real part")
        ax.set_ylabel("Imaginary part")

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4)

        plt.title(f"Eigenvalue distribution and field of values for {num_nodes} nodes")

        filename = "data" + "/" + f"LINEAR-TEST" + "/" + f"eigvals_distribution_{num_nodes=}.png"
        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)
