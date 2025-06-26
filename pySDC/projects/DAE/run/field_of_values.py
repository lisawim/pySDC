import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.DAE.plotting.linearTest_spectral_radius import (
    compute_Q_coefficients, compute_QI_coefficients
)


import numpy as np

def random_unit_vector(n):
    """Generate a random unit vector of dimension n with real entries."""
    v = np.random.randn(n)  # Generate n Gaussian-distributed random numbers
    v /= np.linalg.norm(v)  # Normalize to unit length
    return v


if __name__ == "__main__":
    num_nodes_list = range(2, 31)
    QI_list = ["IE", "LU", "MIN-SR-S", "MIN-SR-NS"]

    dt = 1e-2

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

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for QI in QI_list:
        print(f"\n{QI}:\n")
        fov_std = []

        for num_nodes in num_nodes_list:
            list_of_unit_vectors = [random_unit_vector(2 * num_nodes) for _ in range(num)]

            Qmat = Q_coefficients[num_nodes]["matrix"]
            QImat = QI_coefficients[QI][num_nodes]["matrix"]

            inv = np.linalg.inv(np.kron(np.identity(num_nodes), Ieps) - dt * np.kron(QImat, A))
            K_stiff = np.matmul(inv, dt * np.kron(Qmat - QImat, A))

            fov = [x.T.dot(K_stiff.dot(x)) for x in list_of_unit_vectors]
            fov_std.append(np.std(fov))

            print(f"Standardabweichung: {np.std(fov)}")
            print()

        ax.plot(num_nodes_list, fov_std, marker="o", label=f"{QI}")

    ax.set_xlabel("number of nodes")
    ax.set_ylabel("standard deviation of field of values")
    ax.set_xticks(num_nodes_list[::4])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)

    plt.tight_layout()

    filename = "data" + "/" + "LINEAR-TEST" + "/" + f"standard_deviation_fov_fullyImplicitDAE.png"
    fig.savefig(filename, dpi=400, bbox_inches="tight")
