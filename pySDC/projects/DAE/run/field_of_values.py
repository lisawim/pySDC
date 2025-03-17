import numpy as np

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
    num_nodes = 17
    QI_list = ["IE", "LU", "MIN-SR-S"]

    dt = 0.1

    Q_coefficients = compute_Q_coefficients(num_nodes)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    num = 3000

    for QI in QI_list:
        print(f"\n{QI}:\n")

        sum = 0
        i=0
        for _ in range(num):
            Qmat = Q_coefficients[num_nodes]["matrix"]
            QImat = QI_coefficients[QI][num_nodes]["matrix"]

            K_stiff = np.identity(num_nodes) - np.linalg.inv(QImat).dot(Qmat)

            x = random_unit_vector(num_nodes)
            val = x.T.dot(K_stiff.dot(x))
            # print(val)

            sum += abs(val)

            i+= 1

        print("Count:", i)

        mean = sum / num
        print(f"Mean for {QI}: {mean}")
        print()
