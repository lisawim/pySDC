import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE.plotting.spectral_radius import (
    compute_Q_coefficients,
    compute_QI_coefficients,
    get_iteration_matrices_ordered_by_nodes,
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
        x /= np.linalg.norm(x, ord=2)
        fov_values[i] = np.vdot(x, A @ x)

    return fov_values


import numpy as np

def random_unit_vector(n):
    """Generate a random unit vector of dimension n with real entries."""
    v = np.random.randn(n)  # Generate n Gaussian-distributed random numbers
    v /= np.linalg.norm(v)  # Normalize to unit length
    return v


def plot_field_of_values(
    dt,
    eps=0.0,
    num_nodes=4,
    QI="MIN-SR-NS",
    sweeper_type="constrainedDAE",
    journal="BUW_thesis",
    ax=None,
    return_ax=False,
):
    created_fig = ax is None

    problem_name = "LINEAR-TEST"

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt = dt_list[3]
    num_nodes = 4

    Q_coefficients = compute_Q_coefficients([num_nodes])

    QI_coefficients = compute_QI_coefficients(Q_coefficients, [QI])

    a = np.cos(np.linspace(0, 2 * np.pi, 200))
    b = np.sin(np.linspace(0, 2 * np.pi, 200))

    if created_fig:
        figsize = figsize_by_journal(journal, scale=0.7, ratio=0.83)
        my_setup_mpl(fontsize=7)
        colors, markers, sweeper_labels = my_plot_style_config()
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(a, b, color="black", linewidth=0.6)

    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.axvline(0.0, color="black", linewidth=0.6)

    Qmat = Q_coefficients[num_nodes]["matrix"]
    QImat = QI_coefficients[QI][num_nodes]["matrix"]

    LHS, RHS = get_iteration_matrices_ordered_by_nodes(
        dt, num_nodes, Qmat, QImat, problem_name, sweeper_type, eps=eps
    )
    K = np.linalg.inv(LHS) @ RHS

    fov = field_of_values(K)
    lambdas = np.linalg.eigvals(K)

    for k in range(1, num_nodes + 1):
        K_power = np.linalg.matrix_power(K, k)
        spectral_radius = max(abs(np.linalg.eigvals(K_power)))
        print(f"For {sweeper_type} with {eps=}, spectral radius with power {k}: {spectral_radius}")
        U, S, Vh = np.linalg.svd(K_power, full_matrices=True)
        print(f"Singular values are: {S}\n")
    print()
    ax.scatter(fov.real, fov.imag, s=1, alpha=0.5, label=f"field of values")
    ax.scatter(lambdas.real, lambdas.imag, marker="^", label=f"eigenvalue distribution")

    ax.set_xlim((-6.0, 6.0))
    ax.set_ylim((-6.0, 6.0))

    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")

    if created_fig:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

        label = rf"SDC-SPP with $\varepsilon=${eps}" if sweeper_type == "SPP" else sweeper_labels[sweeper_type]
        ax.set_title(rf"EV and FoV for {label} with {num_nodes} nodes")

        eps = f"{eps:.1e}" if sweeper_type == "SPP" else ""
        filename = "data" + "/" + f"{problem_name}" + "/" + f"fov_and_ev_{sweeper_type}_{eps}_{num_nodes=}_{dt=}.png"
        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)

        if not return_ax:
            plt.close(fig)
            return None

    return ax if return_ax else None


def plot_field_of_values_thesis(
    QI="MIN-SR-NS",
    sweeper_types=["SPP", "embeddedDAE", "constrainedDAE"],
    eps_list=[0.1, 1e-5],
    journal="BUW_thesis",
):
    problem_name = "LINEAR-TEST"

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt = dt_list[3]
    num_nodes = 4

    figsize = figsize_by_journal(journal, scale=0.6, ratio=1.1)
    my_setup_mpl(fontsize=6)
    _, _, sweeper_labels = my_plot_style_config()
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    ind = 0
    for sweeper_type in sweeper_types:
        eps_list = eps_list if sweeper_type == "SPP" else [0.0]

        for e, eps in enumerate(eps_list):
            label = rf"SDC-SPP with $\varepsilon=${eps}" if sweeper_type == "SPP" else sweeper_labels[sweeper_type]
            ax_flatten[ind].set_title(label)

            plot_field_of_values(
                dt=dt,
                eps=eps,
                num_nodes=num_nodes,
                QI=QI,
                sweeper_type=sweeper_type,
                journal=journal,
                ax=ax_flatten[ind],
                return_ax=False,
            )

            ind += 1

    for ax_obj in ax_flatten:
        ax_obj.set_xlim((-6.0, 3.0))
        ax_obj.set_ylim((-5.0, 5.0))

    handles, labels = ax_flatten[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"fov_and_ev_{num_nodes=}_{dt=}.png"
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_field_of_values_all_sweepers(journal="BUW_thesis"):
    problem_name = "LINEAR-TEST"

    sweeper_types=["SPP", "embeddedDAE", "constrainedDAE"]
    QI="MIN-SR-NS"

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt = dt_list[3]
    num_nodes = 4

    for sweeper_type in sweeper_types:
        eps_list = [10 ** (-m) for m in range(1, 12)] if sweeper_type == "SPP" else [0.0]

        for e, eps in enumerate(eps_list):
            plot_field_of_values(
                dt=dt,
                eps=eps,
                num_nodes=num_nodes,
                QI=QI,
                sweeper_type=sweeper_type,
                journal=journal,
            )



if __name__ == "__main__":
    # plot_field_of_values_thesis()
    plot_field_of_values_all_sweepers()
