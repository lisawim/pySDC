import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes
from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.plotting.spectral_radius import (
    compute_Q_coefficients,
    compute_QI_coefficients,
    get_iteration_matrices_ordered_by_nodes,
)


def plot_svd(num_nodes, QI="MIN-SR-NS", journal="Springer_Scientific_Computing"):
    problem_name = "LINEAR-TEST"

    figsize = figsize_by_journal(journal, scale=0.55, ratio=0.7)

    sweeper_types = ["SPP", "embeddedDAE", "constrainedDAE"]

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt = dt_list[3]

    Q_coefficients = compute_Q_coefficients(num_nodes)
    QI_coefficients = compute_QI_coefficients(Q_coefficients, [QI])

    my_setup_mpl(fontsize=5)
    colors, markers, sweeper_labels = my_plot_style_config()
    colors_spp = [
        'mistyrose',
        'lightsalmon',
        'lightcoral',
        'indianred',
        'firebrick',
        'brown',
        'maroon',
        'lightgray',
        'darkgray',
        'gray',
        'dimgray',
    ]
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ind = 0
    for sweeper_type in sweeper_types:
        eps_list = [10 ** (-m) for m in range(1, 12)] if sweeper_type == "SPP" else [0.0]

        for e, eps in enumerate(eps_list):
            key = f"{sweeper_type}_IE"

            Qmat = Q_coefficients[num_nodes]["matrix"]
            QImat = QI_coefficients[QI][num_nodes]["matrix"]

            LHS, RHS = get_iteration_matrices_ordered_by_nodes(dt, num_nodes, Qmat, QImat, problem_name, sweeper_type, eps=eps)
            K = np.linalg.inv(LHS) @ RHS

            U, S, Vh = np.linalg.svd(K, full_matrices=True)
            print(f"For {sweeper_type} with {eps=}, the singular values are: {S}\n")
            x = [ind] * len(S)
            color = colors_spp[e] if sweeper_type == "SPP" else colors[key]
            marker = None if sweeper_type == "SPP" else markers[key]
            label = rf"$\varepsilon=${eps}" if sweeper_type == "SPP" else sweeper_labels[sweeper_type]
            ax.scatter(
                x, S, color=color, marker=marker, label=label, edgecolors="black", s=10, linewidth=0.2
            )

            ind += 1

    ax.tick_params(axis="both", which="minor", bottom=False, left=False)
    ax.set_ylabel(r"singular values")

    # ax.set_ylim((0.0, 0.3))
    # ax.set_ylim((0.0, 1.0))

    ax.set_yscale("log", base=10)
    # ax.set_ylim((1e-1, 3e0))
    ax.set_ylim((1e-2, 1.5e2))

    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.grid(linewidth=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=4)
    filename = "data" + "/" + f"{problem_name}" + "/" + "svd_thesis" + "/" + f"svd_{QI}_{num_nodes=}_{dt=}.png"
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_svd_of_iteration_matrix_powers(num_nodes, QI="MIN-SR-NS", journal="Springer_Scientific_Computing"):
    problem_name = "LINEAR-TEST"

    figsize = figsize_by_journal(journal, scale=0.8, ratio=0.55)

    sweeper_types = ["SPP", "embeddedDAE", "constrainedDAE"]

    dt_list, _ = choose_time_step_sizes(problem_name)
    dt = dt_list[3]

    Q_coefficients = compute_Q_coefficients(num_nodes)
    QI_coefficients = compute_QI_coefficients(Q_coefficients, [QI])

    my_setup_mpl(fontsize=6)
    colors, markers, sweeper_labels = my_plot_style_config()
    colors_spp = [
        'mistyrose',
        'lightsalmon',
        'lightcoral',
        'indianred',
        'firebrick',
        'brown',
        'maroon',
        'lightgray',
        'darkgray',
        'gray',
        'dimgray',
    ]
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ind, n = 0, 1
    ind_list = []
    for sweeper_type in sweeper_types:
        eps_list = [10 ** (-m) for m in range(1, 12)] if sweeper_type == "SPP" else [0.0]

        for e, eps in enumerate(eps_list):
            key = f"{sweeper_type}_IE"

            Qmat = Q_coefficients[num_nodes]["matrix"]
            QImat = QI_coefficients[QI][num_nodes]["matrix"]

            LHS, RHS = get_iteration_matrices_ordered_by_nodes(dt, num_nodes, Qmat, QImat, problem_name, sweeper_type, eps=eps)
            K = np.linalg.inv(LHS) @ RHS

            for k in range(1, num_nodes + 1):
                K_power = np.linalg.matrix_power(K, k)
                U, S, Vh = np.linalg.svd(K_power, full_matrices=True)
                print(f"For {sweeper_type} with {eps=}, the singular values to power {k} are: {S}\n")

                x = [ind] * len(S)
                ind_list.append(ind)
                color = colors_spp[e] if sweeper_type == "SPP" else colors[key]
                marker = None if sweeper_type == "SPP" else markers[key]
                label = rf"$\varepsilon=${eps}" if sweeper_type == "SPP" else sweeper_labels[sweeper_type]
                ax.scatter(
                    x, S, color=color, marker=marker, label=label if k == 1 else None, edgecolors="black", s=7, linewidth=0.2
                )

                ind += 35
            print()
            ind += 80
            n += 1

    ax.tick_params(axis="both", which="minor", bottom=True, left=False)

    ax.grid(which="major", axis="x", linewidth=0.45, alpha=0.3)
    
    ax.set_xlabel(r"power k")
    ax.set_ylabel(r"singular values of $K^k$")

    # ax.set_ylim((0.0, 0.3))
    # ax.set_ylim((0.0, 1.0))

    ax.set_yscale("log", base=10)
    # ax.set_ylim((1e-1, 3e0))
    ax.set_xlim((ind_list[0] - 20, ind_list[-1] + 20))
    ax.set_ylim((1e-2, 1.5e3))

    ax.set_xticks(ind_list)
    ax.set_xticklabels((n-1) * [k for k in range(1, num_nodes + 1)])

    ax.grid(linewidth=0.5)
    ax.set_axisbelow(True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5)
    filename = "data" + "/" + f"{problem_name}" + "/" + "svd_thesis" + "/" + f"svd_powers_{QI}_{num_nodes=}_{dt=}.png"
    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)



if __name__ == "__main__":
    num_nodes = 8
    QI = "MIN-SR-S"

    plot_svd(num_nodes=num_nodes, QI=QI, journal="BUW_thesis")
    plot_svd_of_iteration_matrix_powers(num_nodes=num_nodes, QI=QI, journal="BUW_thesis")
