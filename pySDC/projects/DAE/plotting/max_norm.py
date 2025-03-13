import numpy as np

from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.projects.DAE.plotting.linearTest_spectral_radius import (
    compute_Q_coefficients,
    compute_QI_coefficients,
    get_iteration_matrices,
    get_nodes_range_of_convergence,
)
from pySDC.projects.DAE import getColor, getLabel, get_linestyle, getMarker, Plotter


def finalize_plot(dt, k, k_flex, num_nodes_list, problem_name, QI_list, norm_plotter):
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
    norm_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17}

    for q, QI in enumerate(QI_list):
        norm_plotter.set_xticks(num_nodes_list[0::2], subplot_index=q)
        norm_plotter.set_xlabel("number of nodes", subplot_index=q)

        norm_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

    norm_plotter.set_ylabel("max norm", subplot_index=None)

    norm_plotter.set_ylim((-0.2, 5.0), subplot_index=None)

    norm_plotter.adjust_layout(num_subplots=len(QI_list))

    norm_plotter.set_grid()

    bbox_pos = bbox_position[k]
    norm_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"max_norm_{dt=}_case{k}_{k_flex=}.png"
    norm_plotter.save(filename)


if __name__ == "__main__":
    problem_name = "PROTHERO-ROBINSON"

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes_list = get_nodes_range_of_convergence(problem_name, node_every=1)

    k_flex = 1

    dt = 1e-2
    case = 6

    problems = get_problem_cases(k=case, problem_name=problem_name)

    Q_coefficients = compute_Q_coefficients(num_nodes_list)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list, k_flex)

    norm_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                max_norms = []

                for num_nodes in num_nodes_list:

                    Qmat = Q_coefficients[num_nodes]["matrix"]
                    QImat = QI_coefficients[QI][num_nodes]["matrix"]

                    K = get_iteration_matrices(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type)

                    max_norm = np.linalg.norm(K, np.inf)
                    max_norms.append(max_norm)

                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)

                marker, markersize = res["marker"], res["markersize"]
                norm_plotter = plot_result(
                    norm_plotter,
                    num_nodes_list,
                    max_norms,
                    q,
                    color,
                    marker,
                    markersize,
                    linestyle,
                    problem_label,
                    plot_type="plot",
                )

    finalize_plot(dt, case, k_flex, num_nodes_list, problem_name, QI_list, norm_plotter)