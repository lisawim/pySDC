import numpy as np

from projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.projects.DAE.plotting.linearTest_spectral_radius import compute_Q_coefficients, compute_QI_coefficients, get_iteration_matrices
from pySDC.projects.DAE import computeNormalityDeviation, getColor, getLabel, get_linestyle, getMarker, Plotter


def finalize_plot(k, k_flex, num_nodes, problem_name, QI_list, spectral_radii_plotter):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    k : int
        Case number.
    num_nodes : int
        Number of collocation nodes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    spectral_radii_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    """

    bbox_position = {1: -0.17, 2: -0.17, 3: -0.25, 4: -0.05, 5: -0.05, 6: -0.17}

    for q, QI in enumerate(QI_list):
        spectral_radii_plotter.set_xlabel('time step sizes', subplot_index=q)

        spectral_radii_plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=20)

    spectral_radii_plotter.set_ylabel('spectral radius', subplot_index=None)

    spectral_radii_plotter.set_ylim((-0.03, 0.85), subplot_index=None)
    spectral_radii_plotter.set_ylim((-0.03, 10.0), subplot_index=3)

    spectral_radii_plotter.adjust_layout(num_subplots=len(QI_list))

    spectral_radii_plotter.set_grid()

    bbox_pos = bbox_position[k]
    spectral_radii_plotter.set_shared_legend(loc='lower center', bbox_to_anchor=(0.5, bbox_pos), ncol=4, fontsize=22)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"spectral_radius_stiff_limit_{num_nodes=}_case{k}_{k_flex=}.png"
    spectral_radii_plotter.save(filename)


if __name__ == "__main__":
    problem_name = 'LINEAR-TEST'

    QI_list = ['IE', 'LU', 'MIN-SR-S']#, 'MIN-SR-FLEX']
    num_nodes = 6

    k_flex = 1

    dt_list = np.logspace(-2.5, 0.0, num=11)
    case = 6

    problems = get_problem_cases(k=case)

    Q_coefficients = compute_Q_coefficients(num_nodes)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list, k_flex)

    spectral_radii_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                spectral_radii = []

                for dt in dt_list:

                    Qmat = Q_coefficients[num_nodes]['matrix']
                    QImat = QI_coefficients[QI][num_nodes]['matrix']

                    K = get_iteration_matrices(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type)

                    spectral_radius = max(abs(np.linalg.eigvals(K)))
                    spectral_radii.append(spectral_radius)

                    derivation = computeNormalityDeviation(K)
                    print(f"{QI}-{problem_type} for {num_nodes} nodes: Derivation from normality is: {derivation}\n")

                color, res, problem_label, linestyle = getColor(problem_type, i), getMarker(problem_type, i), getLabel(problem_type, eps), get_linestyle(problem_type)

                marker, markersize = res["marker"], res["markersize"]
                plot_result(
                    spectral_radii_plotter,
                    dt_list,
                    spectral_radii,
                    q,
                    color,
                    marker,
                    markersize,
                    linestyle,
                    problem_label,
                    plot_type="semilogx",
                )

    finalize_plot(case, k_flex, num_nodes, problem_name, QI_list, spectral_radii_plotter)