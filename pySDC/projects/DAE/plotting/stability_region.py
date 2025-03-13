import numpy as np

from pySDC.projects.DAE.run.error import plot_result
from pySDC.projects.DAE.plotting.stability_function_vs_iteration import (
    get_sweeper_mats, perform_sweeps, perform_collocation_update
)
from pySDC.projects.DAE.plotting.linearTest_spectral_radius import (
    compute_Q_coefficients, compute_QI_coefficients
)
from pySDC.projects.DAE import getLabel, Plotter


def plot_stability_region_DAE(dt_list, num_nodes, num_sweeps, problem_name, problems, Q_coefficients, QI_coefficients):
    for problem_type, eps_values in problems.items():
        for i, eps in enumerate(eps_values):
            region_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

            for q, QI in enumerate(QI_list):
                stabi_region = np.zeros(len(dt_list))

                for d, dt in enumerate(dt_list):
                    print(f"\n{QI}: Running test for {problem_type} with {eps=} using {dt=}...\n")
                    Qmat = Q_coefficients[num_nodes]["matrix"]
                    QImat = QI_coefficients[QI][num_nodes]["matrix"]
                    weights = Q_coefficients[num_nodes]["weights"]

                    L, R = get_sweeper_mats(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type)

                    Mat_sweep = perform_sweeps(L, R, num_sweeps)

                    stabi_function = perform_collocation_update(do_coll_update, dt, eps, Mat_sweep, num_nodes, problem_name, problem_type, weights)

                    val = np.linalg.norm(stabi_function, np.inf) if np.size(stabi_function) > 1 else np.absolute(stabi_function)
                    stabi_region[d] = val

                # Define the stability threshold |uend| <= 1
                stability_mask = stabi_region <= 1

                region_plotter.fill_between(dt_list, stabi_region, 1, subplot_index=q, where=stability_mask, color="white")

                region_plotter.fill_between(dt_list, stabi_region, 1, subplot_index=q, where=~stability_mask, color="gray", alpha=0.6)

                problem_label = getLabel(problem_type, eps, QI)
                label = problem_label
                region_plotter = plot_result(region_plotter, dt_list, stabi_region, q, "black", None, None, "dotted", label, plot_type="semilogx")

    finalize_plot_DAE(region_plotter, num_nodes, problem_name, QI_list)


def plot_stability_region_SPP(dt_list, eps_list, num_nodes, num_sweeps, problem_name, problems, Q_coefficients, QI_coefficients):
    EPS, DT = np.meshgrid(eps_list, dt_list)

    for problem_type, eps_values in problems.items():
        region_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

        for q, QI in enumerate(QI_list):
            stabi_region = np.zeros((len(eps_list), len(dt_list)))

            for i, eps in enumerate(eps_values):
                for d, dt in enumerate(dt_list):
                    print(f"\n{QI}: Running test for {problem_type} with {eps=} using {dt=}...\n")
                    Qmat = Q_coefficients[num_nodes]["matrix"]
                    QImat = QI_coefficients[QI][num_nodes]["matrix"]
                    weights = Q_coefficients[num_nodes]["weights"]

                    L, R = get_sweeper_mats(dt, eps, num_nodes, Qmat, QImat, problem_name, problem_type)

                    Mat_sweep = perform_sweeps(L, R, num_sweeps)

                    stabi_function = perform_collocation_update(do_coll_update, dt, eps, Mat_sweep, num_nodes, problem_name, problem_type, weights)

                    val = np.linalg.norm(stabi_function, np.inf) if np.size(stabi_function) > 1 else np.absolute(stabi_function)
                    stabi_region[i, d] = val
            print(stabi_region)
            # Define the stability threshold |uend| <= 1
            stability_mask = stabi_region <= 1

            region_plotter.contour(EPS, DT, stabi_region, subplot_index=q, levels=[1], colors="black", linewidths=1)

            region_plotter.contourf(EPS, DT, stability_mask, subplot_index=q, levels=[1, np.inf], colors="gray", alpha=0.6)

    finalize_plot_SPP(region_plotter, num_nodes, problem_name, QI_list)


def finalize_plot_SPP(plotter, num_nodes, problem_name, QI_list):
    for q, QI in enumerate(QI_list):
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=24)

        plotter.set_ylabel(f"Time step size", subplot_index=q)
        plotter.set_xlabel(f"Perturbation parameter", subplot_index=q)

        plotter.set_xscale(scale="log", subplot_index=q)
        plotter.set_yscale(scale="log", subplot_index=q)

    plotter.adjust_layout(num_subplots=len(QI_list))

    filename = "data" + "/" + f"{problem_name}" + "/" + f"stability_region_SPP_{num_nodes=}.png"
    plotter.save(filename)


def finalize_plot_DAE(plotter, num_nodes, problem_name, QI_list):
    for q, QI in enumerate(QI_list):
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=24)

        plotter.set_ylabel(r"$|R(z)|$", subplot_index=q)
        plotter.set_xlabel(f"Time step size", subplot_index=q)

        plotter.set_xscale(scale="log", subplot_index=q)

    plotter.adjust_layout(num_subplots=len(QI_list))

    filename = "data" + "/" + f"{problem_name}" + "/" + f"stability_region_DAE_{num_nodes=}.png"
    plotter.save(filename)


if __name__ == "__main__":
    problem_name = "PROTHERO-ROBINSON"
    # problem_name = "LINEAR-TEST"

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = 16
    num_sweeps = 40
    do_coll_update = True

    t0 = 0.0

    num = 40
    dt_list = np.logspace(-6.0, 0.0, num=num)
    eps_list = np.logspace(-11.0, 0.0, num=num)

    # problems = {"SPP": eps_list}
    problems = {"embeddedDAE": [0.0]}

    Q_coefficients = compute_Q_coefficients(num_nodes)

    QI_coefficients = compute_QI_coefficients(Q_coefficients, QI_list)

    if "SPP" in problems.keys() or "SPP-yp" in problems.keys():
        plot_stability_region_SPP(dt_list, eps_list, num_nodes, num_sweeps, problem_name, problems, Q_coefficients, QI_coefficients)
    elif "embeddedDAE" in problems.keys():
        plot_stability_region_DAE(dt_list, num_nodes, num_sweeps, problem_name, problems, Q_coefficients, QI_coefficients)
