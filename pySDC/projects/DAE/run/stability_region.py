import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run.error import get_problem_cases, plot_result
from pySDC.projects.DAE import computeSolution, getEndTime, getColor, getLabel, get_linestyle, getMarker, Plotter


def finalize_plot(k: int, dt, plotter, num_nodes, problem_name, QI_list):
    for q, QI in enumerate(QI_list):
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=q, fontsize=24)

        plotter.set_ylabel(f"Time step size", subplot_index=q)
        plotter.set_xlabel(f"Perturbation parameter", subplot_index=q)

        plotter.set_xscale(scale="log", subplot_index=q)
        plotter.set_yscale(scale="log", subplot_index=q)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"stability_region_{num_nodes=}_{dt=}_case{k}.png"
    plotter.save(filename)


if __name__ == "__main__":
    problem_name = "PROTHERO-ROBINSON"

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = 15

    kwargs = {"e_tol": -1, "maxiter": 80}

    t0 = 0.0
    dt_list = np.linspace(1e-6, 1.0, num=251)

    eps_list = np.linspace(1e-11, 1.0, num=251)

    EPS, DT = np.meshgrid(eps_list, dt_list)

    problems = {"SPP": eps_list}

    hook_class = [LogSolution]

    for problem_type, eps_values in problems.items():
        region_plotter = Plotter(nrows=2, ncols=2, figsize=(12, 12))

        for q, QI in enumerate(QI_list):
            stabi_region = np.zeros((len(eps_list), len(dt_list)))

            for i, eps in enumerate(eps_values):
                for d, dt in enumerate(dt_list):
                    print(f"\n{QI}: Running test for {problem_type} with {eps=} using {dt=}...\n")

                    # Let's do the simulation to get results
                    solution_stats = computeSolution(
                        problemName=problem_name,
                        t0=t0,
                        dt=dt,
                        Tend=t0 + dt,
                        nNodes=num_nodes,
                        QI=QI,
                        problemType=problem_type,
                        eps=eps,
                        hookClass=hook_class,
                        **kwargs,
                    )

                    u_vals = np.array([me[1][1] for me in get_sorted(solution_stats, type="u", sortby="time")])
                    uend = u_vals[-1]

                    stabi_region[i, d] = np.abs(uend)

            # Define the stability threshold |uend| <= 1
            stability_mask = stabi_region <= 1

            region_plotter.contourf(EPS, DT, stability_mask, subplot_index=q, levels=[0, 1], colors=["gray"], alpha=0.6)

            region_plotter.contour(EPS, DT, stabi_region, levels=[0.5], colors="black", linewidths=1)

    finalize_plot(1, dt, region_plotter, num_nodes, problem_name, QI_list)