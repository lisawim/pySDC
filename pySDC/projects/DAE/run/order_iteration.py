import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter
from pySDC.projects.DAE.run.error import get_hooks, get_error_label, plot_result
from pySDC.projects.DAE.run.run_single_qi import choose_time_step_sizes
from pySDC.projects.DAE.misc.hooksDAE import LogGlobalErrorPreIterDifferentialVariable, LogGlobalErrorPreIterationAlgebraicVariable


def finalize_plot(plotter, num_nodes, problem_type, problem_name, QI):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    k : int
        Case number
    dt : float
        Time step size.
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    num_nodes : int
        Number of collocation nodes.
    problems : dict
        Contains different problem classes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    subplot_indices : tuple
        Subplot indices as tuple for y and z.
    """

    plotter.set_xlabel("time step size " + r"$\Delta t$", subplot_index=None)

    plotter.set_ylabel(f"local truncation error in " + r"$y$", subplot_index=0)
    plotter.set_ylabel(f"local truncation error in " + r"$z$", subplot_index=1)

    plotter.set_xscale(scale="log", subplot_index=0)
    plotter.set_xscale(scale="log", subplot_index=1)

    plotter.set_yscale(scale="log", subplot_index=0)
    plotter.set_yscale(scale="log", subplot_index=1)

    plotter.sync_ylim(min_y_set=1e-15)

    plotter.set_grid()

    plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=4, fontsize=22)
    # plotter.set_group_shared_legend([0], loc='lower center', bbox_to_anchor=(0.25, -0.2), ncol=2, fontsize=22)
    # plotter.set_group_shared_legend([1], loc='lower center', bbox_to_anchor=(0.75, -0.2), ncol=2, fontsize=22)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"order_iteration_{num_nodes=}_{problem_type}_{QI}.png"
    plotter.save(filename)


"""Main routine"""
if __name__ == "__main__":
    problem_name = "LINEAR-TEST"
    # problem_name = "MICHAELIS-MENTEN"
    # problem_name = "DPR"
    # problem_name = "ANDREWS-SQUEEZER"

    QI_list = ["IE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]
    num_nodes = 3

    p = 2 * num_nodes - 1

    t0 = 0.0
    dt_list = choose_time_step_sizes(problem_name)  # np.logspace(-2.0, -1.0, num=11)
    dt_list_short = dt_list[1 : 7]
    dt_ref = dt_list[0]

    case = 4

    hook_class = get_hooks(k=case, hook_for="iteration")
    hook_class_pre = [LogGlobalErrorPreIterDifferentialVariable, LogGlobalErrorPreIterationAlgebraicVariable]
    # hook_class += hook_class_pre
    # print(hook_class)
    problem_type = "constrainedDAE"
    eps = 0.0

    kwargs = {"solver_type": "direct"}

    colors = ["yellow", "gold", "orange", "red", "pink", "purple", "blue", "green", "black"]
    markers = ["o", "^", "h", "s", "d", "H", "*", "v", "D"]

    maxiter = 2 * num_nodes - 1
    for q, QI in enumerate(QI_list):
        print(QI)
        order_plotter = Plotter(nrows=1, ncols=2, figsize=(12, 6))

        errors_y, errors_z = [], []

        for dt in dt_list:
            hooks = hook_class[problem_type] if isinstance(hook_class, dict) else hook_class
            hooks += hook_class_pre
            # Let's do the simulation to get results
            solution_stats = computeSolution(
                problemName=problem_name,
                t0=t0,
                dt=dt,
                Tend=t0 + dt,
                nNodes=num_nodes,
                QI=QI,
                problemType=problem_type,
                hookClass=hooks,
                eps=eps,
                maxiter=maxiter,
                e_tol=-1,
                newton_tol=1e-12,
                **kwargs,
            )

            err_diff_values_spread = [me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_pre_iteration", sortby="iter")][0]
            err_alg_values_spread = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_pre_iteration", sortby="iter")][0]

            err_diff_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_post_iteration", sortby="iter")]
            err_alg_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")]

            err_diff_values.insert(0, err_diff_values_spread)
            err_alg_values.insert(0, err_alg_values_spread)

            errors_y.append(err_diff_values)
            errors_z.append(err_alg_values)

        for k in range(maxiter + 1):
            err_y_iter = [res[k] for res in errors_y]
            err_z_iter = [res[k] for res in errors_z]

            err_ref_y = err_y_iter[0]
            err_ref_z = err_z_iter[0]

            C_y = 1e0  # err_ref_y / dt_ref**(k+1)
            C_z = 1e0  # err_ref_z / dt_ref**(k+1)

            res = getMarker(problem_type, 0, QI)
            problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
            marker = res["marker"]
            markersize = res["markersize"]

            # Get things done in plot
            order_plotter = plot_result(
                order_plotter,
                dt_list,
                err_y_iter,
                0,
                colors[k],
                markers[k],
                markersize,
                "solid",
                f"k = {k}",
            )

            order_plotter = plot_result(
                order_plotter,
                dt_list,
                err_z_iter,
                1,
                colors[k],
                markers[k],
                markersize,
                "solid",
                None,# f"k = {k + 1}",
            )

        # for k in range(maxiter + 1):
            order_plotter = plot_result(
                order_plotter,
                dt_list_short,
                [C_y * dt ** (k + 1) for dt in dt_list_short],
                0,
                "black",
                None,  # marker,
                None,  # markersize,
                "dashed",
                None,# f"ref. order {k + 1}",
                linewidth=1.5,
            )

            order_plotter = plot_result(
                order_plotter,
                dt_list_short,
                [C_z * dt ** (k + 1) for dt in dt_list_short],
                1,
                "black",
                None,  # marker,
                None,  # markersize,
                "dashed",
                None,#f"ref. order = {k + 2}",
                linewidth=1.5,
            )

        finalize_plot(order_plotter, num_nodes, problem_type, problem_name, QI)
