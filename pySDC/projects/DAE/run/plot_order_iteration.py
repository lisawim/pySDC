from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE.run.utils import compute_solution, my_setup_mpl


def choose_time_step_sizes(problem_name):
    """Returns time step sizes suitable for each problem."""
    if problem_name == "ANDREWS-SQUEEZER":
        n_steps_list = [30, 60, 100, 300, 600, 1000, 3000, 6000]
        Tend = 0.03
    elif problem_name == "LINEAR-TEST":
        n_steps_list = [2, 5, 10, 20, 50, 100, 200, 500]
        Tend = 1.0
    elif problem_name == "REACTION-DIFFUSION":
        n_steps_list = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        Tend = 0.25
    elif problem_name == "SIMPLE-DAE":
        n_steps_list = [100, 200, 500, 1000, 2000, 5000, 10000]#[10, 20, 50, 100, 200, 500, 1000]
        Tend = 1.0
    else:
        raise NotImplementedError

    dt_list = [Tend / n_steps for n_steps in n_steps_list]
    return dt_list, Tend


def compute_constant_reference_order(dt_list, err_iter, k):
    """Computes constants to shift reference order lines in plot."""

    dt_ref = dt_list[0]
    err_ref = err_iter[0]

    C = err_ref / dt_ref ** (k + 1)
    return C


def get_error_values(solution_stats, variable, mode):
    sort_key = "sweep" if mode == "sweep" else "iter"
    pre_suffix = f"pre_{mode}"
    post_suffix = f"post_{mode}"

    err_pre = [
        me[1]
        for me in get_sorted(
            solution_stats,
            type=f"e_global_{variable}_{pre_suffix}",
            sortby=sort_key,
        )
    ][0]

    err_post = [
        me[1]
        for me in get_sorted(
            solution_stats,
            type=f"e_global_{variable}_{post_suffix}",
            sortby=sort_key,
        )
    ]
    return [err_pre] + err_post


def sync_xlim(axs, min_x_set=1e-15):
    """Synchronize x-axis limits across all subplots by finding the global min/max."""
    min_x, max_x = None, None

    # Find global min/max y-limits across all axes
    for ax in axs:
        x_limits = ax.get_xlim()

        # Ignore non-positive values for log scale
        if ax.get_xscale() == "log":
            x_limits = [x for x in x_limits if x > 0]  
            if not x_limits:
                continue

        if min_x is None or x_limits[0] < min_x:
            min_x = x_limits[0]
        if max_x is None or x_limits[1] > max_x:
            max_x = x_limits[1]

    # Apply the same limits to all subplots
    for ax in axs:
        if ax.get_xscale() == "log":
            if min_x is not None and min_x <= 0:
                min_x = min_x_set
            ax.set_xlim(min_x, max_x)
        else:
            ax.set_xlim(min_x, max_x)

    return axs


def sync_ylim(axs, min_y_set=1e-15):
    """Synchronize y-axis limits across all subplots by finding the global min/max."""
    min_y, max_y = None, None

    # Find global min/max y-limits across all axes
    for ax in axs:
        y_limits = ax.get_ylim()

        # Ignore non-positive values for log scale
        if ax.get_yscale() == "log":
            y_limits = [y for y in y_limits if y > 0]
            if not y_limits:
                continue

        if min_y is None or y_limits[0] < min_y:
            min_y = y_limits[0]
        if max_y is None or y_limits[1] > max_y:
            max_y = y_limits[1]

    # Apply the same limits to all subplots
    for ax in axs:
        if ax.get_yscale() == "log":
            if min_y is not None and min_y <= 0:
                min_y = min_y_set
            ax.set_ylim(min_y, max_y)
        else:
            ax.set_ylim(min_y, max_y)

    return axs


def save_fig(plot, plot_name, problem_name, pad_inches=0.01):
    out = Path("data") / problem_name / f"{plot_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(out, dpi=600, bbox_inches="tight")#, pad_inches=0.01)
    plot.close()


def plot_order_andrews(num_nodes=3, sweeper_type="SDC-C", journal="BUW_thesis"):
    """Plots the order in each iteration for Andrews' problem"""

    from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import (
        LogGlobalErrorMechanicalVars,
    )

    problem_name = "ANDREWS-SQUEEZER"
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)

    colors = ["yellow", "gold", "orange", "red", "pink", "mediumpurple"]
    markers = ["o", "^", "h", "s", "d", "H", "*", "v", "D"]
    linestyles = ["solid", "dotted"]

    QI_list = ["IE", "LU", "MIN-SR-NS", "MIN-SR-S"]
    maxiter = 2 * num_nodes - 1
    e_tol = -1

    kwargs = {"e_tol": e_tol}

    t0 = 0.0
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[3:7]

    hook_class = [LogGlobalErrorMechanicalVars]

    my_setup_mpl(fontsize=8)
    plt.rcParams['patch.linewidth'] = 0.3

    offsets_pos = [0.8, 0.1, 0.6, 0.25, 0.55, 0.5, 0.45]
    offsets_vel = [0.15, 0.57, 0.6, 0.55, 0.55, 0.5, 0.35]
    offsets_acc = [0.8, 0.1, 0.6, 0.55, 0.55, 0.5, 0.45]
    offsets_lag = [0.8, 0.1, 0.6, 0.1, 0.55, 0.5, 0.45]

    for q, QI in enumerate(QI_list):
        print(f"Running for {QI}..")
        errors_pos, errors_vel = [], []
        errors_acc, errors_lag = [], []

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        ax_flatten = axs.flatten()

        for dt in dt_list:
            solution_stats = compute_solution(
                problem_name=problem_name,
                t0=t0,
                dt=dt,
                Tend=t0 + dt,
                num_nodes=num_nodes,
                QI=QI,
                sweeper_type=sweeper_type,
                use_mpi=False,
                hook_class=hook_class,
                measure=False,
                maxiter=maxiter,
                nsweeps=1,
                initial_guess="spread",
                **kwargs,
            )

            errors_pos.append(get_error_values(solution_stats, "position", "iteration"))
            errors_vel.append(get_error_values(solution_stats, "velocity", "iteration"))
            errors_acc.append(get_error_values(solution_stats, "acceleration", "iteration"))
            errors_lag.append(get_error_values(solution_stats, "lagrange", "iteration"))

        for k in range(len(errors_pos[0])):
            err_pos_iter = [res[k] for res in errors_pos]
            err_vel_iter = [res[k] for res in errors_vel]
            err_acc_iter = [res[k] for res in errors_acc]
            err_lag_iter = [res[k] for res in errors_lag]

            ax_flatten[0].loglog(
                dt_list,
                err_pos_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
                label=f"k = {k}",
            )

            ax_flatten[1].loglog(
                dt_list,
                err_vel_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
            )

            ax_flatten[2].loglog(
                dt_list,
                err_acc_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
            )

            ax_flatten[3].loglog(
                dt_list,
                err_lag_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
            )

            C_pos = compute_constant_reference_order(dt_list, err_pos_iter, k)
            C_vel = compute_constant_reference_order(dt_list, err_vel_iter, k)
            C_acc = compute_constant_reference_order(dt_list, err_acc_iter, k)
            C_lag = compute_constant_reference_order(dt_list, err_lag_iter, k)

            # Reference order
            ref_pos = [C_pos * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[0].loglog(
                dt_list_short,
                ref_pos,
                color="black",
                linewidth=1.0,
                linestyle="dashed",
            )

            ax_flatten[0].text(
                dt_list_short[-1] * 0.75,
                ref_pos[-1] * offsets_pos[k],
                rf"${k+1}$",
                fontsize=7,
                va="center",
                ha="left",
                color="black",
            )

            ref_vel = [C_vel * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[1].loglog(
                dt_list_short,
                ref_vel,
                color="black",
                linewidth=1.0,
                linestyle="dashed",
            )

            ax_flatten[1].text(
                dt_list_short[-1] * 0.75,
                ref_vel[-1] * offsets_vel[k],
                rf"${k+1}$",
                fontsize=7,
                va="center",
                ha="left",
                color="black",
            )

            ref_acc = [C_acc * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[2].loglog(
                dt_list_short,
                ref_acc,
                color="black",
                linewidth=1.0,
                linestyle="dashed",
            )

            ax_flatten[2].text(
                dt_list_short[-1] * 0.75,
                ref_acc[-1] * offsets_acc[k],
                rf"${k+1}$",
                fontsize=7,
                va="center",
                ha="left",
                color="black",
            )

            ref_lag = [C_lag * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[3].loglog(
                dt_list_short,
                ref_lag,
                color="black",
                linewidth=1.0,
                linestyle="dashed",
            )

            ax_flatten[3].text(
                dt_list_short[-1] * 0.75,
                ref_lag[-1] * offsets_lag[k],
                rf"${k+1}$",
                fontsize=7,
                va="center",
                ha="left",
                color="black",
            )

        for ax in ax_flatten:
            ax.tick_params(axis="both", which="minor", bottom=False, left=False)

            ax.set_xlabel(r"time step size $\Delta t$")

            ax.grid(linewidth=0.5)

        ax_flatten[0].set_ylabel(r"LTE $||q(t_1) - q^k_{M,t_1}||_{\infty}$")
        ax_flatten[1].set_ylabel(r"LTE $||v(t_1) - v^k_{M,t_1}||_{\infty}$")
        ax_flatten[2].set_ylabel(r"LTE $||w(t_1) - w^k_{M,t_1}||_{\infty}$")
        ax_flatten[3].set_ylabel(r"LTE $||\lambda(t_1) - \lambda^k_{M,t_1}||_{\infty}$")

        ax_flatten = sync_ylim(ax_flatten, min_y_set=1e-15)

        handles, labels = ax_flatten[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.02), ncol=3)

        plot_name = f"order_iteration_andrews_index_two_{num_nodes=}_{sweeper_type}_{QI}"
        # plot_name = f"order_iteration_andrews_{num_nodes=}_{sweeper_type}_{QI}"
        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + ".png"
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=500, bbox_inches="tight")
        plt.close(fig)


def plot_order_simple_dae(num_nodes=2, sweeper_type="SDC-C", journal="Springer_proceedings"):
    """Plots the order in each iteration."""

    from pySDC.projects.DAE.problems.simpleDAE import (
        LogGlobalErrorSimpleDAE
    )

    problem_name = "SIMPLE-DAE"
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)

    colors = ["yellow", "gold", "orange", "red", "pink", "mediumpurple", "yellow", "gold", "orange", "red", "pink", "mediumpurple"]
    markers = ["o", "^", "h", "s", "d", "H", "*", "v", "D", "o", "^", "h", "s", "d", "H", "*", "v", "D"]
    linestyles = ["solid", "dotted"]

    QE_list = ["IE", "LU"]
    maxiter = 2 * num_nodes - 1
    e_tol = -1

    kwargs = {"e_tol": e_tol, "maxiter": maxiter}

    t0 = 0.0
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[1:5]

    hook_class = [LogGlobalErrorSimpleDAE]

    my_setup_mpl(fontsize=6)

    offsets = [0.7, 0.45, 0.6, 0.55, 0.55, 0.5, 0.45, 0.45, 0.4, 0.4, 0.35, 0.35]

    for q, QE in enumerate(QE_list):
        err_u1_values_spread, err_u2_values_spread, err_z_values_spread = [], [], []
        errors_u1, errors_u2, errors_z = [], [], []

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        ax_flat = axs.flatten()

        for dt in dt_list:
            solution_stats = compute_solution(
                problem_name=problem_name,
                t0=t0,
                dt=dt,
                Tend=t0 + dt,
                num_nodes=num_nodes,
                QI=QE,
                sweeper_type=sweeper_type,
                use_mpi=False,
                hook_class=hook_class,
                measure=False,
                initial_guess="spread",
                **kwargs,
            )

            err_u1_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_u1_pre_iteration", sortby="iter")
            ][0]
            err_u2_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_u2_pre_iteration", sortby="iter")
            ][0]
            err_z_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_z_pre_iteration", sortby="iter")
            ][0]

            err_u1_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_u1_post_iteration", sortby="iter")
            ]
            err_u2_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_u2_post_iteration", sortby="iter")
            ]
            err_z_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_z_post_iteration", sortby="iter")
            ]

            err_u1_values.insert(0, err_u1_values_spread)
            err_u2_values.insert(0, err_u2_values_spread)
            err_z_values.insert(0, err_z_values_spread)

            errors_u1.append(err_u1_values)
            errors_u2.append(err_u2_values)
            errors_z.append(err_z_values)

        for k in range(maxiter + 1):
            err_u1_iter = [res[k] for res in errors_u1]
            err_u2_iter = [res[k] for res in errors_u2]
            err_z_iter = [res[k] for res in errors_z]

            ax_flat[0].loglog(
                dt_list,
                err_u1_iter,
                linewidth=0.7,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
                label=f"k = {k}",
            )

            ax_flat[1].loglog(
                dt_list,
                err_u2_iter,
                linewidth=0.7,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
                label=f"k = {k}",
            )

            ax_flat[2].loglog(
                dt_list,
                err_z_iter,
                linewidth=0.7,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
            )

            Cu1 = compute_constant_reference_order(dt_list, err_u1_iter, k)
            Cu2 = compute_constant_reference_order(dt_list, err_u2_iter, k)
            Cz = compute_constant_reference_order(dt_list, err_z_iter, k)

            # Reference order
            p_diff = k+1#k + 2 if sweeper_type == "HF-SDC-C" else k + 1
            ref_u1 = [Cu1 * dt ** p_diff for dt in dt_list_short]
            ax_flat[0].loglog(
                dt_list_short,
                ref_u1,
                linewidth=0.7,
                color="black",
                linestyle="dashed",
            )

            ax_flat[0].text(
                dt_list_short[-1] * 0.8,
                ref_u1[-1] * offsets[k],
                rf"${p_diff}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

            ref_u2 = [Cu2 * dt ** p_diff for dt in dt_list_short]
            ax_flat[1].loglog(
                dt_list_short,
                ref_u2,
                linewidth=0.7,
                color="black",
                linestyle="dashed",
            )

            ax_flat[1].text(
                dt_list_short[-1] * 0.8,
                ref_u2[-1] * offsets[k],
                rf"${p_diff}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

            p_alg = k + 1
            ref_z = [Cz * dt ** p_alg for dt in dt_list_short]
            ax_flat[2].loglog(
                dt_list_short,
                ref_z,
                linewidth=0.7,
                color="black",
                linestyle="dashed",
            )

            ax_flat[2].text(
                dt_list_short[-1] * 0.8,
                ref_z[-1] * offsets[k],
                rf"${p_alg}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

        ax_flat[3].remove()

        for ax in ax_flat:
            ax.tick_params(axis="both", which="minor", bottom=False, left=False)

            ax.set_xlabel(r"time step size $\Delta t$")

            ax.grid(linewidth=0.5)

        ax_flat[0].set_ylabel(r"LTE $||u_1(t_1) - u^k_{1,M,t_1}||_{\infty}$")
        ax_flat[1].set_ylabel(r"LTE $||u_2(t_1) - u^k_{2,M,t_1}||_{\infty}$")
        ax_flat[2].set_ylabel(r"LTE $||z(t_1) - z^k_{M,t_1}||_{\infty}$")

        ax_flat = sync_ylim(ax_flat, min_y_set=1e-15)

        handles, labels = ax_flat[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

        plot_name = f"order_iteration_simple_dae_{num_nodes=}_{sweeper_type}_{QE}"
        save_fig(plt, plot_name, problem_name)


if __name__ == "__main__":
    plot_order_andrews()
    # plot_order_simple_dae()
    # plot_order_reaction_diffusion(num_nodes=5, journal="BUW_thesis")
    # plot_order_reaction_diffusion(num_nodes=5, sweeper_type="imexConstrainedDAE", journal="BUW_thesis")
