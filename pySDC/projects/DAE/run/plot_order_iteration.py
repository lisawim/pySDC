from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE import my_setup_mpl
from pySDC.projects.DAE.run.utils import compute_solution


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


def plot_order_linear(sweeper_type="constrainedDAE", journal="Springer_Scientific_Computing"):
    """Plots the order in each iteration."""

    from pySDC.projects.DAE.misc.hooksDAE import (
        LogGlobalErrorDiffVar,
        LogGlobalErrorAlgVar,
    )

    problem_name = "LINEAR-TEST"
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.5)

    colors = ["yellow", "gold", "orange", "red", "pink", "mediumpurple"]
    markers = ["o", "^", "h", "s", "d", "H", "*", "v", "D"]
    linestyles = ["solid", "dotted"]

    QI_list = ["EE", "IE", "LU", "MIN-SR-S", "MIN-SR-NS", "MIN-SR-FLEX", "Picard"]
    num_nodes = 3
    maxiter = 2 * num_nodes - 1
    e_tol = -1

    kwargs = {"e_tol": e_tol, "maxiter": maxiter}

    t0 = 0.0
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[3:7]

    hook_class = [LogGlobalErrorDiffVar, LogGlobalErrorAlgVar]

    my_setup_mpl(fontsize=6)

    offsets = [0.7, 0.45, 0.6, 0.55, 0.55, 0.5, 0.45]

    for q, QI in enumerate(QI_list):
        errors_y, errors_z = [], []

        fig, axs = plt.subplots(1, 2, figsize=figsize)

        for dt in dt_list:
            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                t0 + dt,
                num_nodes,
                QI,
                sweeper_type,
                False,
                hook_class=hook_class,
                measure=False,
                **kwargs,
            )

            err_diff_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_pre_iteration", sortby="iter")
            ][0]
            err_alg_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_pre_iteration", sortby="iter")
            ][0]

            err_diff_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_post_iteration", sortby="iter")
            ]
            err_alg_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_iteration", sortby="iter")
            ]

            err_diff_values.insert(0, err_diff_values_spread)
            err_alg_values.insert(0, err_alg_values_spread)

            errors_y.append(err_diff_values)
            errors_z.append(err_alg_values)

        for k in range(maxiter + 1):
            err_y_iter = [res[k] for res in errors_y]
            err_z_iter = [res[k] for res in errors_z]

            axs[0].loglog(
                dt_list,
                err_y_iter,
                linewidth=0.7,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
                label=f"k = {k}",
            )

            axs[1].loglog(
                dt_list,
                err_z_iter,
                linewidth=0.7,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
            )

            Cy = compute_constant_reference_order(dt_list, err_y_iter, k)
            Cz = compute_constant_reference_order(dt_list, err_z_iter, k)

            # Reference order
            ref_y = [Cy * dt ** (k + 1) for dt in dt_list_short]
            axs[0].loglog(
                dt_list_short,
                ref_y,
                linewidth=0.7,
                color="black",
                linestyle="dashed",
            )

            axs[0].text(
                dt_list_short[-1] * 0.8,
                ref_y[-1] * offsets[k],
                rf"${k+1}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

            ref_z = [Cz * dt ** (k + 1) for dt in dt_list_short]
            axs[1].loglog(
                dt_list_short,
                ref_z,
                linewidth=0.7,
                color="black",
                linestyle="dashed",
            )

            axs[1].text(
                dt_list_short[-1] * 0.8,
                ref_z[-1] * offsets[k],
                rf"${k+1}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

        for ax in axs:
            ax.tick_params(axis="both", which="minor", bottom=False, left=False)

            ax.set_xlabel(r"time step size $\Delta t$")

            ax.grid(linewidth=0.5)

        axs[0].set_ylabel(r"LTE $||y(t_1) - y^k_{M,t_1}||_{\infty}$")
        axs[1].set_ylabel(r"LTE $||z(t_1) - z^k_{M,t_1}||_{\infty}$")

        axs = sync_ylim(axs, min_y_set=1e-15)

        handles, labels = axs[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

        # plot_name = "Fig3" if QI == "MIN-SR-NS" else f"order_iteration_linear_{num_nodes=}_{sweeper_type}_{QI}"
        plot_name = f"order_iteration_linear_{num_nodes=}_{sweeper_type}_{QI}"
        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + ".png"
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)


def plot_order_andrews(sweeper_type="constrainedDAE", journal="Springer_Scientific_Computing"):
    """Plots the order in each iteration for Andrews' problem"""

    from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import (
        LogGlobalErrorPreIterMechanicalVars,
        LogGlobalErrorPostIterMechanicalVars,
    )

    problem_name = "ANDREWS-SQUEEZER"
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)
    figsize_g = figsize_by_journal(journal, scale=0.5, ratio=0.9)

    colors = ["yellow", "gold", "orange", "red", "pink", "mediumpurple"]
    markers = ["o", "^", "h", "s", "d", "H", "*", "v", "D"]
    linestyles = ["solid", "dotted"]

    QI_list = ["IE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]
    num_nodes = 3
    maxiter = 2 * num_nodes - 1
    e_tol = -1

    kwargs = {"e_tol": e_tol, "maxiter": maxiter}

    t0 = 0.0
    dt_list, Tend = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[3:7]

    hook_class = [
        LogGlobalErrorPreIterMechanicalVars,
        LogGlobalErrorPostIterMechanicalVars,
    ]

    my_setup_mpl(fontsize=7)

    offsets_pos = [0.8, 0.1, 0.6, 0.25, 0.55, 0.5, 0.45]
    offsets_vel = [0.15, 0.57, 0.6, 0.55, 0.55, 0.5, 0.35]
    offsets_acc = [0.8, 0.1, 0.6, 0.55, 0.55, 0.5, 0.45]
    offsets_lag = [0.8, 0.1, 0.6, 0.1, 0.55, 0.5, 0.45]

    for q, QI in enumerate(QI_list):
        print(f"Running for {QI}..")
        abs_g_vals = []
        errors_pos, errors_vel = [], []
        errors_acc, errors_lag = [], []

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        ax_flatten = axs.flatten()

        fig_g, axs_g = plt.subplots(1, 1, figsize=figsize_g)

        for dt in dt_list:
            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                t0 + dt,
                num_nodes,
                QI,
                sweeper_type,
                False,
                hook_class=hook_class,
                measure=False,
                **kwargs,
            )

            err_pos_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_position_pre_iteration", sortby="iter")
            ][0]
            err_vel_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_velocity_pre_iteration", sortby="iter")
            ][0]
            err_acc_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_acceleration_pre_iteration", sortby="iter")
            ][0]
            err_lag_values_spread = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_lagrange_pre_iteration", sortby="iter")
            ][0]

            err_pos_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_position_post_iteration", sortby="iter")
            ]
            err_vel_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_velocity_post_iteration", sortby="iter")
            ]
            err_acc_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_acceleration_post_iteration", sortby="iter")
            ]
            err_lag_values = [
                me[1] for me in get_sorted(solution_stats, type=f"e_global_lagrange_post_iteration", sortby="iter")
            ]

            err_pos_values.insert(0, err_pos_values_spread)
            err_vel_values.insert(0, err_vel_values_spread)
            err_acc_values.insert(0, err_acc_values_spread)
            err_lag_values.insert(0, err_lag_values_spread)

            errors_pos.append(err_pos_values)
            errors_vel.append(err_vel_values)
            errors_acc.append(err_acc_values)
            errors_lag.append(err_lag_values)

        for k in range(maxiter + 1):
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
                linewidth=0.9,
                linestyle="dashed",
            )

            ax_flatten[0].text(
                dt_list_short[-1] * 0.75,
                ref_pos[-1] * offsets_pos[k],
                rf"${k+1}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

            ref_vel = [C_vel * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[1].loglog(
                dt_list_short,
                ref_vel,
                color="black",
                linewidth=0.9,
                linestyle="dashed",
            )

            ax_flatten[1].text(
                dt_list_short[-1] * 0.75,
                ref_vel[-1] * offsets_vel[k],
                rf"${k+1}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

            ref_acc = [C_acc * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[2].loglog(
                dt_list_short,
                ref_acc,
                color="black",
                linewidth=0.9,
                linestyle="dashed",
            )

            ax_flatten[2].text(
                dt_list_short[-1] * 0.75,
                ref_acc[-1] * offsets_acc[k],
                rf"${k+1}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

            ref_lag = [C_lag * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[3].loglog(
                dt_list_short,
                ref_lag,
                color="black",
                linewidth=0.9,
                linestyle="dashed",
            )

            ax_flatten[3].text(
                dt_list_short[-1] * 0.75,
                ref_lag[-1] * offsets_lag[k],
                rf"${k+1}$",
                fontsize=6,
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
        y_limits = ax_flatten[0].get_ylim()
        axs_g.set_ylim(y_limits)

        handles, labels = ax_flatten[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

        plot_name = "Fig6" if QI == "MIN-SR-NS" else f"order_iteration_andrews_{num_nodes=}_{sweeper_type}_{QI}"
        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + ".png"
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)


def plot_order_reaction_diffusion(format="eps", sweeper_type="constrainedDAE", journal="Springer_Scientific_Computing"):
    """Plots the order in each iteration for reaction-diffusion problem"""

    from pySDC.projects.DAE.problems.reactionDiffusionPDAE import (
        LogGlobalErrorPreIterConcentrations,
        LogGlobalErrorPostIterConcentrations,
    )

    problem_name = "REACTION-DIFFUSION"
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)

    colors = ["yellow", "gold", "orange", "red", "pink", "mediumpurple"]
    markers = ["o", "^", "h", "s", "d", "H", "*", "v", "D", "<", ">", "o", "^"]
    linestyles = ["solid", "dotted"]

    QI_list = ["IE", "LU", "MIN-SR-S"]
    num_nodes = 3
    maxiter = 2 * num_nodes - 1
    e_tol = -1

    kwargs = {"e_tol": e_tol, "maxiter": maxiter}

    t0 = 0.0
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_list_pov = dt_list[3:]
    dt_list_short = dt_list_pov[:4]

    hook_class = [
        LogGlobalErrorPreIterConcentrations,
        LogGlobalErrorPostIterConcentrations,
    ]

    my_setup_mpl(fontsize=8)

    offsets = [0.18, 0.18, 0.2, 0.25, 0.3, 0.3, 0.35]

    for q, QI in enumerate(QI_list):
        print(f"Running for {QI}..")
        errors_u, errors_v = [], []
        errors_w = []

        fig, axs = plt.subplots(2, 2, figsize=figsize)
        ax_flatten = axs.flatten()

        for dt in dt_list_pov:
            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                t0 + dt,
                num_nodes,
                QI,
                sweeper_type,
                False,
                hook_class=hook_class,
                measure=False,
                **kwargs,
            )

            err_u_values_spread = [
                me[1]
                for me in get_sorted(solution_stats, type=f"e_global_concentration_u_pre_iteration", sortby="iter")
            ][0]
            err_v_values_spread = [
                me[1]
                for me in get_sorted(solution_stats, type=f"e_global_concentration_v_pre_iteration", sortby="iter")
            ][0]
            err_w_values_spread = [
                me[1]
                for me in get_sorted(solution_stats, type=f"e_global_concentration_w_pre_iteration", sortby="iter")
            ][0]

            err_u_values = [
                me[1]
                for me in get_sorted(solution_stats, type=f"e_global_concentration_u_post_iteration", sortby="iter")
            ]
            err_v_values = [
                me[1]
                for me in get_sorted(solution_stats, type=f"e_global_concentration_v_post_iteration", sortby="iter")
            ]
            err_w_values = [
                me[1]
                for me in get_sorted(solution_stats, type=f"e_global_concentration_w_post_iteration", sortby="iter")
            ]

            err_u_values.insert(0, err_u_values_spread)
            err_v_values.insert(0, err_v_values_spread)
            err_w_values.insert(0, err_w_values_spread)

            errors_u.append(err_u_values)
            errors_v.append(err_v_values)
            errors_w.append(err_w_values)

        for k in range(maxiter + 1):
            err_u_iter = [res[k] for res in errors_u]
            err_v_iter = [res[k] for res in errors_v]
            err_w_iter = [res[k] for res in errors_w]

            ax_flatten[0].loglog(
                dt_list_pov,
                err_u_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
                label=f"k = {k}",
            )

            ax_flatten[1].loglog(
                dt_list_pov,
                err_v_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
            )

            ax_flatten[2].loglog(
                dt_list_pov,
                err_w_iter,
                color=colors[k],
                marker=markers[k],
                linestyle=linestyles[k % 2],
            )

            C_u = compute_constant_reference_order(dt_list_pov, err_u_iter, k)
            C_v = compute_constant_reference_order(dt_list_pov, err_v_iter, k)
            C_w = compute_constant_reference_order(dt_list_pov, err_w_iter, k)

            # Reference order
            ref_u = [C_u * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[0].loglog(
                dt_list_short,
                ref_u,
                color="black",
                linewidth=0.9,
                linestyle="dashed",
            )

            ax_flatten[0].text(
                dt_list_short[-1] * 0.82,
                ref_u[-1] * offsets[k],
                rf"${k+1}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

            ref_v = [C_v * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[1].loglog(
                dt_list_short,
                ref_v,
                color="black",
                linewidth=0.9,
                linestyle="dashed",
            )

            ax_flatten[1].text(
                dt_list_short[-1] * 0.82,
                ref_v[-1] * offsets[k],
                rf"${k+1}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

            ref_w = [C_w * dt ** (k + 1) for dt in dt_list_short]
            ax_flatten[2].loglog(
                dt_list_short,
                ref_w,
                color="black",
                linewidth=0.9,
                linestyle="dashed",
            )

            ax_flatten[2].text(
                dt_list_short[-1] * 0.82,
                ref_w[-1] * offsets[k],
                rf"${k+1}$",
                fontsize=6,
                va="center",
                ha="left",
                color="black",
            )

        for ax in ax_flatten:
            ax.tick_params(axis="both", which="minor", bottom=False, left=False)

            ax.set_xlabel(r"time step size $\Delta t$")

            ax.grid(linewidth=0.5)

        ax_flatten[0].set_ylabel(r"LTE $||u(t_1) - u^k_{M,t_1}||_{\infty}$")
        ax_flatten[1].set_ylabel(r"LTE $||v(t_1) - v^k_{M,t_1}||_{\infty}$")
        ax_flatten[2].set_ylabel(r"LTE $||w(t_1) - w^k_{M,t_1}||_{\infty}$")

        ax_flatten = sync_ylim(ax_flatten, min_y_set=5e-17)

        handles, labels = ax_flatten[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

        ax_flatten[3].remove()

        plot_name = "Fig10" if QI == "IE" else f"order_iteration_{num_nodes=}_{sweeper_type}_{QI}"
        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + ".png"
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    plot_order_linear(journal="BUW_thesis")
    plot_order_linear(sweeper_type="semiImplicitDAE", journal="BUW_thesis")
