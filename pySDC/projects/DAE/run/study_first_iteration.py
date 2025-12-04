import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.core.step import Step
from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.projects.DAE import my_setup_mpl, my_plot_style_config
from pySDC.projects.DAE.run.utils import setup_problem, setup_sweeper_sdc
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes, sync_ylim

from pySDC.projects.DAE.misc.configurations import BaseConfig

from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import (
    AndrewsSqueezingMechanismDAE,
    SemiImplicitAndrewsSqueezingMechanismDAE,
    AndrewsSqueezingMechanismDAEConstrained,
)
from pySDC.projects.DAE.problems.reactionDiffusionPDAE import (
    ReactionDiffusionPDAE, SemiImplicitReactionDiffusionPDAE, ReactionDiffusionPDAEConstrained
)
from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE


def plot_errorcoll_nodes_after_one_iteration(
        dt,
        num_nodes=3,
        problem_name="REACTION-DIFFUSION",
        config=BaseConfig(),
        use_mpi=False,
        journal="Springer_Scientific_Computing"
):
    figsize = figsize_by_journal(journal, scale=1.0, ratio=0.8)

    sweeper_types = ["constrainedDAE", "fullyImplicitDAE", "semiImplicitDAE"]
    QI_list = ["IE", "LU", "MIN-SR-S"]

    spectral = True
    kwargs = {"e_tol": -1, "maxiter": 2 * num_nodes - 1, "nvars": 256, "spectral": spectral}

    t0 = 0.0

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(3, 3, figsize=figsize)
    ax_flatten = axs.flatten()

    x = np.arange(num_nodes + 1)

    for q, QI in enumerate(QI_list):
        axs[0, q].set_title(QI + ": " + r"$k = 0$"), axs[1, q].set_title(QI + ": " + r"$k = 1$"), axs[2, q].set_title(QI + ": " + r"$k = 2M - 1$")

        for sweeper_type in sweeper_types:
            print(f"Running for {sweeper_type} with {QI}..")
            key = f"{sweeper_type}_LU"

            description = {}
            description["level_params"] = {"dt": dt}

            description = setup_problem(problem_name, QI, description, sweeper_type, **kwargs)

            assert QI in config.qDeltas

            description = setup_sweeper_sdc(
                description, num_nodes, sweeper_type, QI, use_mpi, **kwargs
            )

            # Initialize step
            S = Step(description=description)

            L = S.levels[0]
            P = L.prob

            L.status.time = t0

            # We do k = 0 here!
            L.u[0] = P.u_exact(L.time)
            L.sweep.predict()

            # Compute error in w at each node
            e0 = compute_error_w(t0, L, num_nodes)

            label = sweeper_labels[sweeper_type]
            axs[0, q].semilogy(x, e0, color=colors[key], marker=markers[key], label=label)

            S.status.iter = 0

            # We now do iteration k = 1 here!
            L.sweep.update_nodes()

            S.status.iter += 1

            e1 = compute_error_w(t0, L, num_nodes)

            label = sweeper_labels[sweeper_type]
            axs[1, q].semilogy(x, e1, color=colors[key], marker=markers[key], label=label)

            # Rest of iterations is done here!
            while S.status.iter < S.params.maxiter:
                L.sweep.update_nodes()

                S.status.iter += 1

            e_maxiter = compute_error_w(t0, L, num_nodes)

            label = sweeper_labels[sweeper_type]
            axs[2, q].semilogy(x, e_maxiter, color=colors[key], marker=markers[key], label=label)

    labels = [r"$t_0$"] + [fr"$\tau_{i}$".replace("i", str(i)) for i in range(1, num_nodes + 1)]

    for ax in ax_flatten:
        ax.set_ylabel("local truncation error in w")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.tick_params(axis='y', which='minor', length=0)

    ax_flatten = sync_ylim(ax_flatten)

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.00), ncol=3)

    plot_name = f"error_w_after_one_iteration_{num_nodes=}_{dt=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + "study_first_iteration" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def plot_error_solve_alg_constraints(
        dt,
        num_nodes=3,
        problem_name="REACTION-DIFFUSION",
        config=BaseConfig(),
        use_mpi=False,
        journal="Springer_Scientific_Computing",
):
    figsize = figsize_by_journal(journal, scale=1.0, ratio=0.8)

    sweeper_types = ["constrainedDAE", "fullyImplicitDAE", "semiImplicitDAE"]
    QI_list = ["IE", "LU", "MIN-SR-S"]

    spectral = True
    kwargs = {"e_tol": -1, "maxiter": 2 * num_nodes - 1, "nvars": 256, "spectral": spectral}

    t0 = 0.0

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(3, 3, figsize=figsize)
    ax_flatten = axs.flatten()

    x = np.arange(num_nodes + 1)

    for q, QI in enumerate(QI_list):
        axs[0, q].set_title(QI + ": " + r"$k = 0$"), axs[1, q].set_title(QI + ": " + r"$k = 1$"), axs[2, q].set_title(QI + ": " + r"$k = 2M - 1$")

        for sweeper_type in sweeper_types:
            print(f"Running for {sweeper_type} with {QI}..")
            key = f"{sweeper_type}_LU"

            description = {}
            description["level_params"] = {"dt": dt}

            description = setup_problem(problem_name, QI, description, sweeper_type, **kwargs)

            assert QI in config.qDeltas

            description = setup_sweeper_sdc(
                description, num_nodes, sweeper_type, QI, use_mpi, **kwargs
            )

            # Initialize step
            S = Step(description=description)

            L = S.levels[0]
            P = L.prob

            L.status.time = t0

            # We do k = 0 here!
            L.u[0] = P.u_exact(L.time)
            L.sweep.predict()

            # Evaluate algebraic constraints at each node
            g0 = eval_g(t0, L, num_nodes, sweeper_type)

            label = sweeper_labels[sweeper_type]
            axs[0, q].semilogy(x, g0, color=colors[key], marker=markers[key], label=label)

            S.status.iter = 0

            # We now do iteration k = 1 here!
            L.sweep.update_nodes()

            S.status.iter += 1

            g1 = eval_g(t0, L, num_nodes, sweeper_type)

            label = sweeper_labels[sweeper_type]
            axs[1, q].semilogy(x, g1, color=colors[key], marker=markers[key], label=label)

            # Rest of iterations is done here!
            while S.status.iter < S.params.maxiter:
                L.sweep.update_nodes()

                S.status.iter += 1

            g_maxiter = eval_g(t0, L, num_nodes, sweeper_type)

            label = sweeper_labels[sweeper_type]
            axs[2, q].semilogy(x, g_maxiter, color=colors[key], marker=markers[key], label=label)

    labels = [r"$t_0$"] + [fr"$\tau_{i}$".replace("i", str(i)) for i in range(1, num_nodes + 1)]

    for ax in ax_flatten:
        ax.set_ylabel(r"$|g(y^k,z^k)|$")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        ax.tick_params(axis='y', which='minor', length=0)

    ax_flatten = sync_ylim(ax_flatten)

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.00), ncol=3)

    plot_name = f"val_g_predict_first_iteration_{num_nodes=}_{dt=}_{spectral=}.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + "study_first_iteration" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def eval_g(t0, L, num_nodes, sweeper_type):
    taus = [t0] + list(L.time + L.dt * L.sweep.coll.nodes[:])
    g_nodes = []
    for m in range(num_nodes + 1):
        g = L.prob.algebraic_constraints(L.u[m], taus[m])
        g_nodes.append(abs(g))
    return g_nodes

def compute_error_w(t0, L, num_nodes):
    e_norm = []
    taus = [t0] + list(L.time + L.dt * L.sweep.coll.nodes[:])
    for m in range(num_nodes + 1):
        w = L.u[m].alg[:]  # in k = 0 we have w0 at all nodes

        w_true = L.prob.u_exact(taus[m]).alg[:]
        e_norm.append(np.linalg.norm(w - w_true))
    return e_norm

def plot_error_implicit_Euler_step(
        problem_name="REACTION-DIFFUSION", journal="Springer_Scientific_Computing"
):
    """
    We do the same as in the previous plotting function, but instead of L.sweep.update_nodes()
    we only call L.prob.solve_system() to imitate an implicit Euler step, i.e., only QI=IE.
    """

    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)

    sweeper_types = ["constrainedDAE", "fullyImplicitDAE", "semiImplicitDAE"]
    dt_list, _ = choose_time_step_sizes(problem_name)

    t0 = 0.0

    my_setup_mpl(fontsize=8)
    colors, markers, sweeper_labels = my_plot_style_config()

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    for sweeper_type in sweeper_types:
        print(f"Running for {sweeper_type}..")
        key = f"{sweeper_type}_LU"

        label = sweeper_labels[sweeper_type]

        err_u, err_v, err_w, err_g = [], [], [], []
        prob = get_problem_class(problem_name, sweeper_type)

        for dt in dt_list:
            u0 = prob.u_exact(t0)
            du0 = prob.du_exact(t0)
            rhs = u0.copy()
            factor = dt

            # u_init = u0 if sweeper_type == "constrainedDAE" else du0
            if sweeper_type == "constrainedDAE":
                u1 = prob.solve_system(rhs, factor, u0, t0 + dt)

            else:
                impl_sys = FullyImplicitDAE.F if problem_name == "ANDREWS-SQUEEZER" else None
                du1 = prob.solve_system(impl_sys, rhs, factor, du0, t0 + dt)
                u1 = recover_solution(dt, du1, prob, sweeper_type, u0)

            g = prob.algebraic_constraints(u1, t0 + dt)
            if problem_name == "ANDREWS-SQUEEZER":
                err_u.append(abs(u1.diff[: 7] - prob.u_exact(t0 + dt).diff[: 7]))
                err_v.append(abs(u1.diff[7 : 14] - prob.u_exact(t0 + dt).diff[7 : 14]))
                err_w.append(abs(u1.alg[: 7] - prob.u_exact(t0 + dt).alg[: 7]))    

            elif problem_name in ["REACTION-DIFFUSION"]:
                err_u.append(abs(u1.diff[: prob.nvars] - prob.u_exact(t0 + dt).diff[: prob.nvars]))
                err_v.append(abs(u1.diff[prob.nvars :] - prob.u_exact(t0 + dt).diff[prob.nvars :]))
                err_w.append(abs(u1.alg[: prob.nvars] - prob.u_exact(t0 + dt).alg[: prob.nvars]))
            err_g.append(abs(g))

        ax_flatten[0].loglog(dt_list, err_u, color=colors[key], marker=markers[key], label=label)
        ax_flatten[1].loglog(dt_list, err_v, color=colors[key], marker=markers[key], label=label)
        ax_flatten[2].loglog(dt_list, err_w, color=colors[key], marker=markers[key], label=label)
        ax_flatten[3].loglog(dt_list, err_g, color=colors[key], marker=markers[key], label=label)

        for i in range(3):
            ax_flatten[i].loglog(dt_list, [dt ** 2 for dt in dt_list], color="black", linewidth=0.9, linestyle="dashed")

    for ax in ax_flatten:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)

        ax.set_xlabel(r"$\Delta t$")

        ax.grid(linewidth=0.5)

    ax_flatten[0].set_ylabel("local truncation error in u")
    ax_flatten[1].set_ylabel("local truncation error in v")
    ax_flatten[2].set_ylabel("local truncation error in w")
    ax_flatten[3].set_ylabel(r"$|g(y,z)|$")

    ax_flatten = sync_ylim(ax_flatten, min_y_set=1e-16)

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    plot_name = f"error_implicit_Euler_step.png"
    filename = "data" + "/" + f"{problem_name}" + "/" + "study_first_iteration" + "/" + plot_name
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)

def get_problem_class(problem_name, sweeper_type):
    if problem_name in ["REACTION-FIFFUSION"]:
        problem_params = {"nvars": 256, "spectral": True}
        if sweeper_type == "fullyImplicitDAE":
            return ReactionDiffusionPDAE(**problem_params)
        elif sweeper_type == "semiImplicitDAE":
            return SemiImplicitReactionDiffusionPDAE(**problem_params)
        elif sweeper_type == "constrainedDAE":
            return ReactionDiffusionPDAEConstrained(**problem_params)
    elif problem_name == "ANDREWS-SQUEEZER":
        problem_params = {"index": 1, "solver_type": "newton"}
        if sweeper_type == "fullyImplicitDAE":
            return AndrewsSqueezingMechanismDAE(**problem_params)
        elif sweeper_type == "semiImplicitDAE":
            return SemiImplicitAndrewsSqueezingMechanismDAE(**problem_params)
        elif sweeper_type == "constrainedDAE":
            return AndrewsSqueezingMechanismDAEConstrained(**problem_params)

def recover_solution(dt, du, prob, sweeper_type, u0):
    u = prob.dtype_u(prob.init, val=0.0)
    if sweeper_type == "fullyImplicitDAE":
        u[:] = u0[:] + dt * du[:]
    elif sweeper_type == "semiImplicitDAE":
        u.diff[:] = u0.diff[:] + dt * du.diff[:]
        u.alg[:] = du.alg[:]
    return u


if __name__ == "__main__":
    dt = 1e-3 #1e-2
    # problem_name = "REACTION-DIFFUSION"
    problem_name = "ANDREWS-SQUEEZER"
    # plot_errorcoll_nodes_after_one_iteration(dt=dt, problem_name=problem_name)
    plot_error_solve_alg_constraints(dt=dt, problem_name=problem_name)
    plot_error_implicit_Euler_step(problem_name=problem_name)