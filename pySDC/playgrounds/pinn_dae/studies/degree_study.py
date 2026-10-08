import torch
import numpy as np
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.playgrounds.pinn_dae.utils import my_setup_mpl
from pySDC.playgrounds.pinn_dae.studies.order_study import choose_time_step_sizes
from pySDC.playgrounds.pinn_dae.experiments.sdc_provisional_solution import (
    save_fig,
    u_exact,
    get_collocation,
)


def study_linear_dae_across_degree(problem_name, degree_list, num_nodes):
    from pySDC.playgrounds.pinn_dae.models.linear_dae_lpnet import LinearDAELPNet
    from pySDC.playgrounds.pinn_dae.trainers.trainer_linear_dae_lpnet import TrainerLinearDAELPNet

    t0 = torch.tensor([[0.0]], dtype=torch.float64)

    y0, z0 = u_exact(t0)

    tau_train = torch.linspace(
        0.0,
        1.0,
        1000,
        dtype=torch.float64,
    ).reshape(-1, 1)

    _, nodes = get_collocation(num_nodes)

    histories = {}
    for degree in degree_list:
        print(f"\nTraining for degree = {degree}")

        model = LinearDAELPNet(
            y0=y0,
            z0=z0,
            t0=t0,
            degree_y=degree,
            degree_z=degree,
            fix_a1=False,
        ).double()

        trainer = TrainerLinearDAELPNet(model=model)

        train_diagnostics = trainer.train_model(tau_train=tau_train)

        _, diagnostics = trainer.compute_loss(tau_train=tau_train, model=model)

        final_diagnostics = {
            key: value.item() for key, value in diagnostics.items()
        }

        # model.print_coefficients()

        # The model is evaluated at normalized collocation nodes.
        t_eval = np.concatenate(
            (
                np.asarray(t0, dtype=np.float64).reshape(-1),
                np.asarray(nodes, dtype=np.float64).reshape(-1),
            )
        )
        t_eval_tensor = torch.as_tensor(t_eval, dtype=torch.float64).reshape(-1, 1)

        y_pred_eval = model.evaluate_y(t_eval_tensor)
        z_pred_eval = model.evaluate_z(t_eval_tensor)

        # Exact solution is evaluated at the corresponding physical times.
        y_ex_eval, z_ex_eval = u_exact(t_eval_tensor)
        y_ex_eval = y_ex_eval.detach()
        z_ex_eval = z_ex_eval.detach()

        assert y_ex_eval.shape == y_pred_eval.shape
        assert z_ex_eval.shape == z_pred_eval.shape

        error_y = torch.max(torch.abs(y_pred_eval - y_ex_eval)).item()
        error_z = torch.max(torch.abs(z_pred_eval - z_ex_eval)).item()

        histories[degree] = {
            "train": train_diagnostics,
            "final": final_diagnostics,
            "error_y": error_y,
            "error_z": error_z,
            "t_eval": t_eval,
            "y_pred": y_pred_eval,
            "z_pred": z_pred_eval,
        }

    return histories


def study_nonstiff_linear_ode_across_degree(problem_name, degree_list, num_nodes):
    from pySDC.playgrounds.pinn_dae.models.nonstiff_linear_lpnet import NonstiffLinearLPNet
    from pySDC.playgrounds.pinn_dae.trainers.trainer_nonstiff_linear_lpnet import TrainerNonstiffLinearLPNet

    t0 = torch.tensor([[0.0]], dtype=torch.float64)

    y0 = y_exact_ex1(t0)

    tau_train = torch.linspace(
        0.0,
        1.0,
        100,
        dtype=torch.float64,
    ).reshape(-1, 1)

    _, nodes = get_collocation(num_nodes)

    histories = {}
    for degree in degree_list:
        print(f"\nTraining for degree = {degree}")

        model = NonstiffLinearLPNet(
            y0=y0,
            t0=t0,
            degree=degree,
            fix_a1=False,
        ).double()

        trainer = TrainerNonstiffLinearLPNet(model=model)

        train_diagnostics = trainer.train_model(tau_train=tau_train)

        _, diagnostics = trainer.compute_loss(tau_train=tau_train, model=model)

        final_diagnostics = {
            key: value.item() for key, value in diagnostics.items()
        }

        # model.print_coefficients()

        # The model is evaluated at normalized collocation nodes.
        t_eval = np.concatenate(
            (
                np.asarray(t0, dtype=np.float64).reshape(-1),
                np.asarray(nodes, dtype=np.float64).reshape(-1),
            )
        )
        t_eval_tensor = torch.as_tensor(t_eval, dtype=torch.float64).reshape(-1, 1)

        y_pred_eval = model.evaluate_y(tau_train)#model.evaluate_y(t_eval_tensor)

        # Exact solution is evaluated at the corresponding physical times.
        y_ex_eval = y_exact_ex1(tau_train)
        y_ex_eval = y_ex_eval.detach()

        assert y_ex_eval.shape == y_pred_eval.shape

        error_y = torch.max(torch.abs(y_pred_eval - y_ex_eval)).item()

        histories[degree] = {
            "train": train_diagnostics,
            "final": final_diagnostics,
            "error_y": error_y,
            "error_z": None,
            "t_eval": t_eval,
            "t_train": tau_train,
            "y_pred": y_pred_eval,
            "z_pred": None,
        }

    return histories


def y_exact(t, lambda_d=-2.0):
    return np.exp(2 * lambda_d * t)


def z_exact(t, lambda_d=-2.0, lambda_a=1.0):
    return lambda_d / lambda_a * np.exp(2 * lambda_d * t)


def y_exact_ex1(t):
    if isinstance(t, torch.Tensor):
        return torch.exp(-t)

    return np.exp(-np.asarray(t))


def plot_linear_dae_solution_across_degree(problem_name, degree_list, histories):
    t0 = 0.0

    t_eval = histories[degree_list[0]]["t_eval"]

    y_ex = y_exact(t_eval)
    z_ex = z_exact(t_eval)

    linestyles = ["solid", "dashdot", "dotted", "solid", "dashdot", "dotted", "solid", "dashdot", "dotted", "solid"]
    markers = ["o", "s", "^", "h", "*", "H", "<", ">", "d", "X"]

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    plt.rcParams["axes.linewidth"] = 0.45
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    for d, degree in enumerate(degree_list):
        history = histories[degree]

        y_pred = history["y_pred"].detach()
        z_pred = history["z_pred"].detach()

        axs[0].plot(t_eval, y_pred, linestyle=linestyles[d], marker=markers[d], label=rf"$p =${degree}")
        axs[1].plot(t_eval, z_pred, linestyle=linestyles[d], marker=markers[d], label=rf"$p =${degree}")

    axs[0].plot(t_eval, y_ex, color="black", linestyle="dashed", label=r"analytical")
    axs[1].plot(t_eval, z_ex, color="black", linestyle="dashed", label=r"analytical")

    for ax in axs:
        ax.tick_params(axis="both", which="major", width=0.4)

        ax.grid(which="major", axis="both", linewidth=0.35, alpha=0.3)

        ax.set_xlabel(r"time $t$")

        ax.set_xlim((t0 - 0.025, max(t_eval) + 0.025))

    axs[0].set_ylabel(r"$y(t)$")
    axs[1].set_ylabel(r"$z(t)$")
    
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=5)

    plot_name = f"learned_series_different_degrees"
    save_fig(plt, plot_name, problem_name)


def plot_nonstiff_linear_ode_solution_across_degree(problem_name, degree_list, histories):
    t0 = 0.0

    t_train = histories[degree_list[0]]["t_train"]

    y_ex = y_exact_ex1(t_train)

    linestyles = ["solid", "dashdot", "dotted", "solid", "dashdot", "dotted", "solid", "dashdot", "dotted", "solid"]
    markers = ["o", "s", "^", "h", "*", "H", "<", ">", "d", "X"]
    colors = ["royalblue", "orangered", "gold", "forestgreen"]

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    plt.rcParams["axes.linewidth"] = 0.45
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for d, degree in enumerate(degree_list):
        history = histories[degree]

        y_pred = history["y_pred"].detach()

        ax.plot(
            t_train,
            y_pred,
            linestyle=linestyles[d],
            marker=markers[d],
            color=colors[d],
            markersize=1.5,
            markeredgecolor=colors[d],
            label=rf"$p =${degree}",
        )

    ax.plot(t_train, y_ex, color="black", linestyle="dashed", label=r"analytical")

    ax.tick_params(axis="both", which="major", width=0.4)

    ax.grid(which="major", axis="both", linewidth=0.35, alpha=0.3)

    ax.set_xlabel(r"time $t$")

    ax.set_xlim((t0 - 0.025, max(t_train) + 0.025))

    ax.set_ylabel(r"$y(t)$")
    
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=5)

    plot_name = f"learned_series_different_degrees"
    save_fig(plt, plot_name, problem_name)


def plot_error_across_time(problem_name, degree_list, histories):
    t0 = 0.0

    t_train = histories[degree_list[0]]["t_train"]

    y_ex = y_exact_ex1(t_train).reshape(-1, 1)

    linestyles = ["dotted", "dotted", "dotted", "solid", "dashdot", "dotted", "solid", "dashdot", "dotted", "solid"]
    markers = ["+", "X", "o", "h", "*", "H", "<", ">", "d", "X"]
    colors = ["royalblue", "orangered", "gold", "forestgreen"]

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    plt.rcParams["axes.linewidth"] = 0.45
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for d, degree in enumerate(degree_list):
        history = histories[degree]

        y_pred = history["y_pred"].detach()
        err = torch.abs(y_pred - y_ex)

        ax.plot(
            t_train,
            err,
            linestyle=linestyles[d],
            marker=markers[d],
            color=colors[d],
            markersize=1.5,
            markeredgecolor=colors[d],
            label=rf"$p =${degree}",
        )

    ax.tick_params(axis="both", which="major", width=0.4)

    ax.grid(which="major", axis="both", linewidth=0.35, alpha=0.3)

    ax.set_xlabel(r"time $t$")

    ax.set_xlim((t0, max(t_train)))

    ax.set_yscale("log", base=10)

    ax.set_ylabel(r"$y(t)$")
    
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=5)

    plot_name = f"error_across_time"
    save_fig(plt, plot_name, problem_name)


def plot_loss_across_degree(problem_name, degree_list, histories):
    loss = [histories[d]["final"]["loss"] for d in degree_list]
    loss_diff = [histories[d]["final"]["loss_differential"] for d in degree_list]
    loss_alg = [histories[d]["final"]["loss_algebraic"] for d in degree_list]

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    plt.rcParams["axes.linewidth"] = 0.45
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(degree_list, loss, linestyle="solid", marker="X", label="total loss")
    ax.plot(degree_list, loss_diff, linestyle="solid", marker="X", label="loss differential equation")
    ax.plot(degree_list, loss_alg, linestyle="solid", marker="X", label="loss algebraic equation")

    ax.tick_params(axis="both", which="major", width=0.4)
    ax.tick_params(axis="both", which="minor", width=0.4)

    ax.grid(which="major", axis="both", linewidth=0.35, alpha=0.3)
    ax.grid(which="major", axis="both", linewidth=0.15, alpha=0.15)

    ax.set_xlabel(r"degree $p$")

    ax.set_xticks(degree_list)
    ax.set_xticklabels(degree_list)

    ax.set_xlim((min(degree_list) - 0.07, max(degree_list) + 0.07))

    ax.set_ylabel("loss")

    ax.set_yscale("log", base=10)

    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=3)

    plot_name = f"loss_across_degrees"
    save_fig(plt, plot_name, problem_name)


def plot_loss_across_epochs(problem_name, degree_list, histories):
    linestyles = ["solid", "dashdot", "dotted", "solid", "dashdot", "dotted", "solid", "dashdot", "dotted", "solid"]
    markers = ["+", "X", "o", "h", "*", "H", "<", ">", "d", "X"]
    colors = ["royalblue", "orangered", "gold", "forestgreen"]

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    plt.rcParams["axes.linewidth"] = 0.45
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for d, degree in enumerate(degree_list):
        adam_history = histories[degree]["train"]

        steps = [entry["epoch"] for entry in adam_history]
        losses = [entry["loss"] for entry in adam_history]

        ax.plot(
            steps,
            losses,
            linestyle=linestyles[d],
            color=colors[d],
            label=rf"$p =${degree}",
        )

    ax.tick_params(axis="both", which="major", width=0.4)
    ax.tick_params(axis="both", which="minor", width=0.4)

    ax.grid(which="major", axis="both", linewidth=0.35, alpha=0.3)

    ax.set_xlabel(r"epoch")

    ax.set_xlim((1.0, max(steps)))

    ax.set_yscale("log", base=10)

    ax.set_ylabel("loss")
    
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=5)

    plot_name = f"epochs_versus_loss_different_degrees"
    save_fig(plt, plot_name, problem_name)


def make_plots_linear_dae():
    problem_name = "LINEAR-TEST"
    num_nodes = 5

    degree_list = [1, 2, 3, 4]#[3, 4, 5, 6, 7, 8, 9, 10]
    dt_list, _ = choose_time_step_sizes(problem_name)

    # Study across different degrees of MacLaurin series
    histories = study_linear_dae_across_degree(problem_name, degree_list, num_nodes)
    plot_linear_dae_solution_across_degree(problem_name, degree_list, histories)
    plot_loss_across_degree(problem_name, degree_list, histories)
    plot_loss_across_epochs(problem_name, degree_list, histories)


def make_plots_nonstiff_linear_ode():
    problem_name = "NONSTIFF-LINEAR-EX1"
    num_nodes = 5

    degree_list = [1, 2, 3]#[3, 4, 5, 6, 7, 8, 9, 10]
    dt_list, _ = choose_time_step_sizes(problem_name)

    # Study across different degrees of MacLaurin series
    histories = study_nonstiff_linear_ode_across_degree(problem_name, degree_list, num_nodes)
    plot_nonstiff_linear_ode_solution_across_degree(problem_name, degree_list, histories)
    plot_loss_across_epochs(problem_name, degree_list, histories)
    plot_error_across_time(problem_name, degree_list, histories)


if __name__ == "__main__":
    make_plots_linear_dae()
    # make_plots_nonstiff_linear_ode()