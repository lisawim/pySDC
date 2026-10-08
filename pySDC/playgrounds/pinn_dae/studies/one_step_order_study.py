from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.playgrounds.pinn_dae.utils import my_setup_mpl
from pySDC.playgrounds.pinn_dae.models.linear_dae_lpnet import LinearDAELPNet
from pySDC.playgrounds.pinn_dae.trainers.trainer_linear_dae_lpnet import TrainerLinearDAELPNet
from pySDC.playgrounds.pinn_dae.experiments.sdc_provisional_solution import (
    compute_constant_reference_order,
    choose_time_step_sizes,
    save_fig,
    sync_ylim,
    u_exact,
    get_collocation,
)

Tensor = torch.Tensor


def study_across_dt(degree) -> None:
    problem_name = "LINEAR-TEST"
    num_nodes = 5

    t0 = torch.as_tensor(0.0, dtype=torch.float64)
    dt_list, Tend = choose_time_step_sizes(problem_name)

    y0, z0 = u_exact(t0)
    errors_y0: list[float] = []
    errors_z0: list[float] = []

    histories: dict[float, dict[str, object]] = {}

    _, nodes = get_collocation(num_nodes)
    tau_nodes = torch.as_tensor(nodes, dtype=torch.float64).reshape(-1, 1)

    degree_y = degree
    degree_z = degree

    for dt_value in dt_list:
        print(f"\n .. Training for dt = {dt_value} .. \n")

        dt = torch.as_tensor(dt_value, dtype=torch.float64)

        tau_train = torch.linspace(
            t0,
            t0 + dt,
            2000,
            dtype=torch.float64,
        ).reshape(-1, 1)

        model = LinearDAELPNet(
            y0=y0,
            z0=z0,
            t0=t0,
            degree_y=degree_y,
            degree_z=degree_z,
            fix_a1=False,
        ).double()

        trainer = TrainerLinearDAELPNet(model=model)

        train_diagnostics = trainer.train_model(tau_train=tau_train, num_epochs=200000)

        refine_diagnostics = trainer.refine(tau_train=tau_train)

        _, diagnostics = trainer.compute_loss(tau_train=tau_train, model=model)

        final_diagnostics = {
            key: value.item() for key, value in diagnostics.items()
        }

        # The model is evaluated at normalized collocation nodes.
        t_eval = t0 + dt * tau_nodes
        y_pred_eval = model.evaluate_y(t_eval)
        z_pred_eval = model.evaluate_z(t_eval)

        y_ex_eval, z_ex_eval = u_exact(t_eval)
        y_ex_eval = y_ex_eval.detach()
        z_ex_eval = z_ex_eval.detach()

        assert y_ex_eval.shape == y_pred_eval.shape
        assert z_ex_eval.shape == z_pred_eval.shape

        errors_y0.append(torch.max(torch.abs(y_pred_eval - y_ex_eval)).item())
        errors_z0.append(torch.max(torch.abs(z_pred_eval - z_ex_eval)).item())

        histories[dt_value] = {
            "train": train_diagnostics,
            "refine": refine_diagnostics,
            "final": final_diagnostics,
            "error_y": errors_y0,
            "error_z": errors_z0,
            "t_eval": t_eval,
            "y_pred": y_pred_eval,
            "z_pred": z_pred_eval,
        }

    plot_local_order(degree_y, degree_z, dt_list, errors_y0, errors_z0)

    plot_epochs_versus_loss(degree_y, degree_z, histories=histories)

    plot_epochs_versus_loss_to_refine(degree_y, degree_z, histories)

    return histories


def plot_local_order(degree_y, degree_z, dt_list, errors_y0, errors_z0):
    dt_list_short = dt_list[1:5]

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    plt.rcParams["axes.linewidth"] = 0.45
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].loglog(
        dt_list,
        errors_y0,
        linewidth=0.7,
        color="blue",
        marker="o",
        linestyle="solid",
        label=r"$y$",
    )
    axs[1].loglog(
        dt_list,
        errors_z0,
        linewidth=0.7,
        color="blue",
        marker="o",
        linestyle="solid",
        label=r"$z$",
    )

    # A degree-p Taylor polynomial has endpoint/local approximation error
    # O(dt^(p+1)) when its coefficients are sufficiently accurate.
    p_y = degree_y + 1
    p_z = degree_z + 1
    Cy = compute_constant_reference_order(dt_list, errors_y0, k=0)
    Cz = compute_constant_reference_order(dt_list, errors_z0, k=0)

    ref_y = [Cy * dt**p_y for dt in dt_list_short]
    ref_z = [Cz * dt**p_z for dt in dt_list_short]

    axs[0].loglog(
        dt_list_short,
        ref_y,
        linewidth=0.7,
        color="black",
        linestyle="dashed",
    )
    axs[1].loglog(
        dt_list_short,
        ref_z,
        linewidth=0.7,
        color="black",
        linestyle="dashed",
    )

    axs[0].text(
        dt_list_short[-1] * 0.8,
        ref_y[-1] * 0.7,
        rf"${p_y}$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )
    axs[1].text(
        dt_list_short[-1] * 0.8,
        ref_z[-1] * 0.7,
        rf"${p_z}$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    for ax in axs:
        ax.tick_params(axis="both", which="major", width=0.4)
        ax.tick_params(axis="both", which="minor", width=0.4)

        ax.set_xlabel(r"time step size $\Delta t$")

        ax.grid(linewidth=0.5)

    axs[0].set_ylabel(r"$\max_m |y(\tau_m)-\widehat y(\tau_m)|$")
    axs[1].set_ylabel(r"$\max_m |z(\tau_m)-\widehat z(\tau_m)|$")

    axs = sync_ylim(axs, min_y_set=1e-15)

    plot_name = f"local_order"
    if degree_y == degree_z:
        plot_name += f"_degree={degree_y}"
    else:
        plot_name += f"_{degree_y=}_{degree_z=}"
    save_fig(plt, plot_name, problem_name)


def plot_epochs_versus_loss(degree_y, degree_z, histories):
    problem_name = "LINEAR-TEST"

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    plt.rcParams["axes.linewidth"] = 0.45
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    dt_list = list(histories.keys())

    num_lines = len(dt_list)

    cmap = plt.colormaps["viridis"]

    colors = cmap(np.linspace(0.1, 0.9, num_lines))

    for d, dt_value in enumerate(dt_list):
        history = histories[dt_value]["train"]

        epochs = [entry["epoch"] for entry in history]
        losses = [entry["loss"] for entry in history]

        ax.plot(epochs, losses, color=colors[d], label=rf"$\Delta t =${dt_value}")

    ax.tick_params(axis="both", which="major", width=0.4)
    ax.tick_params(axis="both", which="minor", width=0.4)

    ax.set_yscale("log", base=10)

    ax.set_xlim((min(epochs), max(epochs)))

    ax.set_xlabel(f"number of epochs")
    ax.set_ylabel("loss")

    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=4)

    plot_name = f"epochs_versus_loss_time_step_sizes"
    if degree_y == degree_z:
        plot_name += f"_degree={degree_y}"
    else:
        plot_name += f"_{degree_y=}_{degree_z=}"

    save_fig(plt, plot_name, problem_name)


def plot_epochs_versus_loss_to_refine(degree_y, degree_z, histories):
    problem_name = "LINEAR-TEST"

    figsize = figsize_by_journal(
        journal="BUW_thesis",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
    plt.rcParams["axes.linewidth"] = 0.45
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    dt_list = list(histories.keys())

    num_lines = len(dt_list)

    cmap = plt.colormaps["viridis"]

    colors = cmap(np.linspace(0.1, 0.9, num_lines))

    for d, dt_value in enumerate(dt_list):
        history = histories[dt_value]["refine"]

        evals = [entry["evaluation"] for entry in history]
        losses = [entry["loss"] for entry in history]

        ax.plot(evals, losses, color=colors[d], label=rf"$\Delta t =${dt_value}")

    ax.tick_params(axis="both", which="major", width=0.4)
    ax.tick_params(axis="both", which="minor", width=0.4)

    ax.set_yscale("log", base=10)

    ax.set_xlim((min(evals), max(evals)))

    ax.set_xlabel(f"number of evaluations")
    ax.set_ylabel("loss")

    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 0.04), ncol=4)

    plot_name = "evals_versus_loss_refine_time_step_sizes"
    if degree_y == degree_z:
        plot_name += f"_degree={degree_y}"
    else:
        plot_name += f"_{degree_y=}_{degree_z=}"

    save_fig(plt, plot_name, problem_name)


if __name__ == "__main__":
    problem_name = "LINEAR-TEST"
    num_nodes = 5

    degree_list = [1, 2, 3, 4]
    dt_list, Tend = choose_time_step_sizes(problem_name)

    # Study across different time step sizes
    for degree in degree_list:
        histories = study_across_dt(degree=degree)