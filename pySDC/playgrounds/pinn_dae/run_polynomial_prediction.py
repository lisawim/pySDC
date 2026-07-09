import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.playgrounds.pinn_dae.utils import my_setup_mpl
from pySDC.playgrounds.pinn_dae.models.polynomial_predictor import LPNet, MaclaurinCoefficientNet, evaluate_series, evaluate_series_vectorized
from pySDC.playgrounds.pinn_dae.run_prediction_provisional_solution import (
    compute_constant_reference_order,
    choose_time_step_sizes,
    save_fig,
    sync_ylim,
    u_exact,
    get_collocation,
    get_Qdelta,
)


def plot_order_polynomial_prediction():
    problem_name = "LINEAR-TEST"

    # Define the number of collocation nodes
    num_nodes = 5
    degree = 5

    t0 = 0.0
    t0 = torch.tensor([[t0]], dtype=torch.float64)
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[1 : 5]

    y0, z0 = u_exact(t0)

    errors_y0, errors_z0 = [], []
    for dt in dt_list:
        print(f"Training for dt = {dt}")

        dt = torch.tensor([[dt]], dtype=torch.float64)

        # Unabhängiges Auswertungsgitter
        tau_sdc = torch.linspace(
            t0.item(),
            (t0 + dt).item(),
            100000,
            dtype=torch.float64,
        ).reshape(-1, 1)
        tau_sdc.requires_grad_(True)

        # Create an instance of the PredictorNet model
        diff_model = LPNet(M=num_nodes, u0=y0, degree=degree, diff=True).double()
        alg_model = LPNet(M=num_nodes, u0=z0, degree=degree, diff=False).double()

        y_ex, z_ex = u_exact(tau_sdc)
        y_ex = y_ex.detach()
        z_ex = z_ex.detach()

        y_pred = diff_model.train_model(
            tau_nodes=tau_sdc,
            y_ex=y_ex,
            z_ex=z_ex,
        )

        z_pred = alg_model.train_model(
            tau_nodes=tau_sdc,
            y_ex=y_ex,
            z_ex=z_ex,
        )

        _, nodes = get_collocation(num_nodes)
        tau_nodes = [t0 + dt * coll_node for coll_node in nodes]

        tau_eval = torch.tensor(
            tau_nodes,
            dtype=torch.float64,
        ).reshape(-1, 1)
        y_ex_eval, z_ex_eval = u_exact(tau_eval)
        y_ex_eval = y_ex_eval.detach()
        z_ex_eval = z_ex_eval.detach()

        y_pred_eval = diff_model(tau_eval).detach()
        z_pred_eval = alg_model(tau_eval).detach()

        errors_y0.append(
            torch.max(torch.abs(y_pred_eval - y_ex_eval)).item()
        )
        errors_z0.append(
            torch.max(torch.abs(z_pred_eval - z_ex_eval)).item()
        )

        # errors_y0.append(torch.max(torch.abs(y_pred - y_ex)).detach().cpu().item())
        # errors_z0.append(torch.max(torch.abs(z_pred - z_ex)).detach().cpu().item())

    figsize = figsize_by_journal(journal="Springer_proceedings", scale=0.7, ratio=0.5)
    my_setup_mpl(fontsize=5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].loglog(
        dt_list,
        errors_y0,
        linewidth=0.7,
        color="blue",
        marker="o",
        linestyle="solid",
        label=f"y_0",
    )

    axs[1].loglog(
        dt_list,
        errors_z0,
        linewidth=0.7,
        color="blue",
        marker="o",
        linestyle="solid",
        label=f"z_0",
    )

    Cy = compute_constant_reference_order(dt_list, errors_y0, k=0)
    Cz = compute_constant_reference_order(dt_list, errors_z0, k=0)

    # Reference order
    ref_y = [Cy * dt for dt in dt_list_short]
    axs[0].loglog(
        dt_list_short,
        ref_y,
        linewidth=0.7,
        color="black",
        linestyle="dashed",
    )

    axs[0].text(
        dt_list_short[-1] * 0.8,
        ref_y[-1] * 0.7,
        r"$1$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    ref_z = [Cz * dt for dt in dt_list_short]
    axs[1].loglog(
        dt_list_short,
        ref_z,
        linewidth=0.7,
        color="black",
        linestyle="dashed",
    )

    axs[1].text(
        dt_list_short[-1] * 0.8,
        ref_z[-1] * 0.7,
        r"$1$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)

        ax.set_xlabel(r"time step size $\Delta t$")

        ax.grid(linewidth=0.5)

    axs[0].set_ylabel(r"$\max_{m\in\{1,..,M\}}|y(\tau_m) - y^0_{m,t_1}|$")
    axs[1].set_ylabel(r"$\max_{m\in\{1,..,M\}}|z(\tau_m) - z^0_{m,t_1}|$")

    axs = sync_ylim(axs, min_y_set=1e-15)

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    plot_name = f"order_polynomial_prediction_{num_nodes=}"
    save_fig(plt, plot_name, problem_name)


def predict_coeffs():
    problem_name = "LINEAR-TEST"

    degree = 5
    num_nodes = 4

    _, nodes = get_collocation(num_nodes)

    t0 = 0.0
    t0 = torch.tensor([[t0]], dtype=torch.float64)
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[1 : 5]

    y0, z0 = u_exact(t0)

    model_y = MaclaurinCoefficientNet(u0=y0, is_diff=True, degree=degree).double()
    model_z = MaclaurinCoefficientNet(u0=z0, is_diff=False, degree=degree).double()

    coeffs_y_learned = model_y.train_coefficient_net(
        model=model_y,
    )
    coeffs_z_learned = model_z.train_coefficient_net(
        model=model_z,
    )

    coeffs_y_exact = model_y.exact_coefficients()
    coeffs_z_exact = model_z.exact_coefficients()

    errors_y0, errors_z0 = [], []
    for dt in dt_list:
        print(f"Training for dt = {dt}")

        dt = torch.tensor([[dt]], dtype=torch.float64)

        tau_nodes = [t0 + dt * coll_node for coll_node in nodes]

        tau_eval = torch.tensor(
            tau_nodes,
            dtype=torch.float64,
        ).reshape(-1, 1)

        y_eval = evaluate_series_vectorized(
            tau_eval,
            coeffs_y_learned,
        )
        z_eval = evaluate_series_vectorized(
            tau_eval,
            coeffs_z_learned,
        )

        y_ex, z_ex = u_exact(tau_eval)
        y_ex = y_ex.detach()
        z_ex = z_ex.detach()

        errors_y0.append(
            torch.max(torch.abs(y_eval - y_ex)).item()
        )
        errors_z0.append(
            torch.max(torch.abs(z_eval - z_ex)).item()
        )

    figsize = figsize_by_journal(journal="Springer_proceedings", scale=0.7, ratio=0.5)
    my_setup_mpl(fontsize=5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].loglog(
        dt_list,
        errors_y0,
        linewidth=0.7,
        color="blue",
        marker="o",
        linestyle="solid",
        label=f"y_0",
    )

    axs[1].loglog(
        dt_list,
        errors_z0,
        linewidth=0.7,
        color="blue",
        marker="o",
        linestyle="solid",
        label=f"z_0",
    )

    Cy = compute_constant_reference_order(dt_list, errors_y0, k=0)
    Cz = compute_constant_reference_order(dt_list, errors_z0, k=0)

    # Reference order
    ref_y = [Cy * dt ** (degree + 1) for dt in dt_list_short]
    axs[0].loglog(
        dt_list_short,
        ref_y,
        linewidth=0.7,
        color="black",
        linestyle="dashed",
    )

    axs[0].text(
        dt_list_short[-1] * 0.8,
        ref_y[-1] * 0.7,
        r"$1$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    ref_z = [Cz * dt ** (degree + 1) for dt in dt_list_short]
    axs[1].loglog(
        dt_list_short,
        ref_z,
        linewidth=0.7,
        color="black",
        linestyle="dashed",
    )

    axs[1].text(
        dt_list_short[-1] * 0.8,
        ref_z[-1] * 0.7,
        r"$1$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)

        ax.set_xlabel(r"time step size $\Delta t$")

        ax.grid(linewidth=0.5)

    axs[0].set_ylabel(r"$\max_{m\in\{1,..,M\}}|y(\tau_m) - y^0_{m,t_1}|$")
    axs[1].set_ylabel(r"$\max_{m\in\{1,..,M\}}|z(\tau_m) - z^0_{m,t_1}|$")

    axs = sync_ylim(axs, min_y_set=1e-15)

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    plot_name = f"order_polynomial_prediction_coeffs_{num_nodes=}"
    save_fig(plt, plot_name, problem_name)


if __name__ == "__main__":
    # plot_order_polynomial_prediction()
    predict_coeffs()