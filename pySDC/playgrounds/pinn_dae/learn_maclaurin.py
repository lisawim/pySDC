from __future__ import annotations

from typing import Callable

import math
import torch
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.playgrounds.pinn_dae.utils import my_setup_mpl
from pySDC.playgrounds.pinn_dae.models.dae_lp_net import DAELPNet
from pySDC.playgrounds.pinn_dae.run_prediction_provisional_solution import (
    compute_constant_reference_order,
    choose_time_step_sizes,
    save_fig,
    sync_ylim,
    u_exact,
    get_collocation,
    get_Qdelta,
)


Tensor = torch.Tensor


def dae_lpnet_loss(
    model: DAELPNet,
    t_train: Tensor,
    f: Callable[[Tensor, Tensor, Tensor], Tensor],
    g: Callable[[Tensor, Tensor, Tensor], Tensor],
    weight_differential: float = 1.0,
    weight_algebraic: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Compute the physics-informed LPNet loss for a semi-explicit DAE.

    Parameters
    ----------
    model:
        Polynomial DAE-LPNet.
    tau:
        Training points in [0, 1], shape (N, 1).
    t0:
        Beginning of the current time interval.
    dt:
        Length of the current time interval.
    f:
        Differential right-hand side f(y, z, t).
    g:
        Algebraic constraint g(y, z, t).
    """
    t_train = t_train.reshape(-1, 1)

    y_hat, z_hat = model(t_train)

    dy_dt = model.evaluate_dy_dtau(t_train)

    assert y_hat.shape == t_train.shape
    assert z_hat.shape == t_train.shape
    assert dy_dt.shape == t_train.shape

    differential_residual = dy_dt - f(y_hat, z_hat, t_train)
    algebraic_residual = g(y_hat, z_hat, t_train)

    loss_differential = torch.mean(differential_residual.square())
    loss_algebraic = torch.mean(algebraic_residual.square())

    loss = (
        weight_differential * loss_differential
        + weight_algebraic * loss_algebraic
    )

    diagnostics = {
        "loss": loss.detach(),
        "loss_differential": loss_differential.detach(),
        "loss_algebraic": loss_algebraic.detach(),
        "max_differential_residual": (
            differential_residual.detach().abs().max()
        ),
        "max_algebraic_residual": (
            algebraic_residual.detach().abs().max()
        ),
    }

    return loss, diagnostics


def train_dae_lpnet(
    model: DAELPNet,
    t_train: Tensor,
    f: Callable[[Tensor, Tensor, Tensor], Tensor],
    g: Callable[[Tensor, Tensor, Tensor], Tensor],
    num_adam_steps: int = 5000,
    learning_rate: float = 1.0e-3,
    weight_algebraic: float = 1.0,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    history: list[dict[str, float]] = []

    for step in range(num_adam_steps):
        optimizer.zero_grad()

        loss, diagnostics = dae_lpnet_loss(
            model=model,
            t_train=t_train,
            f=f,
            g=g,
            weight_algebraic=weight_algebraic,
        )

        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == num_adam_steps - 1:
            # print(step, loss)
            history.append(
                {
                    "step": float(step),
                    **{
                        key: value.item()
                        for key, value in diagnostics.items()
                    },
                }
            )

    with torch.no_grad():
        # a1 = model.coeffs_y.reshape(-1)[0]

        # y0_batch = model.y0.reshape(1, 1)
        # z0_batch = model.z0.reshape(1, 1)
        # t_zero = torch.zeros(
        #     (1, 1),
        #     dtype=model.y0.dtype,
        #     device=model.y0.device,
        # )

        # a1_exact = f(y0_batch, z0_batch, t_zero).reshape(-1)[0]

        # print("a1 learned:", a1.item())
        # print("a1 exact:  ", a1_exact.item())
        # print("a1 error:  ", abs(a1 - a1_exact).item())

        coeffs_y = model.get_coeffs_y().detach().reshape(-1)

        for j, coeff in enumerate(coeffs_y, start=1):
            exact = (-4.0) ** j / math.factorial(j)

            print(
                f"a{j}: learned={coeff.item():.16e}, "
                f"exact={exact:.16e}, "
                f"error={abs(coeff.item() - exact):.6e}"
            )

        coeffs_z = model.get_coeffs_z().detach().reshape(-1)

        for j, coeff in enumerate(coeffs_z, start=1):
            exact = -2.0 * (-4.0) ** j / math.factorial(j)

            print(
                f"b{j}: learned={coeff.item():.16e}, "
                f"exact={exact:.16e}, "
                f"error={abs(coeff.item() - exact):.6e}"
            )

    return history


def initialize_first_coefficient(
    model: DAELPNet,
    f: Callable[[Tensor, Tensor, Tensor], Tensor],
    t0: float,
    dt: float,
) -> None:
    with torch.no_grad():
        y0_batch = model.y0.unsqueeze(0)
        z0_batch = model.z0.unsqueeze(0)

        t0_tensor = torch.tensor(
            [[t0]],
            dtype=model.y0.dtype,
            device=model.y0.device,
        )

        model.coeffs_y[0].copy_(
            dt * f(y0_batch, z0_batch, t0_tensor).squeeze(0)
        )


def f(y: Tensor, z: Tensor, t: Tensor, lambda_d=-2.0, lambda_a=1.0) -> Tensor:
    return lambda_d * y + lambda_a * z


def g(y: Tensor, z: Tensor, t: Tensor, lambda_d=-2.0, lambda_a=1.0) -> Tensor:
    return lambda_d * y - lambda_a * z


def main():
    problem_name = "LINEAR-TEST"

    # Define the number of collocation nodes
    num_nodes = 5
    degree = 2

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
        t_train = torch.linspace(
            t0.item(),
            (t0 + dt).item(),
            int(100),
            dtype=torch.float64,
        ).reshape(-1, 1)
        t_train.requires_grad_(True)

        dae_lpnet = DAELPNet(
            y0=y0,
            z0=z0,
            # a1=torch.tensor([[-4.0]], dtype=torch.float64),
            degree_y=degree,
            degree_z=degree,
        ).double()

        history = train_dae_lpnet(
            model=dae_lpnet,
            t_train=t_train,
            f=f,
            g=g,
            num_adam_steps=20000,
            learning_rate=1.0e-3,
        )

        for key, value in history[-1].items():
            print(f"{key}: {value:.6e}")

        _, nodes = get_collocation(num_nodes)
        t_nodes = [t0 + dt * coll_node for coll_node in nodes]

        t_eval = torch.tensor(
            t_nodes,
            dtype=torch.float64,
        ).reshape(-1, 1)
        y_ex_eval, z_ex_eval = u_exact(t_eval)

        y_ex_eval = y_ex_eval.detach()
        z_ex_eval = z_ex_eval.detach()

        y_pred_eval = dae_lpnet.evaluate_y(t_eval)
        z_pred_eval = dae_lpnet.evaluate_z(t_eval)

        assert y_ex_eval.shape == y_pred_eval.shape
        assert z_ex_eval.shape == z_pred_eval.shape

        errors_y0.append(
            torch.max(torch.abs(y_pred_eval - y_ex_eval)).item()
        )
        errors_z0.append(
            torch.max(torch.abs(z_pred_eval - z_ex_eval)).item()
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
    expected_order = degree + 1
    ref_y = [Cy * dt ** expected_order for dt in dt_list_short]
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
        r"$2$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    ref_z = [Cz * dt ** expected_order for dt in dt_list_short]
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
        r"$2$",
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

    plot_name = f"order_learn_maclaurin_{num_nodes=}"
    save_fig(plt, plot_name, problem_name)


if __name__ == "__main__":
    main()