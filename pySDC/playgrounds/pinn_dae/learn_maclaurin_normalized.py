from __future__ import annotations

from typing import Callable

import math
import torch
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.playgrounds.pinn_dae.utils import my_setup_mpl
from pySDC.playgrounds.pinn_dae.models.dae_lp_net_normalized import DAELPNet
from pySDC.playgrounds.pinn_dae.run_prediction_provisional_solution import (
    compute_constant_reference_order,
    choose_time_step_sizes,
    save_fig,
    sync_ylim,
    u_exact,
    get_collocation,
)

Tensor = torch.Tensor


def f(
    y: Tensor,
    z: Tensor,
    t: Tensor,
    lambda_d: float = -2.0,
    lambda_a: float = 1.0,
) -> Tensor:
    return lambda_d * y + lambda_a * z


def g(
    y: Tensor,
    z: Tensor,
    t: Tensor,
    lambda_d: float = -2.0,
    lambda_a: float = 1.0,
) -> Tensor:
    return lambda_d * y - lambda_a * z


def dae_lpnet_loss(
    model: DAELPNet,
    xi_train: Tensor,
    t0: Tensor,
    dt: Tensor,
    f_rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    g_rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    weight_differential: float = 1.0,
    weight_algebraic: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Physics-informed loss in normalized time.

    With ``t = t0 + dt*xi`` and ``xi in [0, 1]``, the DAE becomes

        dy/dxi - dt*f(y, z, t) = 0,
        g(y, z, t)             = 0.
    """
    xi_train = xi_train.reshape(-1, 1)
    t0 = torch.as_tensor(t0, dtype=xi_train.dtype, device=xi_train.device)
    dt = torch.as_tensor(dt, dtype=xi_train.dtype, device=xi_train.device)

    t_train = t0 + dt * xi_train
    y_hat, z_hat = model(xi_train)
    dy_dxi = model.evaluate_dy_dxi(xi_train)

    assert y_hat.shape == xi_train.shape
    assert z_hat.shape == xi_train.shape
    assert dy_dxi.shape == xi_train.shape
    assert t_train.shape == xi_train.shape

    differential_residual = dy_dxi - dt * f_rhs(y_hat, z_hat, t_train)
    algebraic_residual = g_rhs(y_hat, z_hat, t_train)

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
        "max_differential_residual": differential_residual.detach().abs().max(),
        "max_algebraic_residual": algebraic_residual.detach().abs().max(),
    }
    return loss, diagnostics


def train_with_adam(
    model: DAELPNet,
    xi_train: Tensor,
    t0: Tensor,
    dt: Tensor,
    f_rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    g_rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    num_steps: int = 10_000,
    learning_rate: float = 1.0e-3,
    weight_algebraic: float = 1.0,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for step in range(num_steps):
        optimizer.zero_grad()
        loss, diagnostics = dae_lpnet_loss(
            model=model,
            xi_train=xi_train,
            t0=t0,
            dt=dt,
            f_rhs=f_rhs,
            g_rhs=g_rhs,
            weight_algebraic=weight_algebraic,
        )
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == num_steps - 1:
            history.append(
                {
                    "step": float(step),
                    **{key: value.item() for key, value in diagnostics.items()},
                }
            )

    return history


def refine_with_lbfgs(
    model: DAELPNet,
    xi_train: Tensor,
    t0: Tensor,
    dt: Tensor,
    f_rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    g_rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    max_iter: int = 1000,
    weight_algebraic: float = 1.0,
) -> None:
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1.0e-14,
        tolerance_change=1.0e-16,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss, _ = dae_lpnet_loss(
            model=model,
            xi_train=xi_train,
            t0=t0,
            dt=dt,
            f_rhs=f_rhs,
            g_rhs=g_rhs,
            weight_algebraic=weight_algebraic,
        )
        loss.backward()
        return loss

    optimizer.step(closure)


def print_scaled_coefficients(model: DAELPNet, dt: Tensor) -> None:
    """Compare learned coefficients in xi with exact scaled coefficients."""
    dt_value = float(dt.item())

    coeffs_y = model.full_coeffs_y().detach().reshape(-1)
    for j, coeff in enumerate(coeffs_y, start=1):
        exact = ((-4.0) ** j / math.factorial(j)) * dt_value**j
        print(
            f"alpha{j}: learned={coeff.item():.16e}, "
            f"exact={exact:.16e}, "
            f"error={abs(coeff.item() - exact):.6e}"
        )

    coeffs_z = model.full_coeffs_z().detach().reshape(-1)
    for j, coeff in enumerate(coeffs_z, start=1):
        exact = (-2.0 * (-4.0) ** j / math.factorial(j)) * dt_value**j
        print(
            f"beta{j}: learned={coeff.item():.16e}, "
            f"exact={exact:.16e}, "
            f"error={abs(coeff.item() - exact):.6e}"
        )


def main() -> None:
    problem_name = "LINEAR-TEST"
    num_nodes = 5
    degree = 2

    t0 = torch.tensor([[0.0]], dtype=torch.float64)
    dt_list, Tend = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[1:5]

    y0, z0 = u_exact(t0)
    errors_y0: list[float] = []
    errors_z0: list[float] = []

    # Normalized training grid. requires_grad is unnecessary because the
    # polynomial derivative is evaluated analytically.
    xi_train = torch.linspace(
        0.0,
        1.0,
        1000,
        dtype=torch.float64,
    ).reshape(-1, 1)

    _, nodes = get_collocation(num_nodes)
    xi_nodes = torch.as_tensor(nodes, dtype=torch.float64).reshape(-1, 1)

    for dt_value in dt_list:
        n_steps = int(Tend / dt_value)
        dt = torch.tensor([[n_steps * dt_value]], dtype=torch.float64)

        print(f"\nTraining for dt = {dt}")

        model = DAELPNet(
            y0=y0,
            z0=z0,
            degree_y=degree,
            degree_z=degree,
        ).double()

        history = train_with_adam(
            model=model,
            xi_train=xi_train,
            t0=t0,
            dt=dt,
            f_rhs=f,
            g_rhs=g,
            num_steps=20000,
            learning_rate=1.0e-3,
        )

        refine_with_lbfgs(
            model=model,
            xi_train=xi_train,
            t0=t0,
            dt=dt,
            f_rhs=f,
            g_rhs=g,
            max_iter=1000,
        )

        _, diagnostics = dae_lpnet_loss(
            model=model,
            xi_train=xi_train,
            t0=t0,
            dt=dt,
            f_rhs=f,
            g_rhs=g,
        )

        print_scaled_coefficients(model, dt)
        print(f"step: {history[-1]['step']:.6e}")
        for key, value in diagnostics.items():
            print(f"{key}: {value.item():.6e}")

        # The model is evaluated at normalized collocation nodes.
        t_eval = t0 + dt * xi_nodes
        y_pred_eval = model.evaluate_y(xi_nodes)
        z_pred_eval = model.evaluate_z(xi_nodes)

        # Exact solution is evaluated at the corresponding physical times.
        y_ex_eval, z_ex_eval = u_exact(t_eval)
        y_ex_eval = y_ex_eval.detach()
        z_ex_eval = z_ex_eval.detach()

        assert y_ex_eval.shape == y_pred_eval.shape
        assert z_ex_eval.shape == z_pred_eval.shape

        errors_y0.append(torch.max(torch.abs(y_pred_eval - y_ex_eval)).item())
        errors_z0.append(torch.max(torch.abs(z_pred_eval - z_ex_eval)).item())

    figsize = figsize_by_journal(
        journal="Springer_proceedings",
        scale=0.7,
        ratio=0.5,
    )
    my_setup_mpl(fontsize=5)
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
    expected_order = degree + 1
    Cy = compute_constant_reference_order(dt_list, errors_y0, k=0)
    Cz = compute_constant_reference_order(dt_list, errors_z0, k=0)

    ref_y = [Cy * dt**expected_order for dt in dt_list_short]
    ref_z = [Cz * dt**expected_order for dt in dt_list_short]

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
        rf"${expected_order}$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )
    axs[1].text(
        dt_list_short[-1] * 0.8,
        ref_z[-1] * 0.7,
        rf"${expected_order}$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_xlabel(r"time step size $\Delta t$")
        ax.grid(linewidth=0.5)

    axs[0].set_ylabel(r"$\max_m |y(t_m)-\widehat y(\xi_m)|$")
    axs[1].set_ylabel(r"$\max_m |z(t_m)-\widehat z(\xi_m)|$")

    axs = sync_ylim(axs, min_y_set=1e-15)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
    )

    plot_name = f"order_learn_maclaurin_normalized_{num_nodes=}_{degree=}"
    save_fig(plt, plot_name, problem_name)


if __name__ == "__main__":
    main()
