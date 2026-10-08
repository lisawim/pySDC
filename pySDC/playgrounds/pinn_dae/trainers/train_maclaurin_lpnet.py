from __future__ import annotations

from typing import Callable

import math
import torch

from pySDC.playgrounds.pinn_dae.models.linear_dae_lpnet import LinearDAELPNet

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
    model: LinearDAELPNet,
    tau_train: Tensor,
    f_rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    g_rhs: Callable[[Tensor, Tensor, Tensor], Tensor],
    weight_differential: float = 1.0,
    weight_algebraic: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Physics-informed loss in normalized time.

    With ``t = t0 + dt*xi`` and ``xi in [0, 1]``, the DAE becomes

        dy/dt - dt*f(y, z, t) = 0,
        g(y, z, t)             = 0.
    """
    tau_train = tau_train.reshape(-1, 1)

    t_train = tau_train
    y_hat, z_hat = model(tau_train)
    dy_dt = model.evaluate_dy_dt(tau_train)

    assert y_hat.shape == tau_train.shape
    assert z_hat.shape == tau_train.shape
    assert dy_dt.shape == tau_train.shape
    assert t_train.shape == tau_train.shape

    differential_residual = dy_dt - f_rhs(y_hat, z_hat, t_train)
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
    model: LinearDAELPNet,
    tau_train: Tensor,
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
            tau_train=tau_train,
            f_rhs=f_rhs,
            g_rhs=g_rhs,
            weight_algebraic=weight_algebraic,
        )
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == num_steps - 1:
            history.append(
                {
                    "epoch": int(step),
                    **{key: value.item() for key, value in diagnostics.items()},
                }
            )

    return history


def refine_with_lbfgs(
    model: LinearDAELPNet,
    tau_train: Tensor,
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

    history: list[dict[str, float]] = []
    evaluation = 0

    def closure() -> Tensor:
        nonlocal evaluation

        optimizer.zero_grad()
        loss, diagnostics = dae_lpnet_loss(
            model=model,
            tau_train=tau_train,
            f_rhs=f_rhs,
            g_rhs=g_rhs,
            weight_algebraic=weight_algebraic,
        )
        loss.backward()

        history.append(
            {
                "evaluation": int(evaluation),
                **{key: value.item() for key, value in diagnostics.items()},
            }
        )
        evaluation += 1
        return loss

    optimizer.step(closure)
    return history


def print_coefficients(model: LinearDAELPNet, lambda_d=-2.0, lambda_a=1.0) -> None:
    """Compare learned coefficients in xi with exact coefficients."""

    coeffs_y = model.full_coeffs_y().detach().reshape(-1)
    for j, coeff in enumerate(coeffs_y, start=1):
        exact = ((2.0 * lambda_d) ** j) / math.factorial(j)
        print(
            f"alpha{j}: learned={coeff.item():.16e}, "
            f"exact={exact:.16e}, "
            f"error={abs(coeff.item() - exact):.6e}"
        )

    coeffs_z = model.full_coeffs_z().detach().reshape(-1)
    for j, coeff in enumerate(coeffs_z, start=1):
        fac = lambda_d / lambda_a
        exact = fac * ((2.0 * lambda_d) ** j) / math.factorial(j)
        print(
            f"beta{j}: learned={coeff.item():.16e}, "
            f"exact={exact:.16e}, "
            f"error={abs(coeff.item() - exact):.6e}"
        )
