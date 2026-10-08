from collections.abc import Callable

import torch

from pySDC.playgrounds.pinn_dae.models.linear_dae_lpnet import LinearDAELPNet

Tensor = torch.Tensor
DAERightHandSide = Callable[[Tensor, Tensor, Tensor], Tensor]


class TrainerLinearDAELPNet:
    def __init__(self, model, device="cpu"):
        self.model = model

        self.device = device
        self.dtype = torch.float64
    
    def f(
        self,
        y: Tensor,
        z: Tensor,
        t: Tensor,
        lambda_d: float = -2.0,
        lambda_a: float = 1.0,
    ) -> Tensor:
        return lambda_d * y + lambda_a * z

    def g(
        self,
        y: Tensor,
        z: Tensor,
        t: Tensor,
        lambda_d: float = -2.0,
        lambda_a: float = 1.0,
    ) -> Tensor:
        return lambda_d * y - lambda_a * z
    
    def compute_loss(
        self,
        tau_train: Tensor,
        model: LinearDAELPNet,
        w_d: float = 1.0,
        w_a: float = 1.0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Physics-informed loss in normalized time.

        With ``t = t0 + dt*xi`` and ``xi in [0, 1]``, the DAE becomes

            dy/dxi - dt*f(y, z, t) = 0,
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

        differential_residual = dy_dt - self.f(y_hat, z_hat, t_train)
        algebraic_residual = self.g(y_hat, z_hat, t_train)

        loss_differential = torch.mean(differential_residual.square())
        loss_algebraic = torch.mean(algebraic_residual.square())
        loss = w_d * loss_differential + w_a * loss_algebraic

        diagnostics = {
            "loss": loss.detach(),
            "loss_differential": loss_differential.detach(),
            "loss_algebraic": loss_algebraic.detach(),
            "max_differential_residual": differential_residual.detach().abs().max(),
            "max_algebraic_residual": algebraic_residual.detach().abs().max(),
        }
        return loss, diagnostics
    
    def train_model(
        self,
        tau_train: Tensor,
        num_epochs: int = 20000,
        w_d: float = 1.0,
        w_a: float = 1.0,
        beta: float = 1.1,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=100,
            gamma=0.5,
        )

        history: list[dict[str, float]] = []

        w_d_k, w_a_k = w_d, w_a

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            loss, diagnostics = self.compute_loss(tau_train=tau_train, model=self.model, w_d=w_d_k, w_a=w_a_k)
            loss.backward()

            optimizer.step()

            # scheduler.step()

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                history.append(
                    {
                        "epoch": float(epoch),
                        **{key: value.item() for key, value in diagnostics.items()},
                    }
                )

        return history
    
    def refine(
        self,
        tau_train: Tensor,
        max_iter: int = 1000,
        w_d: float = 1.0,
        w_a: float = 1.0,
    ) -> None:
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
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
            loss, diagnostics = self.compute_loss(
                tau_train=tau_train,
                model=self.model,
                w_d=w_d,
                w_a=w_a,
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