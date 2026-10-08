from collections.abc import Callable

import torch

from pySDC.playgrounds.pinn_dae.models.nonstiff_linear_lpnet import NonstiffLinearLPNet

Tensor = torch.Tensor
DAERightHandSide = Callable[[Tensor, Tensor, Tensor], Tensor]


class TrainerNonstiffLinearLPNet:
    def __init__(self, model, device="cpu"):
        self.model = model

        self.device = device
        self.dtype = torch.float64
    
    def f(
        self,
        y: Tensor,
        t: Tensor,
    ) -> Tensor:
        return -y
    
    def compute_loss(
        self,
        tau_train: Tensor,
        model: NonstiffLinearLPNet,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Physics-informed loss in normalized time.

        With ``t = t0 + dt*xi`` and ``xi in [0, 1]``, the DAE becomes

            dy/dxi - dt*f(y, z, t) = 0,
            g(y, z, t)             = 0.
        """

        tau_train = tau_train.reshape(-1, 1)

        t_train = tau_train
        y_hat = model(tau_train)
        dy_dt = model.evaluate_dy_dt(tau_train)

        assert y_hat.shape == tau_train.shape
        assert dy_dt.shape == tau_train.shape
        assert t_train.shape == tau_train.shape

        residual = dy_dt - self.f(y_hat, t_train)

        loss = torch.mean(residual.square())

        diagnostics = {
            "loss": loss.detach(),
        }
        return loss, diagnostics
    
    def train_model(
        self,
        tau_train: Tensor,
        num_epochs: int = 10000,
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        history: list[dict[str, float]] = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            loss, diagnostics = self.compute_loss(tau_train=tau_train, model=self.model)
            loss.backward()

            optimizer.step()

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                history.append(
                    {
                        "epoch": float(epoch),
                        **{key: value.item() for key, value in diagnostics.items()},
                    }
                )

        return history
