from __future__ import annotations

import torch
from torch import nn


class DAELPNet(nn.Module):
    """
    Polynomial LPNet for a semi-explicit DAE

        y'(t) = f(y(t), z(t), t),
        0     = g(y(t), z(t), t).

    The network learns polynomial coefficients on the normalized interval

        tau = (t - t0) / dt in [0, 1].
    """

    def __init__(
        self,
        y0: torch.Tensor,
        z0: torch.Tensor,
        degree_y: int,
        degree_z: int | None = None,
    ) -> None:
        super().__init__()

        if degree_y < 1:
            raise ValueError("degree_y must be at least 1.")

        if degree_z is None:
            degree_z = degree_y

        if degree_z < 1:
            raise ValueError("degree_z must be at least 1.")

        y0 = y0.detach().clone().reshape(-1)
        z0 = z0.detach().clone().reshape(-1)

        # Fixed constant terms of the Maclaurin expansions.
        self.register_buffer("y0", y0)
        self.register_buffer("z0", z0)

        self.degree_y = degree_y
        self.degree_z = degree_z

        # coeffs_y[j - 1] is the coefficient of tau**j.
        self.coeffs_y = nn.Parameter(
            torch.zeros(
                degree_y,
                dtype=y0.dtype,
                device=y0.device,
            )
        )

        # coeffs_z[j - 1] is the coefficient of tau**j.
        self.coeffs_z = nn.Parameter(
            torch.zeros(
                degree_z,
                dtype=z0.dtype,
                device=z0.device,
            )
        )

    @staticmethod
    def _powers(tau: torch.Tensor, degree: int) -> torch.Tensor:
        """
        Return [tau, tau^2, ..., tau^degree].

        tau has shape (N, 1), output has shape (N, degree).
        """

        tau = tau.reshape(-1, 1)
        exponents = torch.arange(
            1,
            degree + 1,
            dtype=tau.dtype,
            device=tau.device,
        ).reshape(1, -1)
        return tau ** exponents

    def get_coeffs_y(self):
        return self.coeffs_y
    
    def get_coeffs_z(self):
        return self.coeffs_z

    def evaluate_y(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1)
        powers = self._powers(tau, self.degree_y)

        coeffs_y = self.get_coeffs_y().reshape(-1, 1)
        return self.y0.reshape(1, 1) + powers @ coeffs_y

    def evaluate_z(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1)
        powers = self._powers(tau, self.degree_z)

        coeffs_z = self.get_coeffs_z().reshape(-1, 1)
        return self.z0.reshape(1, 1) + powers @ coeffs_z

    def evaluate_dy_dtau(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Analytic derivative of y_hat with respect to tau.
        """
        tau = tau.reshape(-1, 1)

        exponents = torch.arange(
            1,
            self.degree_y + 1,
            dtype=tau.dtype,
            device=tau.device,
        ).reshape(1, -1)

        derivative_basis = exponents * tau ** (exponents - 1)

        coeffs_y = self.get_coeffs_y().reshape(-1, 1)
        return derivative_basis @ coeffs_y

    def forward(self, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.evaluate_y(tau), self.evaluate_z(tau)