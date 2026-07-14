from __future__ import annotations

import torch
from torch import nn


class DAELPNet(nn.Module):
    """Polynomial LPNet for a semi-explicit DAE on a normalized interval.

    The physical problem is

        y'(t) = f(y(t), z(t), t),
        0     = g(y(t), z(t), t),

    on ``[t0, t0 + dt]``.  The network uses the normalized coordinate

        xi = (t - t0) / dt in [0, 1]

    and represents

        y_hat(xi) = y0 + sum_{j=1}^p alpha_j xi^j,
        z_hat(xi) = z0 + sum_{j=1}^q beta_j  xi^j.

    The coefficients are scaled Taylor coefficients:

        alpha_j = a_j dt^j,   beta_j = b_j dt^j.

    Parameters
    ----------
    y0, z0:
        Consistent initial values.
    degree_y, degree_z:
        Polynomial degrees.
    alpha1:
        Optional fixed first coefficient of y with respect to xi.  For a
        consistent IVP it is ``dt * f(y0, z0, t0)``.  Fixing it removes the
        leading optimization error in y.
    beta1:
        Optional fixed first coefficient of z with respect to xi.  For an
        index-1 DAE it may be computed from the differentiated constraint.
        If omitted, all z coefficients are trained.
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
            raise ValueError("degree_y must be at least 1")
        if degree_z is None:
            degree_z = degree_y
        if degree_z < 1:
            raise ValueError("degree_z must be at least 1")

        y0 = y0.detach().clone().reshape(-1)
        z0 = z0.detach().clone().reshape(-1)

        if y0.numel() != 1 or z0.numel() != 1:
            raise NotImplementedError(
                "This version is written for one scalar y and one scalar z."
            )

        self.register_buffer("y0", y0)
        self.register_buffer("z0", z0)

        self.degree_y = degree_y
        self.degree_z = degree_z

        self.coeffs_y_trainable = nn.Parameter(
            torch.zeros(
                degree_y,
                dtype=y0.dtype,
                device=y0.device,
            )
        )

        self.coeffs_z_trainable = nn.Parameter(
            torch.zeros(
                degree_z,
                dtype=z0.dtype,
                device=z0.device,
            )
        )

    @staticmethod
    def _powers(xi: torch.Tensor, degree: int) -> torch.Tensor:
        """Return ``[xi, xi**2, ..., xi**degree]`` with shape ``(N, degree)``."""
        xi = xi.reshape(-1, 1)
        exponents = torch.arange(
            1,
            degree + 1,
            dtype=xi.dtype,
            device=xi.device,
        ).reshape(1, -1)
        return xi**exponents

    def full_coeffs_y(self) -> torch.Tensor:
        """Return all normalized y coefficients ``[alpha_1, ..., alpha_p]``."""
        return self.coeffs_y_trainable

    def full_coeffs_z(self) -> torch.Tensor:
        """Return all normalized z coefficients ``[beta_1, ..., beta_q]``."""
        return self.coeffs_z_trainable

    def evaluate_y(self, xi: torch.Tensor) -> torch.Tensor:
        xi = xi.reshape(-1, 1)
        powers = self._powers(xi, self.degree_y)
        coeffs_y = self.full_coeffs_y().reshape(-1, 1)
        return self.y0.reshape(1, 1) + powers @ coeffs_y

    def evaluate_z(self, xi: torch.Tensor) -> torch.Tensor:
        xi = xi.reshape(-1, 1)
        powers = self._powers(xi, self.degree_z)
        coeffs_z = self.full_coeffs_z().reshape(-1, 1)
        return self.z0.reshape(1, 1) + powers @ coeffs_z

    def evaluate_dy_dxi(self, xi: torch.Tensor) -> torch.Tensor:
        """Analytic derivative of ``y_hat`` with respect to normalized time xi."""
        xi = xi.reshape(-1, 1)
        exponents = torch.arange(
            1,
            self.degree_y + 1,
            dtype=xi.dtype,
            device=xi.device,
        ).reshape(1, -1)
        derivative_basis = exponents * xi ** (exponents - 1)
        coeffs_y = self.full_coeffs_y().reshape(-1, 1)
        return derivative_basis @ coeffs_y

    def forward(self, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.evaluate_y(xi), self.evaluate_z(xi)
