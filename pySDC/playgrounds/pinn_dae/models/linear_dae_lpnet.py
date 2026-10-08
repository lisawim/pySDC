from collections.abc import Callable

import math
import torch
from torch import nn

Tensor = torch.Tensor
DAERightHandSide = Callable[[Tensor, Tensor, Tensor], Tensor]


class LinearDAELPNet(nn.Module):
    """Polynomial LPNet for a scalar semi-explicit DAE.

    The DAE is

        y'(t) = f(y(t), z(t), t),
        0     = g(y(t), z(t), t).

    The polynomial approximations are

        y_hat(t)
            = y0
            + a1 * (t - t0)
            + sum_{j=2}^p a_j * (t - t0)^j,

        z_hat(t)
            = z0
            + sum_{j=1}^p b_j * (t - t0)^j.

    The first differential coefficient is fixed by

        a1 = f(y0, z0, t0).

    Thus only a2, ..., ap are trained for y. All coefficients
    b1, ..., bp are trained for z.

    This implementation supports every degree >= 1.

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
        y0: Tensor | float,
        z0: Tensor | float,
        t0: Tensor | float,
        degree_y: int = 3,
        degree_z: int = 3,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        fix_a1: bool = True,
    ) -> None:
        super().__init__()

        self.fix_a1 = fix_a1

        self.lambda_d = -2.0
        self.lambda_a = 1.0

        if degree_y < 1 or degree_z < 1:
            raise ValueError("degrees must be at least 1")
        
        self.degree_y = int(degree_y)
        self.degree_z = int(degree_z)

        y0_tensor = self._as_scalar_column(
            y0,
            dtype=dtype,
            device=device,
        )
        z0_tensor = self._as_scalar_column(
            z0,
            dtype=dtype,
            device=device,
        )
        t0_tensor = self._as_scalar_column(
            t0,
            dtype=dtype,
            device=device,
        )

        if self.fix_a1:
            with torch.no_grad():
                a1 = self.lambda_d * y0_tensor - self.lambda_a * z0_tensor

                a1_tensor = self._as_scalar_column(
                    a1,
                    dtype=dtype,
                    device=device,
                )

                self.register_buffer("a1", a1_tensor)

                self.coeffs_y_trainable = nn.Parameter(
                    torch.zeros(
                        self.degree_y - 1,
                        1,
                        dtype=y0.dtype,
                        device=y0.device,
                    )
                )
        else:
            self.coeffs_y_trainable = nn.Parameter(
                torch.zeros(
                    self.degree_y,
                    1,
                    dtype=y0.dtype,
                    device=y0.device,
                )
            )

        self.register_buffer("y0", y0_tensor)
        self.register_buffer("z0", z0_tensor)
        self.register_buffer("t0", t0_tensor)

        self.coeffs_z_trainable = nn.Parameter(
            torch.zeros(
                self.degree_z,
                1,
                dtype=z0.dtype,
                device=z0.device,
            )
        )
    
    @staticmethod
    def _as_scalar_column(
        value: Tensor | float,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> Tensor:
        """Convert a scalar input to a tensor with shape (1, 1)."""
        tensor = torch.as_tensor(
            value,
            dtype=dtype,
            device=device,
        )

        if tensor.numel() != 1:
            raise ValueError(
                f"Tensor must contain exactly one scalar, "
                f"but got shape {tuple(tensor.shape)}."
            )

        return tensor.detach().clone().reshape(1, 1)

    def _prepare_time(self, t: Tensor | float) -> Tensor:
        """Convert evaluation times to shape (N, 1)."""
        return torch.as_tensor(
            t,
            dtype=self.y0.dtype,
            device=self.y0.device,
        ).reshape(-1, 1)
    
    @staticmethod
    def _derivative_basis(
        tau: Tensor,
        first_degree: int,
        last_degree: int,
    ) -> Tensor:
        """Return j*tau**(j-1) for j=first_degree,...,last_degree."""
        tau = tau.reshape(-1, 1)

        if first_degree > last_degree:
            return torch.empty(
                tau.shape[0],
                0,
                dtype=tau.dtype,
                device=tau.device,
            )

        exponents = torch.arange(
            first_degree,
            last_degree + 1,
            dtype=tau.dtype,
            device=tau.device,
        ).reshape(1, -1)

        return exponents * tau ** (exponents - 1)

    @staticmethod
    def _power_basis(
        tau: Tensor,
        first_degree: int,
        last_degree: int,
    ) -> Tensor:
        """Return powers tau**j for j=first_degree,...,last_degree.

        The result has shape

            (number of evaluation points, number of powers).

        If first_degree > last_degree, an empty matrix of shape (N, 0)
        is returned. This makes the degree-one case work naturally.
        """
        tau = tau.reshape(-1, 1)

        if first_degree > last_degree:
            return torch.empty(
                tau.shape[0],
                0,
                dtype=tau.dtype,
                device=tau.device,
            )

        exponents = torch.arange(
            first_degree,
            last_degree + 1,
            dtype=tau.dtype,
            device=tau.device,
        ).reshape(1, -1)

        return tau**exponents

    def full_coeffs_y(self) -> Tensor:
        """Return all normalized y coefficients ``[alpha_1, ..., alpha_p]``."""
        if self.fix_a1:
            return torch.cat(
                (
                    self.a1,
                    self.coeffs_y_trainable,
                ),
                dim=0,
            )
        else:
            return self.coeffs_y_trainable

    def full_coeffs_z(self) -> Tensor:
        """Return all normalized z coefficients ``[beta_1, ..., beta_q]``."""
        return self.coeffs_z_trainable

    def evaluate_y(self, t: Tensor) -> Tensor:
        tau = self._prepare_time(t)

        # Fixed first-order term.
        if self.fix_a1:
            y_hat = (
                self.y0
                + self.a1 * tau
            )

            # For degree == 1, both matrices have shape (N, 0) and (0, 1).
            higher_powers = self._power_basis(
                tau,
                first_degree=2,
                last_degree=self.degree_y,
            )

            y_hat = (
                y_hat
                + higher_powers @ self.coeffs_y_trainable
            )
        else:
            higher_powers = self._power_basis(
                tau,
                first_degree=1,
                last_degree=self.degree_y,
            )

            y_hat = (
                self.y0
                + higher_powers @ self.coeffs_y_trainable
            )

        return y_hat

    def evaluate_z(self, t: Tensor) -> Tensor:
        tau = self._prepare_time(t)

        powers = self._power_basis(
            tau,
            first_degree=1,
            last_degree=self.degree_z,
        )

        return (
            self.z0
            + powers @ self.coeffs_z_trainable
        )

    def evaluate_dy_dt(self, t: Tensor) -> Tensor:
        """Analytic derivative of ``y_hat`` with respect to time t."""
        tau = self._prepare_time(t)

        if self.fix_a1:
            # Contribution from a1 * tau.
            dy_dt = self.a1.expand(tau.shape[0], 1)

            # Contributions from a2*tau**2, ..., ap*tau**p.
            higher_derivative_basis = self._derivative_basis(
                tau,
                first_degree=2,
                last_degree=self.degree_y,
            )

            dy_dt = (
                dy_dt
                + higher_derivative_basis
                @ self.coeffs_y_trainable
            )
        else:
            higher_derivative_basis = self._derivative_basis(
                tau,
                first_degree=1,
                last_degree=self.degree_y,
            )

            dy_dt = higher_derivative_basis @ self.coeffs_y_trainable

        return dy_dt

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return self.evaluate_y(t), self.evaluate_z(t)

    def print_coefficients(self) -> None:
        """Compare learned coefficients in xi with exact coefficients."""

        coeffs_y = self.full_coeffs_y().detach().reshape(-1)
        for j, coeff in enumerate(coeffs_y, start=1):
            exact = ((2.0 * self.lambda_d) ** j) / math.factorial(j)
            print(
                f"alpha{j}: learned={coeff.item():.16e}, "
                f"exact={exact:.16e}, "
                f"error={abs(coeff.item() - exact):.6e}"
            )

        coeffs_z = self.full_coeffs_z().detach().reshape(-1)
        for j, coeff in enumerate(coeffs_z, start=1):
            fac = self.lambda_d / self.lambda_a
            exact = fac * ((2.0 * self.lambda_d) ** j) / math.factorial(j)
            print(
                f"beta{j}: learned={coeff.item():.16e}, "
                f"exact={exact:.16e}, "
                f"error={abs(coeff.item() - exact):.6e}"
            )
