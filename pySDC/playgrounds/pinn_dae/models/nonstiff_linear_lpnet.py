import math
import torch
from torch import nn

Tensor = torch.Tensor


class NonstiffLinearLPNet(nn.Module):
    def __init__(
        self,
        y0: Tensor | float,
        t0: Tensor | float,
        degree: int = 3,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        fix_a1: bool = True,
    ) -> None:
        super().__init__()

        self.fix_a1 = fix_a1

        if degree < 1:
            raise ValueError("degree must be at least 1")
        
        self.degree = int(degree)

        y0_tensor = self._as_scalar_column(
            y0,
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
                a1 = -y0_tensor

                a1_tensor = self._as_scalar_column(
                    a1,
                    dtype=dtype,
                    device=device,
                )

                self.register_buffer("a1", a1_tensor)

                self.coeffs_y_trainable = nn.Parameter(
                    torch.zeros(
                        degree - 1,
                        1,
                        dtype=y0.dtype,
                        device=y0.device,
                    )
                )
        else:
            self.coeffs_y_trainable = nn.Parameter(
                torch.zeros(
                    degree,
                    1,
                    dtype=y0.dtype,
                    device=y0.device,
                )
            )

        self.register_buffer("y0", y0_tensor)
        self.register_buffer("t0", t0_tensor)
    
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
                last_degree=self.degree,
            )
        else:
            y_hat = self.y0

            # For degree == 1, both matrices have shape (N, 0) and (0, 1).
            higher_powers = self._power_basis(
                tau,
                first_degree=1,
                last_degree=self.degree,
            )

        y_hat = (
            y_hat
            + higher_powers @ self.coeffs_y_trainable
        )

        return y_hat

    def evaluate_dy_dt(self, t: Tensor) -> Tensor:
        """Analytic derivative of ``y_hat`` with respect to time t."""
        tau = self._prepare_time(t)

        # Contribution from a1 * tau.
        if self.fix_a1:
            dy_dt = self.a1.expand(tau.shape[0], 1)

            # Contributions from a2*tau**2, ..., ap*tau**p.
            higher_derivative_basis = self._derivative_basis(
                tau,
                first_degree=2,
                last_degree=self.degree,
            )

            dy_dt = (
                dy_dt
                + higher_derivative_basis
                @ self.coeffs_y_trainable
            )
        else:
            # Contributions from a2*tau**2, ..., ap*tau**p.
            higher_derivative_basis = self._derivative_basis(
                tau,
                first_degree=1,
                last_degree=self.degree,
            )

            dy_dt = higher_derivative_basis @ self.coeffs_y_trainable

        return dy_dt

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return self.evaluate_y(t)
    
    def print_coefficients(self) -> None:
        """Compare learned coefficients in xi with exact coefficients."""

        coeffs_y = self.full_coeffs_y().detach().reshape(-1)
        for j, coeff in enumerate(coeffs_y, start=1):
            exact = ((-1) ** j) / math.factorial(j)
            print(
                f"alpha{j}: learned={coeff.item():.16e}, "
                f"exact={exact:.16e}, "
                f"error={abs(coeff.item() - exact):.6e}"
            )
