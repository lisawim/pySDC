import math
import torch
import torch.nn as nn
import torch.optim as optim


class LPNet(nn.Module):
    def __init__(
        self, M, u0, degree, diff
    ):
        super().__init__()

        self.lambda_d = -2.0
        self.lambda_a = 1.0

        u0 = torch.as_tensor(u0, dtype=torch.float64).reshape(1, -1)

        self.degree = degree
        self.output_dim = u0.shape[1]

        self.diff = diff

        # Fest vorgegebener Maclaurin-Koeffizient c_0 = u(0)
        self.register_buffer("u0", u0)

        self.rhs = self.f if diff else self.g

        # Lernbare Gewichte für c_1, ..., c_degree
        self.linear = nn.Linear(
            in_features=degree,
            out_features=self.output_dim,
            bias=False,
            dtype=torch.float64,
        )

        # Sinnvolle Initialisierung: zunächst fast konstante Lösung u(t) ≈ u0
        nn.init.zeros_(self.linear.weight)

        self.optimizer = None
        self.loss_fn = nn.MSELoss()
    
    def polynomial_features(self, t):
        """
        Erzeugt [t, t^2, ..., t^degree].

        Input:
            t: shape (N,) oder (N, 1)

        Output:
            shape (N, degree)
        """
        t = t.reshape(-1, 1)

        return torch.cat(
            [t**j for j in range(1, self.degree + 1)],
            dim=1,
        )

    def forward(self, t):
        """
        Input:
            t: Zeitpunkte, shape (N,) oder (N, 1)

        Output:
            Approximation u(t), shape (N, output_dim)
        """
        features = self.polynomial_features(t)

        return self.u0 + self.linear(features)

    def get_coefficients(self):
        """
        Gibt [c_0, c_1, ..., c_degree] zurück.

        Shape:
            (degree + 1, output_dim)
        """
        c0 = self.u0.reshape(1, -1)

        # linear.weight hat Shape (output_dim, degree);
        # jede Zeile soll ein Koeffizientenvektor c_j sein
        c_rest = self.linear.weight.T

        return torch.cat([c0, c_rest], dim=0)
    
    def exact_maclaurin_coefficients(self):
        # u0 = torch.as_tensor(u0, dtype=torch.float64).reshape(-1)

        coeffs_y = []
        coeffs_z = []

        for j in range(self.degree + 1):
            c_y = self.u0 * (2.0 * self.lambda_d) ** j / math.factorial(j)
            c_z = (self.lambda_d / self.lambda_a) * c_y

            coeffs_y.append(c_y)
            coeffs_z.append(c_z)

        return torch.stack(coeffs_y) if self.diff else torch.stack(coeffs_z)
    
    def f(self, y, z):
        """Right-hand side of differential equation."""
        return self.lambda_d * y + self.lambda_a * z
    
    def g(self, y, z):
        """Right-hand side of algebraic constraints."""
        return self.lambda_d * y - self.lambda_a * z
    
    def initialize_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def train_model(self, tau_nodes, y_ex, z_ex):
        """
        Trains the model to learn values of differential and algebraic variables.
        Returns the prediction at collocation nodes.

        Parameters
        ----------
        tau_sdc : torch.tensor
            Tensor of collocation nodes.
        y_n : torch.tensor
            Exact solution of differential variable at initial time t0.
        z_n : torch.tensor
            Exact solution of algebraic variable at initial time t0.

        Returns
        -------
        y_pred : torch.tensor
            Prediction of differential variable.
        z_pred : torch.tensor
            Prediction of algebraic variable.
        """

        self.initialize_optimizer()

        for epoch in range(5000):
            tau = tau_nodes.clone().detach().requires_grad_(True)

            self.optimizer.zero_grad()

            u_pred = self(tau)

            loss = self.compute_loss(tau, u_pred, y_ex, z_ex)

            loss.backward()
            self.optimizer.step()

        print(f"loss={loss.item():.3e}")

        learned_coeffs = self.get_coefficients()
        exact_coeffs = self.exact_maclaurin_coefficients().reshape(self.degree + 1, 1)
        print(f"learned_coeffs={learned_coeffs.detach().cpu().numpy().flatten()}")
        print(f"exact_coeffs={exact_coeffs.detach().cpu().numpy().flatten()}")

        # u_pred_nodes = self(tau_nodes)
        # u_pred = u_pred_nodes[:, 0]
        return self(tau_nodes).detach()#u_pred

    def compute_loss(self, tau_nodes, u_pred, y_ex, z_ex):
        """
        Computes the loss function for SDC.

        Parameters
        ----------
        y_pred : torch.tensor
            Prediction of differential variable.
        z_pred : torch.tensor
            Prediction of algebraic variable.
        y_n : torch.tensor
            Exact solution of differential variable at initial time t0.
        z_n : torch.tensor
            Exact solution of algebraic variable at initial time t0.

        Returns
        -------
        loss : torch.tensor
            Loss function.
        """

        # Loss function of the differential equation
        if self.diff:
            dy_pred_dtau = torch.autograd.grad(
                outputs=u_pred,
                inputs=tau_nodes,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=True,
            )[0]
            loss_eq = self.loss_fn(dy_pred_dtau, self.f(u_pred, z_ex))
            loss_nodes = self.loss_fn(u_pred, y_ex)
        else:
            # Loss function of the algebraic equation
            loss_eq = self.loss_fn(torch.zeros_like(u_pred), self.g(y_ex, u_pred))
            loss_nodes = self.loss_fn(u_pred, z_ex)

        learned_coeffs = self.get_coefficients()
        exact_coeffs = self.exact_maclaurin_coefficients().reshape(self.degree + 1, 1)

        loss_coeffs =  self.loss_fn(learned_coeffs, exact_coeffs)
        # loss = loss_eq + loss_nodes + loss_coeffs
        loss = loss_coeffs
        return loss


class MaclaurinCoefficientNet(nn.Module):
    """
    Lernt direkt die Maclaurin-Koeffizienten c_1, ..., c_degree.

    c_0 = y0 wird fest vorgegeben.
    """

    def __init__(self, u0, is_diff, degree=5, num_epochs=5000):
        super().__init__()

        u0 = torch.as_tensor(u0, dtype=torch.float64).reshape(1, 1)
        self.register_buffer("c0", u0)

        self.is_diff = is_diff
        self.degree = degree

        self.lambda_d = -2.0
        self.lambda_a = 1.0
        self.num_epochs = num_epochs

        # Lernbare Koeffizienten c_1, ..., c_degree
        self.coeffs = nn.Parameter(
            torch.zeros(self.degree, 1, dtype=torch.float64)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self):
        """
        Output:
            coeffs: Tensor mit Shape (degree + 1, 1)

        Enthält:
            [c_0, c_1, ..., c_degree]
        """
        return torch.cat([self.c0, self.coeffs], dim=0)
    
    def exact_coefficients(self):
        """
        Exakte Koeffizienten für

            y(t) = exp(2 lambda_d t)

        also

            c_j = (2 lambda_d)^j / j!
        """

        coeffs = []

        for j in range(self.degree + 1):
            c_j = 1 if self.is_diff else self.lambda_d / self.lambda_a
            c_j *= (2.0 * self.lambda_d) ** j / math.factorial(j)

            c_j = torch.tensor(c_j, dtype=torch.float64).reshape(1, 1)
            coeffs.append(c_j)

        return torch.cat(coeffs, dim=0)

    def train_coefficient_net(
        self,
        model,
    ):
        coeffs_exact = self.exact_coefficients()

        optimizer = optim.Adam(model.parameters(), lr=1e-2)

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            coeffs_pred = model() # difference between self() and model()!!!

            loss = self.loss_fn(coeffs_pred, coeffs_exact)

            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                print(f"epoch={epoch:5d}, loss={loss.item():.3e}")

        return coeffs_pred


def evaluate_series(t, coeffs, degree):
    """
    Wertet die gelernte Maclaurin-Reihe an Zeitpunkten t aus.

    t: shape (N, 1) oder (N,)
    Output: y(t), shape (N, 1)
    """
    t = t.reshape(-1, 1)

    powers = torch.cat(
        [t**j for j in range(degree + 1)],
        dim=1,
    )

    return powers @ coeffs


def evaluate_series_vectorized(t, coeffs):
    """
    t:      shape (N, 1)
    coeffs shape (degree + 1, 1)

    returns shape (N, 1)
    """
    t = t.reshape(-1, 1)
    coeffs = coeffs.reshape(-1, 1)

    degree = coeffs.shape[0] - 1

    powers = torch.cat(
        [t**j for j in range(degree + 1)],
        dim=1,
    )

    return powers @ coeffs
