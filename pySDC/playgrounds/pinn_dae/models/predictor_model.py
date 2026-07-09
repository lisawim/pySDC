import torch
import torch.nn as nn
import torch.optim as optim


class LinearProblemPredictorNet(nn.Module):
    def __init__(
        self,
        lambda_d=-2.0,
        lambda_a=1.0,
        num_epochs=2000, 
        num_hidden_layers=2,
        weight_diff=1.0,
        weight_alg=10000.0,
        width=32,
    ):
        super().__init__()

        self.lambda_d = lambda_d
        self.lambda_a = lambda_a
        self.num_epochs = num_epochs
        self.num_hidden_layers = num_hidden_layers
        self.weight_diff = weight_diff
        self.weight_alg = weight_alg
        self.width = width

        self.optimizer = None

        self.net = self.build_net()

    def build_net(self):
        layers = [nn.Linear(1, self.width), nn.ReLU()]
        for _ in range(self.num_hidden_layers - 1):
            layers.extend([
                nn.Linear(self.width, self.width),
                nn.ReLU(),
            ])

        layers.append(nn.Linear(self.width, 2))
        return nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)
    
    def f(self, y, z):
        """Right-hand side of differential equation."""
        return self.lambda_d * y + self.lambda_a * z
    
    def g(self, y, z):
        """Right-hand side of algebraic constraints."""
        return self.lambda_d * y - self.lambda_a * z
    
    def initialize_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def train_model(self, model, tau_train, tau_sdc, y_n_tau, z_n_tau):
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

        for epoch in range(self.num_epochs):
            tau = tau_train.clone().detach().requires_grad_(True)

            u_pred = model(tau)
            y_pred = u_pred[:, 0]
            z_pred = u_pred[:, 1]

            loss = self.compute_loss(tau, y_pred, z_pred, y_n_tau, z_n_tau)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 200 == 0:
                print(f"Epoch {epoch}: loss = {loss.item():.10f}")

        with torch.no_grad():
            u_pred_nodes = model(tau_sdc)

        y_pred = u_pred_nodes[:, 0]
        z_pred = u_pred_nodes[:, 1]
        return y_pred, z_pred

    def compute_loss(self, tau_nodes, y_pred, z_pred, y_n_tau, z_n_tau):
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
        dy_pred_dtau = torch.autograd.grad(
            outputs=y_pred,
            inputs=tau_nodes,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]
        loss_f = dy_pred_dtau - self.f(y_pred, z_pred)

        # Loss function of the algebraic equation
        loss_g = self.g(y_pred, z_pred)

        # Loss at nodes
        loss_y = y_pred - y_n_tau
        loss_z = z_pred - z_n_tau
        loss_nodes_mean = torch.mean(loss_y ** 2) + torch.mean(loss_z ** 2)

        loss_f_mean = torch.mean(loss_f ** 2)
        loss_g_mean = torch.mean(loss_g ** 2)

        loss = loss_f_mean + loss_g_mean + loss_nodes_mean
        return loss