import torch

from pySDC.playgrounds.pinn_dae.models.linear_dae_lpnet import LinearDAELPNet, Trainer
from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained


class SDC_C_ML(genericImplicitConstrained):
    def predict(self) -> None:
        """
        Predictor to fill values at nodes before first sweep

        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """

        # get current level and problem description
        L = self.level
        P = L.prob

        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0], L.time)

        if self.params.initial_guess == "ML_train":
            degree = self.params.degree

            model = LinearDAELPNet(
                y0=torch.as_tensor(L.u[0].diff[0], dtype=torch.float64),
                z0=torch.as_tensor(L.u[0].alg[0], dtype=torch.float64),
                degree=degree,
            ).double()

            trainer = Trainer(prob=P, model=model, t0=L.time, dt=L.dt)

            history = trainer.train_model()

            t0 = torch.as_tensor(L.time, dtype=torch.float64)
            dt = torch.as_tensor(L.dt, dtype=torch.float64)
            coll_nodes = torch.as_tensor(self.coll.nodes[:], dtype=torch.float64).reshape(-1, 1)
            t_eval = t0 + dt * coll_nodes

            y_pred_eval = model.evaluate_y(coll_nodes)
            z_pred_eval = model.evaluate_z(coll_nodes)

            for m in range(1, self.coll.num_nodes + 1):
                u_pred = P.dtype_u(P.init)
                u_pred.diff[0] = y_pred_eval[m - 1].detach()
                u_pred.alg[0] = z_pred_eval[m - 1].detach()
                L.u[m] = u_pred[:]
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])

            # indicate that this level is now ready for sweeps
            L.status.unlocked = True
            L.status.updated = True
        else:
            super().predict()
