import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.plot_helper import figsize_by_journal
from pySDC.playgrounds.pinn_dae.utils import my_setup_mpl
from pySDC.playgrounds.pinn_dae.models.predictor_model import LinearProblemPredictorNet

from qmat import Q_GENERATORS, QDELTA_GENERATORS


def compute_constant_reference_order(dt_list, err_iter, k):
    """Computes constants to shift reference order lines in plot."""

    dt_ref = dt_list[0]
    err_ref = err_iter[0]

    C = err_ref / dt_ref ** (k + 1)
    return C


def choose_time_step_sizes(problem_name):
    """Returns time step sizes suitable for each problem."""
    if problem_name == "LINEAR-TEST":
        n_steps_list = [2, 5, 10, 20, 50, 100, 200, 500]
        Tend = 1.0
    else:
        raise NotImplementedError

    dt_list = [Tend / n_steps for n_steps in n_steps_list]
    return dt_list, Tend


def save_fig(plot, plot_name, problem_name):
    out = Path("data") / problem_name / f"{plot_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot.savefig(out, dpi=500, bbox_inches="tight")
    plot.close()


def sync_ylim(axs, min_y_set=1e-15):
    """Synchronize y-axis limits across all subplots by finding the global min/max."""
    min_y, max_y = None, None

    # Find global min/max y-limits across all axes
    for ax in axs:
        y_limits = ax.get_ylim()

        # Ignore non-positive values for log scale
        if ax.get_yscale() == "log":
            y_limits = [y for y in y_limits if y > 0]
            if not y_limits:
                continue

        if min_y is None or y_limits[0] < min_y:
            min_y = y_limits[0]
        if max_y is None or y_limits[1] > max_y:
            max_y = y_limits[1]

    # Apply the same limits to all subplots
    for ax in axs:
        if ax.get_yscale() == "log":
            if min_y is not None and min_y <= 0:
                min_y = min_y_set
            ax.set_ylim(min_y, max_y)
        else:
            ax.set_ylim(min_y, max_y)

    return axs


def u_exact(t, lambda_d=-2.0, lambda_a=1.0):
    return torch.exp(2 * lambda_d * t), lambda_d / lambda_a * torch.exp(2 * lambda_d * t)


def get_collocation(num_nodes):
    QGenerator = Q_GENERATORS["coll"]

    coll = QGenerator(nNodes=num_nodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT")
    nodes = coll.nodes
    Qmat = coll.Q
    return Qmat, nodes

def get_Qdelta(nodes, num_nodes, Qmat, QI):
    QIGenerator = QDELTA_GENERATORS[QI]

    nodes = np.array(nodes)
    approx = QIGenerator(Q=Qmat, nNodes=num_nodes, nodeType="LEGENDRE", quadType="RADAU-RIGHT", nodes=nodes)

    QImat = approx.getQDelta()

    return QImat


def plot_order_provisional_solution():
    problem_name = "LINEAR-TEST"

    width = 200
    num_hidden_layers = 10
    num_extra_points = 100

    # Define the number of collocation nodes
    num_nodes = 3
    QI = "IE"

    t0 = 0.0
    t0 = torch.tensor([[t0]], dtype=torch.float64)
    dt_list, _ = choose_time_step_sizes(problem_name)
    dt_list_short = dt_list[1 : 5]

    errors_y0, errors_z0 = [], []
    for dt in dt_list:
        print(f"Training for dt = {dt}")

        dt = torch.tensor([[dt]], dtype=torch.float64)

        _, nodes = get_collocation(num_nodes)
        tau_nodes = [t0 + dt * coll_node for coll_node in nodes]

        tau0 = torch.zeros((1, 1), dtype=torch.float64).reshape(-1, 1)
        tau_sdc = torch.tensor(
            tau_nodes,
            dtype=torch.float64,
        ).reshape(-1, 1)

        # Generate extra random points in the interval [t0, t0 + dt]
        tau_extra = t0 + dt * torch.rand(num_extra_points, 1, dtype=torch.float64)

        tau_train = torch.cat(
            [tau0, tau_sdc, tau_extra],
            dim=0,
        )
        tau_train.requires_grad_(True)

        # Create an instance of the PredictorNet model
        model = LinearProblemPredictorNet(num_hidden_layers=num_hidden_layers, width=width).double()

        tau_exact = tau_train.detach()

        y_n_tau, z_n_tau = u_exact(tau_exact)

        y_n_tau = y_n_tau.reshape(-1, 1)
        z_n_tau = z_n_tau.reshape(-1, 1)

        y_pred, z_pred = model.train_model(
            model=model,
            tau_train=tau_train,
            tau_sdc=tau_sdc,
            y_n_tau=y_n_tau,
            z_n_tau=z_n_tau,
        )

        y_ex, z_ex = u_exact(tau_sdc)#u_exact(t0 + dt)
        errors_y0.append(torch.max(torch.abs(y_pred - y_ex)).detach().cpu().item())
        errors_z0.append(torch.max(torch.abs(z_pred - z_ex)).detach().cpu().item())

    figsize = figsize_by_journal(journal="Springer_proceedings", scale=0.7, ratio=0.5)
    my_setup_mpl(fontsize=5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    axs[0].loglog(
        dt_list,
        errors_y0,
        linewidth=0.7,
        color="blue",
        marker="o",
        linestyle="solid",
        label=f"y_0",
    )

    axs[1].loglog(
        dt_list,
        errors_z0,
        linewidth=0.7,
        color="blue",
        marker="o",
        linestyle="solid",
        label=f"z_0",
    )

    Cy = compute_constant_reference_order(dt_list, errors_y0, k=0)
    Cz = compute_constant_reference_order(dt_list, errors_z0, k=0)

    # Reference order
    ref_y = [Cy * dt for dt in dt_list_short]
    axs[0].loglog(
        dt_list_short,
        ref_y,
        linewidth=0.7,
        color="black",
        linestyle="dashed",
    )

    axs[0].text(
        dt_list_short[-1] * 0.8,
        ref_y[-1] * 0.7,
        r"$1$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    ref_z = [Cz * dt for dt in dt_list_short]
    axs[1].loglog(
        dt_list_short,
        ref_z,
        linewidth=0.7,
        color="black",
        linestyle="dashed",
    )

    axs[1].text(
        dt_list_short[-1] * 0.8,
        ref_z[-1] * 0.7,
        r"$1$",
        fontsize=6,
        va="center",
        ha="left",
        color="black",
    )

    for ax in axs:
        ax.tick_params(axis="both", which="minor", bottom=False, left=False)

        ax.set_xlabel(r"time step size $\Delta t$")

        ax.grid(linewidth=0.5)

    axs[0].set_ylabel(r"$\max_{m\in\{1,..,M\}}|y(\tau_m) - y^0_{m,t_1}|$")
    axs[1].set_ylabel(r"$\max_{m\in\{1,..,M\}}|z(\tau_m) - z^0_{m,t_1}|$")

    axs = sync_ylim(axs, min_y_set=1e-15)

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    plot_name = f"order_predicted_provisional_solution_{num_nodes=}"
    save_fig(plt, plot_name, problem_name)


if __name__ == "__main__":
    plot_order_provisional_solution()