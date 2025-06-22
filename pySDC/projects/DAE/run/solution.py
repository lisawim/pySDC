import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.projects.DAE import compute_solution, my_setup_mpl

from pySDC.helpers.stats_helper import get_sorted


def plot_numerical_solution(problem_name, dt=1e-2, num_nodes=3, problem_type="constrainedDAE", QI="LU"):
    t0 = 0.0
    Tend = 1.0

    hook_class = [LogSolution]

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class
    )

    u_val = get_sorted(solution_stats, type="u", sortby="time")

    t = np.array([me[0] for me in u_val])
    y = np.array([me[1].diff[0] for me in u_val])
    z = np.array([me[1].alg[0] for me in u_val])

    my_setup_mpl()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(t, y)
    axs[1].plot(t, z)

    for ax in axs:
        ax.set_xlabel(r"$t$")

    axs[0].set_ylabel(r"$y$")
    axs[1].set_ylabel(r"$z$")

    filename = "data" + "/" + f"{problem_name}" + "/" + f"solution.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    plot_numerical_solution("LINEAR-TEST")
