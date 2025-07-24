import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.projects.DAE import compute_solution, my_setup_mpl

from pySDC.helpers.stats_helper import get_sorted


def plot_numerical_solution(problem_name, dt=1e-2, num_nodes=3, problem_type="constrainedDAE", QI="LU", Tend=1.0):
    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    my_setup_mpl()

    u_val = get_sorted(solution_stats, type="u", sortby="time")

    t = np.array([me[0] for me in u_val])
    if problem_name == "ANDREWS-SQUEEZER":
        fig, axs = plt.subplots(1, 1, figsize=(20, 10))

        q1 = np.array([me[1].diff[0] for me in u_val])
        q2 = np.array([me[1].diff[1] for me in u_val])
        q3 = np.array([me[1].diff[2] for me in u_val])
        q4 = np.array([me[1].diff[3] for me in u_val])
        q5 = np.array([me[1].diff[4] for me in u_val])
        q6 = np.array([me[1].diff[5] for me in u_val])
        q7 = np.array([me[1].diff[6] for me in u_val])

        # Transform solution to get plot as in Hairer & Wanner (1996)
        q1 = ((q1 + np.pi) % (2 * np.pi)) - np.pi
        q2 = ((q2 + np.pi) % (2 * np.pi)) - np.pi
        q3 = ((q3 + np.pi) % (2 * np.pi)) - np.pi
        q4 = ((q4 + np.pi) % (2 * np.pi)) - np.pi
        q5 = ((q5 + np.pi) % (2 * np.pi)) - np.pi
        q6 = ((q6 + np.pi) % (2 * np.pi)) - np.pi
        q7 = ((q7 + np.pi) % (2 * np.pi)) - np.pi

        axs.plot(t, q1, label=r"$\beta$")  # label=r"$q_1$")
        axs.plot(t, q2, label=r"$\Theta$")  # label=r"$q_2$")
        axs.plot(t, q3, label=r"$\gamma$")  # label=r"$q_3$")
        axs.plot(t, q4, label=r"$\Phi$")  # label=r"$q_4$")
        axs.plot(t, q5, label=r"$\delta$")  # label=r"$q_5$")
        axs.plot(t, q6, label=r"$\Omega$")  # label=r"$q_6$")
        axs.plot(t, q7, label=r"$\varepsilon$")

        axs.set_xlabel(r'$t$')
        axs.set_ylabel(r'Solution')

        axs.set_xlim((0.0, 0.03))
        # axs.set_ylim((-0.7, 0.7))
        axs.set_ylim((-4.0, 4.0))

        axs.legend(loc='upper right')

    elif problem_name == "LINEAR-TEST":
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        y = np.array([me[1].diff[0] for me in u_val])
        z = np.array([me[1].alg[0] for me in u_val])

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
    # plot_numerical_solution("LINEAR-TEST")
    plot_numerical_solution(
        "ANDREWS-SQUEEZER",
        dt=1e-5,
        num_nodes=6,
        problem_type="fullyImplicitDAE",
        QI="RadauIIA7",
        Tend=0.03,
    )
