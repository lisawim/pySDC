import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE.run.utils import compute_solution, my_setup_mpl

from pySDC.projects.DAE.problems.discontinuousTestDAE import LogEventDiscontinuousTestDAE


def plot_state_function_over_time(format="png"):
    problem_name="DISC-TEST"

    sweeper_type = "constrainedDAE"
    QI_list = ["EE", "IE", "LU", "MIN-SR-S", "MIN-SR-NS", "Picard"]
    num_nodes = 6

    t0 = 1.0
    Tend = 5.0
    n_steps_list = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    dt_list = [(Tend - t0) / n_steps for n_steps in n_steps_list]

    hook_class = [LogEventDiscontinuousTestDAE]

    my_setup_mpl(fontsize=7)
    for QI in QI_list:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for dt in dt_list:
            solution_stats = compute_solution(
                problem_name,
                t0,
                dt,
                Tend,
                num_nodes,
                QI,
                sweeper_type,
                False,
                hook_class=hook_class,
                measure=False,
            )

            state_function_stats = get_sorted(solution_stats, type="abs_state_function", sortby="time", recomputed=False)
            t = np.array([me[0] for me in state_function_stats])
            state_function = np.array([me[1] for me in state_function_stats])

            ax.semilogy(t, state_function, label=rf"$\Delta t = ${dt}")

        ax.tick_params(axis="both", which="minor", bottom=False, left=False)
        ax.set_xlabel(r"time $t$")
        ax.set_xlabel(r"absolute value of state function")
        ax.grid(linewidth=0.5)

        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)

        plot_name = f"state_function_time_{QI}_{num_nodes=}"
        filename = "data" + "/" + f"{problem_name}" + "/" + plot_name + "." + format
        file_path = Path(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    plot_state_function_over_time()
