import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.projects.DAE.problems.reactionDiffusionPDAE import (
    ReactionDiffusionPDAEConstrained,
)
from pySDC.projects.DAE import compute_solution, my_setup_mpl
from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal


def plot_solution_linear(
    dt: float = 1e-2,
    filename: str = "solution",
    journal: str = "Springer_Scientific_Computing",
    num_nodes: int = 3,
    problem_type: str = "constrainedDAE",
    QI: str = "LU",
    Tend: float = 1.0,
):
    problem_name = "LINEAR-TEST"

    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    print(f"\n ... Generating solution of {problem_name} ... \n")

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    my_setup_mpl(fontsize=6)
    figsize = figsize_by_journal(journal, scale=0.62, ratio=0.68)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t = np.array([me[0] for me in u_val])

    y = np.array([me[1].diff[0] for me in u_val])
    z = np.array([me[1].alg[0] for me in u_val])

    ax.plot(t, y, label=r"$y$ (differential variable)")
    ax.plot(t, z, label=r"$z$ (algebraic variable)")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"solution $u$")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_solution_linear_embedded(
    dt: float = 1e-2,
    filename: str ="linear_test_embedded_solution",
    eps_list: list = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0],
    journal: str = "BUW_thesis",
    num_nodes: int = 3,
    problem_type: str = "constrainedDAE",
    QI: str = "LU",
    Tend: float = 1.0,
):
    colors = [
        'mistyrose',
        'lightsalmon',
        'lightcoral',
        'indianred',
        'firebrick',
        'brown',
        'maroon',
        'black',
    ]
    problem_name = "LINEAR-TEST"

    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    my_setup_mpl(fontsize=7)
    figsize = figsize_by_journal(journal, scale=0.75, ratio=0.5)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for e, eps in enumerate(eps_list):
        print(f"\n ... Generating solution of {problem_name} for {eps=}... \n")

        solution_stats = compute_solution(
            problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False, eps=eps
        )

        u_val = get_sorted(solution_stats, type="u", sortby="time")
        t = np.array([me[0] for me in u_val])
        u = np.array([me[1].flatten() for me in u_val])

        y = u[:, 0]
        z = u[:, 1] if eps > 0.0 else u[:, 2]

        axs[0].plot(t, y, color=colors[e], label=rf"$\varepsilon$={eps}")
        axs[1].plot(t, z, color=colors[e])

    for ax in axs:
        ax.set_xlabel(r"time $t$")

    axs[0].set_ylabel(r"solution $y$")
    axs[1].set_ylabel(r"solution $z$")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=4)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_solution_andrews(
    dt: float = 1e-4,
    filename: str = "solution",
    journal: str = "Springer_Scientific_Computing",
    num_nodes: int = 6,
    problem_type: str = "constrainedDAE",
    QI: str = "LU",
    Tend: float = 0.03,
    ax: Optional[Axes] = None,
    return_ax: bool = False,
) -> Optional[Axes]:
    
    problem_name = "ANDREWS-SQUEEZER"

    created_fig = ax is None
    if created_fig:
        my_setup_mpl(fontsize=7)
        figsize = figsize_by_journal(journal, scale=0.69, ratio=0.6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    print(f"\n ... Generating solution of {problem_name} ... \n")

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    u_val = get_sorted(solution_stats, type="u", sortby="time")

    t = np.array([me[0] for me in u_val])

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

    ax.plot(t, q1, label=r"$\beta$")  # label=r"$q_1$")
    ax.plot(t, q2, label=r"$\Theta$")  # label=r"$q_2$")
    ax.plot(t, q3, label=r"$\gamma$")  # label=r"$q_3$")
    ax.plot(t, q4, label=r"$\Phi$")  # label=r"$q_4$")
    ax.plot(t, q5, label=r"$\delta$")  # label=r"$q_5$")
    ax.plot(t, q6, label=r"$\Omega$")  # label=r"$q_6$")
    ax.plot(t, q7, label=r"$\varepsilon$")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"solution $q$")

    ax.set_xlim((dt, 0.03))
    ax.set_ylim((-4.0, 4.0))

    if created_fig:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=7)

        out = Path("data") / problem_name / f"{filename}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=400, bbox_inches="tight")

        if not return_ax:
            plt.close(fig)
            return None


def plot_solution_reaction_diffusion(
    dt: float = 1e-1,
    filename: str = "solution_reaction_diffusion",
    journal: str = "BUW_thesis",
    num_nodes: int = 3,
    problem_type: str = "constrainedDAE",
    QI: str = "LU",
    Tend: float = 1.0,
    time_points_to_plot: list = [0.25, 0.5, 0.75, 1.0],
):
    problem_name = "REACTION-DIFFUSION"

    my_setup_mpl(fontsize=8)
    figsize = figsize_by_journal(journal, scale=0.8, ratio=1.0)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    t0 = 0.0
    Tend = Tend

    nvars = 256

    hook_class = [LogSolution]

    print(f"\n ... Generating solution of {problem_name} ... \n")

    solution_stats = compute_solution(
        problem_name,
        t0,
        dt,
        Tend,
        num_nodes,
        QI,
        problem_type,
        hook_class=hook_class,
        measure=False,
        nvars=nvars,
    )

    problem_params = {"nvars": nvars}
    prob = ReactionDiffusionPDAEConstrained(**problem_params)
    xvalues = prob.xvalues

    u_val = get_sorted(solution_stats, type="u", sortby="time")

    t = np.array([me[0] for me in u_val])
    u = np.array([me[1].diff[:nvars] for me in u_val])
    v = np.array([me[1].diff[nvars:] for me in u_val])
    w = np.array([me[1].alg[:nvars] for me in u_val])

    for i, time_point in enumerate(time_points_to_plot):
        ax_flatten[i].set_title(rf"$t={time_point}$")

        idx = np.argmin(np.abs(t - time_point))

        u_at_time_point = u[idx]
        v_at_time_point = v[idx]
        w_at_time_point = w[idx]

        ax_flatten[i].plot(xvalues, u_at_time_point, label=r"$u$")
        ax_flatten[i].plot(xvalues, v_at_time_point, linestyle="dashed", label=r"$v$")
        ax_flatten[i].plot(xvalues, w_at_time_point, label=r"$w$")

    for ax in ax_flatten:
        ax.set_xlabel(r"space $x$")
        ax.set_ylabel(r"solutions $u,v,w$")

        ax.set_ylim((-3.0, 3.0))

    handles, labels = ax_flatten[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_solution_discontinuous_test(
    dt: float = 1e-2,
    filename: str = "solution_discontinuous_test",
    journal: str = "BUW_thesis",
    num_nodes: int = 3,
    problem_type: str = "fullyImplicitDAE",
    QI: str = "LU",
    Tend: float = 5.0,
):
    problem_name = "DISC-TEST"

    t0 = 1.0
    Tend = Tend

    hook_class = [LogSolution]

    print(f"\n ... Generating solution of {problem_name} ... \n")

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    my_setup_mpl(fontsize=6)
    figsize = figsize_by_journal(journal, scale=0.62, ratio=0.68)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t = np.array([me[0] for me in u_val])

    y = np.array([me[1].diff[0] for me in u_val])
    z = np.array([me[1].alg[0] for me in u_val])

    ax.plot(t, y, color="black", label=r"$y$ (differential variable)")
    ax.plot(t, z, color="gray", linestyle="dashed", label=r"$z$ (algebraic variable)")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"solutions $y,z$")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_solution_wscc9(
    dt: float = 1e-2,
    filename: str = "solution_wscc9",
    journal: str = "BUW_thesis",
    num_nodes: int = 3,
    problem_type: str = "fullyImplicitDAE",
    QI: str = "LU",
    Tend: float = 0.7,
):
    """
    Plots the solution of one simulation run (i.e., for one time step size).

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    recomputed : bool
        Indicate that values after restart should be used.
    use_detection : bool
        Indicate whether switch detection should be used or not.
    t_switch_exact : float
        Exact event time.
    cwd : str
        Current working directory.
    """

    problem_name = "WSCC9"

    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    print(f"\n ... Generating solution of {problem_name} ... \n")

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    my_setup_mpl(fontsize=6)
    figsize = figsize_by_journal(journal, scale=0.62, ratio=0.68)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    m = 3
    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t = np.array([me[0] for me in u_val])
    PSV = np.array([me[1].diff[10 * m : 11 * m] for me in u_val])

    ax.plot(t, PSV[:, 0], color="darkblue", label=r"$PSV_{gen_0}$")
    ax.plot(t, PSV[:, 1], color="royalblue", label=r"$PSV_{gen_1}$")
    ax.plot(t, PSV[:, 2], color="lightblue", label=r"$PSV_{gen_2}$")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"solution $PSV$")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_solution_battery(
    dt: float = 5e-2,
    filename: str = "battery_solution",
    journal: str = "Springer_Scientific_Computing",
    num_nodes: int = 3,
    problem_type: str = "fullyImplicitDAE",
    QI: str = "LU",
    Tend: float = 3.5,
):
    problem_name = "BATTERY"

    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    print(f"\n ... Generating solution of {problem_name} ... \n")

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    my_setup_mpl(fontsize=6)
    figsize = figsize_by_journal(journal, scale=0.62, ratio=0.68)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t = np.array([me[0] for me in u_val])

    iL = np.array([me[1].diff[0] for me in u_val])
    VC = np.array([me[1].diff[1] for me in u_val])

    ax.plot(t, iL, color="royalblue", label=r"$i_L$")
    ax.plot(t, VC, color="firebrick", label=r"$V_C$")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"solutions $i_L, V_C$")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_solution_buck_converter(
    dt: float = 5e-5,
    filename: str = "buck_converter_solution",
    journal: str = "Springer_Scientific_Computing",
    num_nodes: int = 3,
    problem_type: str = "fullyImplicitDAE",
    QI: str = "LU",
    Tend: float = 0.02,
):
    problem_name = "BUCK-CONVERTER"

    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    print(f"\n ... Generating solution of {problem_name} ... \n")

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    my_setup_mpl(fontsize=6)
    figsize = figsize_by_journal(journal, scale=0.62, ratio=0.68)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t = np.array([me[0] for me in u_val])

    iLpi = np.array([me[1].diff[0] for me in u_val])
    VC1 = np.array([me[1].diff[1] for me in u_val])
    VC2 = np.array([me[1].diff[2] for me in u_val])

    ax.plot(t, iLpi, color="royalblue", label=r"$i_{L_\pi}$")
    ax.plot(t, VC1, color="firebrick", label=r"$V_{C_1}$")
    ax.plot(t, VC2, color="red", label=r"$V_{C_2}$")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"solutions $i_{L_\pi}, V_{C_1}, V_{C_2}$")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_solution_piline(
    dt: float = 5e-2,
    filename: str = "piline_solution",
    journal: str = "Springer_Scientific_Computing",
    num_nodes: int = 3,
    problem_type: str = "fullyImplicitDAE",
    QI: str = "LU",
    Tend: float = 10.0,
):
    problem_name = "PILINE"

    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    print(f"\n ... Generating solution of {problem_name} ... \n")

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    my_setup_mpl(fontsize=6)
    figsize = figsize_by_journal(journal, scale=0.62, ratio=0.68)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t = np.array([me[0] for me in u_val])

    iLpi = np.array([me[1].diff[0] for me in u_val])
    VC1 = np.array([me[1].diff[1] for me in u_val])
    VC2 = np.array([me[1].diff[2] for me in u_val])

    ax.plot(t, iLpi, color="royalblue", label=r"$i_{L_\pi}$")
    ax.plot(t, VC1, color="firebrick", label=r"$V_{C_1}$")
    ax.plot(t, VC2, color="red", label=r"$V_{C_2}$")

    ax.set_xlabel(r"time $t$")
    ax.set_ylabel(r"solutions $i_{L_\pi}, V_{C_1}, V_{C_2}$")

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)