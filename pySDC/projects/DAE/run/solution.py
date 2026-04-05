import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes
from typing import Optional

from pySDC.implementations.hooks.log_solution import LogSolution

from pySDC.projects.DAE import compute_solution, my_setup_mpl
from pySDC.projects.DAE.run.plot_order_iteration import choose_time_step_sizes

from pySDC.projects.DAE.problems.reactionDiffusionPDAE import (
    ReactionDiffusionPDAEConstrained,
    ReactionDiffusionPDAE_Radau,
)

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.plot_helper import figsize_by_journal


def plot_solution_linear(
    problem_name:str = "LINEAR-TEST",
    dt: float = 1e-2,
    num_nodes: int = 3,
    problem_type: str = "constrainedDAE",
    QI: str = "LU",
    Tend: float = 1.0,
    filename: str = "solution",
    journal: str = "Springer_Scientific_Computing"
):
    t0 = 0.0
    Tend = Tend

    hook_class = [LogSolution]

    solution_stats = compute_solution(
        problem_name, t0, dt, Tend, num_nodes, QI, problem_type, hook_class=hook_class, measure=False
    )

    my_setup_mpl(fontsize=7)
    figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    u_val = get_sorted(solution_stats, type="u", sortby="time")
    t = np.array([me[0] for me in u_val])

    y = np.array([me[1].diff[0] for me in u_val])
    z = np.array([me[1].alg[0] for me in u_val])

    axs[0].plot(t, y)
    axs[1].plot(t, z)

    for ax in axs:
        ax.set_xlabel(r"$t$")

    axs[0].set_ylabel(r"$y$")
    axs[1].set_ylabel(r"$z$")

    filename = "data" + "/" + f"{problem_name}" + "/" + f"{filename}.png"
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_solution_andrews(
    problem_name: str = "ANDREWS-SQUEEZER",
    dt: float = 1e-4,
    num_nodes: int = 6,
    problem_type: str = "constrainedDAE",
    QI: str = "LU",
    Tend: float = 0.03,
    filename: str = "solution",
    journal: str = "Springer_Scientific_Computing",
    ax: Optional[Axes] = None,
    return_ax: bool = False,
) -> Optional[Axes]:

    created_fig = ax is None
    if created_fig:
        my_setup_mpl(fontsize=7)
        figsize = figsize_by_journal(journal, scale=0.72, ratio=0.55)
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
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=7)

        out = Path("data") / problem_name / f"{filename}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=400, bbox_inches="tight")

        if not return_ax:
            plt.close(fig)
            return None


def plot_numerical_reaction_diffusion(
    problem_name: str = "REACTION-DIFFUSION",
    dt: float = 1e-2,
    num_nodes: int = 3,
    problem_type: str = "semiImplicitDAE",
    QI: str = "LU",
    Tend: float = 0.25,
    journal: str = "Springer_Scientific_Computing",
):
    my_setup_mpl(fontsize=8)
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)

    t0 = 0.0
    Tend = Tend

    nvars = 256

    hook_class = [LogSolution]
    maxiter = 0

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
        maxiter=maxiter,
    )

    problem_params = {"nvars": nvars}
    prob = ReactionDiffusionPDAEConstrained(**problem_params)

    u_val = get_sorted(solution_stats, type="u", sortby="time")

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    if QI.startswith("RadauIIA"):
        u = np.array([np.fft.irfft(me[1][: prob.Nr], n=prob.N) for me in u_val])
        v = np.array([np.fft.irfft(me[1][prob.Nr : 2 * prob.Nr], n=prob.N) for me in u_val])
        w = np.array([np.fft.irfft(me[1][2 * prob.Nr : 3 * prob.Nr], n=prob.N) for me in u_val])
    else:
        u = np.array([me[1].diff[:nvars] for me in u_val])
        v = np.array([me[1].diff[nvars:] for me in u_val])
        w = np.array([me[1].alg[:nvars] for me in u_val])

    xvalues = prob.xvalues
    ax_flatten[0].plot(xvalues, u[-1], label="numerical solution")
    ax_flatten[1].plot(xvalues, v[-1])
    ax_flatten[2].plot(xvalues, w[-1])

    ax_flatten[0].plot(
        xvalues, prob.u_ex(Tend, x_deriv=0, t_deriv=0), linestyle="dotted", color="black", label="exact solution"
    )
    ax_flatten[1].plot(xvalues, prob.v_ex(Tend, x_deriv=0, t_deriv=0), linestyle="dotted", color="black")
    ax_flatten[2].plot(xvalues, prob.w_ex(Tend, x_deriv=0), linestyle="dotted", color="black")

    for ax in ax_flatten:
        ax.set_xlabel(r'$x$')

    ax_flatten[0].set_ylabel(r'u')
    ax_flatten[1].set_ylabel(r'v')
    ax_flatten[2].set_ylabel(r'w')

    # axs.set_xlim((0.0, 0.03))
    # axs.set_ylim((-0.7, 0.7))
    # axs.set_ylim((-4.0, 4.0))

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    ax_flatten[3].remove()

    filename = (
        "data" + "/" + f"{problem_name}" + "/" + f"solution_{problem_type}_{QI}_{dt=}_{num_nodes=}_{maxiter=}.png"
    )
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_numerical_solution_pre_iteration0_reaction_diffusion(
    dt=1e-2,
    journal="Springer_Scientific_Computing",
    num_nodes=3,
    problem_type="constrainedDAE",
    QI="LU",
    Tend=0.5,
):
    problem_name = "REACTION-DIFFUSION"
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)

    t0 = 0.0
    Tend = Tend

    nvars = 256

    problem_params = {"nvars": nvars}
    prob = ReactionDiffusionPDAE_Radau(**problem_params)

    my_setup_mpl(fontsize=8)

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    ax_flatten = axs.flatten()

    u0 = prob.u_exact(t0)
    # u, v = np.fft.irfft(u0[: prob.Nr], n=prob.N), np.fft.irfft(u0[prob.Nr : 2 * prob.Nr], n=prob.N)
    # w = np.fft.irfft(u0[2 * prob.Nr : 3 * prob.Nr], n=prob.N)

    u_hat, v_hat = u0[: prob.Nr], u0[prob.Nr : 2 * prob.Nr]
    w_hat = u0[2 * prob.Nr : 3 * prob.Nr]

    u, v = np.fft.irfft(u_hat, n=prob.N).real, np.fft.irfft(v_hat, n=prob.N).real
    w = np.fft.irfft(w_hat, n=prob.N).real

    xvalues = prob.xvalues

    ax_flatten[0].plot(xvalues, u, label="numerical solution")
    ax_flatten[1].plot(xvalues, v)
    ax_flatten[2].plot(xvalues, w)

    ax_flatten[0].plot(
        xvalues, prob.u_ex(Tend, x_deriv=0, t_deriv=0), linestyle="dotted", color="black", label="exact solution"
    )
    ax_flatten[1].plot(xvalues, prob.v_ex(Tend, x_deriv=0, t_deriv=0), linestyle="dotted", color="black")
    ax_flatten[2].plot(xvalues, prob.w_ex(Tend, x_deriv=0), linestyle="dotted", color="black")

    for ax in ax_flatten:
        ax.set_xlabel(r'$x$')

    ax_flatten[0].set_ylabel(r'u')
    ax_flatten[1].set_ylabel(r'v')
    ax_flatten[2].set_ylabel(r'w')

    # axs.set_xlim((0.0, 0.03))
    # axs.set_ylim((-0.7, 0.7))
    # axs.set_ylim((-4.0, 4.0))

    handles, labels = ax_flatten[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=3)

    ax_flatten[3].remove()

    filename = (
        "data" + "/" + f"{problem_name}" + "/" + f"solution_pre_iteration0_{problem_type}_{QI}_{dt=}_{num_nodes=}.png"
    )
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close(fig)


def plot_numerical_solution_reaction_diffusion_video(
    dt=1e-2,
    journal="Springer_Scientific_Computing",
    num_nodes=3,
    problem_type="constrainedDAE",
    QI="LU",
):
    problem_name = "REACTION-DIFFUSION-FD"
    figsize = figsize_by_journal(journal, scale=0.7, ratio=0.85)

    t0 = 0.0
    _, Tend = choose_time_step_sizes(problem_name)

    nvars = 256

    hook_class = [LogSolution]
    maxiter = 2 * num_nodes - 1

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
        maxiter=maxiter,
    )

    problem_params = {"nvars": nvars}
    prob = ReactionDiffusionPDAEConstrained(**problem_params)

    my_setup_mpl(fontsize=8)

    u_stats = get_sorted(solution_stats, type="u", sortby="time")
    # wx_stats = get_sorted(solution_stats, type="wx", sortby="time")
    # wxx_stats = get_sorted(solution_stats, type="wxx", sortby="time")

    t = [me[0] for me in u_stats]
    if QI.startswith("RadauIIA"):
        u = np.array([np.fft.irfft(me[1][: prob.Nr], n=prob.N) for me in u_stats])
        v = np.array([np.fft.irfft(me[1][prob.Nr : 2 * prob.Nr], n=prob.N) for me in u_stats])
        w = np.array([np.fft.irfft(me[1][2 * prob.Nr : 3 * prob.Nr], n=prob.N) for me in u_stats])
    else:
        u = np.array([me[1].diff[:nvars] for me in u_stats])
        v = np.array([me[1].diff[nvars:] for me in u_stats])
        w = np.array([me[1].alg[:nvars] for me in u_stats])
        # wx = np.array([me[1] for me in wx_stats])
        # wxx = np.array([me[1] for me in wxx_stats])

    xvalues = prob.xvalues

    fig, ax = plt.subplots(figsize=figsize)
    (line_u,) = ax.plot([], [], label="u(x,t)")
    (line_v,) = ax.plot([], [], linestyle="dashed", label="v(x,t)")
    (line_w,) = ax.plot([], [], label="w(x,t)")
    # line_wx, = ax.plot([], [], label="wx(x,t)")
    # line_wxx, = ax.plot([], [], label="wxx(x,t)")
    ax.set_xlim(0, 1)
    # simpel & robust
    # ymin = np.nanmin([np.nanmin(u), np.nanmin(v), np.nanmin(w)])
    # ymax = np.nanmax([np.nanmax(u), np.nanmax(v), np.nanmax(w)])
    # ax.set_ylim(ymin, ymax)
    ax.set_ylim(np.min([u, v, w]), np.max([u, v, w]))
    # ax.set_ylim(np.min([u,v]), np.max([u,v]))
    ax.set_xlabel("x")
    ax.set_ylabel("Concentration")
    ax.legend(loc="upper right")
    title = ax.set_title("")

    def init():
        line_u.set_data([], [])
        line_v.set_data([], [])
        line_w.set_data([], [])
        # line_wx.set_data([], [])
        # line_wxx.set_data([], [])
        title.set_text("")
        return line_u, line_v, line_w, title

    def animate(i):
        line_u.set_data(xvalues, u[i])
        line_v.set_data(xvalues, v[i])
        line_w.set_data(xvalues, w[i])
        # line_wx.set_data(xvalues, wx[i])
        # line_wxx.set_data(xvalues, wxx[i])
        title.set_text(f"t = {t[i]:.2f}")
        return line_u, line_v, line_w, title

    ani = animation.FuncAnimation(fig, animate, frames=len(t), init_func=init, blit=True, interval=50)

    filename_mp4 = "data" + "/" + f"{problem_name}" + "/" + "solution_time.mp4"
    ani.save(filename_mp4, fps=20)


if __name__ == "__main__":
    # plot_numerical_solution("LINEAR-TEST")
    plot_solution_andrews(
        "ANDREWS-SQUEEZER",
        dt=1e-4,
        num_nodes=6,
        problem_type="constrainedDAE",
        QI="LU",
        Tend=0.03,
    )

    # t0 = 0.0
    # dt = 1e-2
    # plot_numerical_solution_reaction_diffusion(
    #     dt=dt, num_nodes=3, problem_type="fullyImplicitDAE", QI="RadauIIA5", Tend=t0 + dt
    # )
    # plot_numerical_solution_pre_iteration0_reaction_diffusion(
    #     dt=dt, num_nodes=3, problem_type="fullyImplicitDAE", QI="RadauIIA5", Tend=t0+dt
    # )
    # plot_numerical_solution_reaction_diffusion_video(
    #     dt=dt, num_nodes=4, problem_type="fullyImplicitDAE", QI="LU"
    # )
