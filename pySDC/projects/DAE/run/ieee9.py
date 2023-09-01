from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.problems.IEEE9BusSystem import IEEE9BusSystem

from pySDC.projects.PinTSimE.battery_model import generate_description, get_recomputed, controller_run
from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

from pySDC.core.Hooks import hooks
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_restarts import LogRestarts


class LogEvent(hooks):
    """
    Logs the data for the discontinuous test DAE problem containing one discrete event.
    Note that this logging data is dependent from the problem itself.
    """

    def post_step(self, step, level_number):
        super(LogEvent, self).post_step(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='state_function',
            value=L.uend[P.m] - P.psv_max,
        )

class LogWork(hooks):
    """
    Logs the number of Newton iterations in the fully-implicit DAE sweeper, and the number of function evaluations
    of the problem class.
    """

    def post_step(self, step, level_number):
        super(LogWork, self).post_step(step, level_number)

        L = step.levels[level_number]
        P = L.prob

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='nfev_inner_solve',
            value=L.sweep.inner_solver_count,
        )


def main():
    """
    Function that executes the main stuff in this file.
    """

    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/ieee9").mkdir(parents=True, exist_ok=True)

    hookclass = [LogSolution, LogEvent, LogRestarts, LogWork]

    problem_class = IEEE9BusSystem

    sweeper = fully_implicit_DAE
    nnodes = [2, 3, 4, 5]
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    maxiter = [60]  # [1, 2, 3, 4, 5, 6, 7]
    newton_tolerances = [1e-5]  # [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

    use_detection = [False]  # [False]
    max_restarts = 400
    tolerances_event = [1e-10]  # [1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    alphas = [1.0]  # [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    t0 = 0.0
    Tend = 0.7

    dt_list = [1e-3]  # [1 / (2 ** m) for m in range(4, 11)]

    res_norm_against_newton_tol = dict()
    res_norm_against_newton_tol_M = dict()

    res_time = dict()

    for iter in maxiter:
        for num_nodes in nnodes:
            for dt in dt_list:
                for use_SE in use_detection:
                    for newton_tol in newton_tolerances:
                        for tol_event in tolerances_event:
                            for alpha in alphas:
                                print(f'Controller run -- Simulation for step size: {dt} -- M={num_nodes} -- newton_tol={newton_tol}')
                                print(tol_event)
                                problem_params = dict()
                                problem_params['newton_tol'] = newton_tol

                                restol = 2e-13
                                recomputed = False if use_SE else None

                                N_dt = int((Tend - t0) / dt)  # !!!

                                description, controller_params = generate_description(
                                    dt,
                                    problem_class,
                                    sweeper,
                                    num_nodes,
                                    quad_type,
                                    QI,
                                    hookclass,
                                    False,
                                    use_SE,
                                    problem_params,
                                    restol,
                                    iter,
                                    max_restarts,
                                    tol_event,
                                    alpha,
                                )

                                stats = controller_run(description, controller_params, False, use_SE, t0, Tend)

                                plot_solution(stats, recomputed, use_SE)

                                # ---- mean number of function evals for inner solve ----
                                nfev_counts = get_sorted(stats, type='nfev_inner_solve', sortby='time', recomputed=recomputed)
                                t_nfev, nfev = [me[0] for me in nfev_counts], [me[1] for me in nfev_counts]

                                N_dt = 0  # only count subintervals with nfev > 0
                                for m in range(len(nfev)):
                                    if nfev[m] != 0:
                                        N_dt += 1
                                nfev_mean = round(sum(nfev) / N_dt)

                                # ---- residual after time step ----
                                residual_post_step = get_sorted(stats, type='residual_post_step', sortby='time', recomputed=recomputed)
                                res_norm = max([me[1] for me in residual_post_step])
                                res_norm_against_newton_tol[newton_tol] = [res_norm, nfev_mean]

                                res_time[num_nodes] = residual_post_step

                                # ---- number of iterations ----
                                iter_counts = get_sorted(stats, type='niter', sortby='time')
                                for item in iter_counts:
                                    out = 'Number of iterations at time %4.2f: %2i' % item
                                    # print(out)

            res_norm_against_newton_tol_M[num_nodes] = res_norm_against_newton_tol
            res_norm_against_newton_tol = dict()

    plot_tolerances_against_errors_or_residuals(
        dt_list[0],
        'data/ieee9/ieee9_residual_against_tolerances_dt{}.png'.format(dt_list[0]),
        res_norm_against_newton_tol_M,
        r'$||r_\mathrm{DAE}||_\infty$',
        [1e-15, 1e-10],
    )

    plot_residuals_over_time(res_time, 'data/ieee9/ieee9_residuals_over_time_dt{}.png'.format(dt_list[0]))


def plot_solution(stats, recomputed, use_detection, cwd='./'):
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

    m = 3
    n = 9
    PSV = np.array([me[1][10*m:11*m] for me in get_sorted(stats, type='u', sortby='time')])
    t = np.array([me[0] for me in get_sorted(stats, type='u', recomputed=recomputed)])

    if use_detection:
        switches = get_recomputed(stats, type='switch', sortby='time')
        # assert len(switches) >= 1, 'No switches found!'
        t_switches = [v[1] for v in switches]
        print('Founded switches: {}'.format(t_switches))

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    ax.plot(t, PSV[:, 0], label='PSV_gen0')
    ax.plot(t, PSV[:, 1], label='PSV_gen1')
    ax.plot(t, PSV[:, 2], label='PSV_gen2')
    if use_detection:
        for m in range(len(t_switches)):
            if m == 0:
                ax.axvline(
                    x=t_switches[m],
                    linestyle='--',
                    linewidth=0.9,
                    color='g',
                    label='{} Event(s) found'.format(len(t_switches)),
                )
            else:
                ax.axvline(x=t_switches[m], linestyle='--', linewidth=0.9, color='g')

    ax.legend(frameon=False, fontsize=14, loc='upper left')
    ax.set_xlabel(r'$t$', fontsize=16)
    ax.set_ylabel(r'$PSV_0(t)$, $PSV_1(t)$, $PSV_2(t)$', fontsize=16)
    fig.savefig('data/ieee9/IEEE9_PSV_solution.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_tolerances_against_errors_or_residuals(dt, file_name, results_dict, label_y_axis, ylim):
    """
    Plots different (newton) tolerances against the error norm, or the residual norm.

    Parameters
    ----------
    dt : float
        Time step size used for the simulation(s).
    file_name : str
        Name of the file to be stored.
    results_dict : dict
        Contains for different number of collocation nodes as keys a dictionary. This dictionary has the different
        newton tolerances as key. The values are the error_norm (or the residual norm) and number of function evaluations
        or (number of Newton iterations).

        ```
        results_dict[num_nodes][dt] = [err_norm, nfev]
        ```

    label_y_axis : str
        Label for the values on y-axis.
    y_lim : list
        Contains the limits for the y-axis.
    """

    colors = {
        2: 'limegreen',
        3: 'firebrick',
        4: 'deepskyblue',
        5: 'purple',
    }
    linestyles = {
        2: 'solid',
        3: 'dashed',
        4: 'dashdot',
        5: 'dotted',
    }
    markers = {
        2: 's',
        3: 'o',
        4: '*',
        5: 'd',
    }
    xytext = {
        2: (-13.0, -7),
        3: (-13.0, 8),
        4: (-13.0, -17),
        5: (-13.0, 10),
    }
    count = 0
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for num_nodes in results_dict.keys():
        lists = sorted(results_dict[num_nodes].items())
        tols, norms_list = zip(*lists)
        norms, niters = [me[0] for me in norms_list], [me[1] for me in norms_list]

        ax.loglog(
            tols,
            norms,
            color=colors[num_nodes],
            linestyle=linestyles[num_nodes],
            marker=markers[num_nodes],
            linewidth=1.1,
            label=r'$M={}$'.format(num_nodes),
        )
        
        for m in range(len(tols)):
            ax.annotate(
                "({})".format(niters[m]),
                (tols[m], norms[m]),
                xytext=xytext[num_nodes],
                textcoords="offset points",
                color=colors[num_nodes],
                fontsize=12,
            )

        count += 1

    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.set_ylim(ylim[0], ylim[-1])
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    # ax.set_xlabel(r'$tol_{newton}$', fontsize=14)
    ax.set_xlabel(r'$tol_{hybr}$', fontsize=21)
    ax.set_ylabel(label_y_axis, fontsize=21)
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=16, loc='lower right')
    ax.minorticks_off()
    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_residuals_over_time(res_time, file_name):
    """
    Plots the error and residual over time for different number of collocation nodes.

    Parameters
    ----------
    err_time : dict
        Contains the stats regarding error for different M.
    res_time : dict
        Contains the stats regarding residual for different M.
    y_label_err : str
        Label for y-label axis for error.
    file_name : str
        Name of the file to be stored.
    """

    colors_err = {
        2: 'limegreen',
        3: 'firebrick',
        4: 'deepskyblue',
        5: 'purple',
    }

    colors_res = {
        2: 'forestgreen',
        3: 'tomato',
        4: 'royalblue',
        5: 'plum',
    }

    linestyles = {
        2: 'solid',
        3: 'dashed',
        4: 'dashdot',
        5: 'dotted',
    }

    lines = []
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for num_nodes in res_time.keys():
        res_vals = res_time[num_nodes]
        t, res = [me[0] for me in res_vals], [me[1] for me in res_vals]

        res_label = r'Residual M={}'.format(num_nodes)

        ax.plot(t, res, color=colors_res[num_nodes], linestyle=linestyles[num_nodes], linewidth=1.1, label=res_label)

        lines.append(res_label)

    ax.tick_params(axis='both', which='major', labelsize=21)
    ax.set_ylim(1e-15, 1e+1)
    ax.set_yscale('log', base=10)
    ax.set_xlabel('t', fontsize=21)
    ax.set_ylabel(r'$r_\mathrm{DAE}$', fontsize=21)
    ax.minorticks_off()
    ax.grid(visible=True)
    labels = [l for l in lines]
    ax.legend(lines, frameon=True, fontsize=16, loc='upper left')
    ax.minorticks_off()

    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    main()

