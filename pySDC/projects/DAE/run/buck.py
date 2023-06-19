from pathlib import Path
import numpy as np
import pickle

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI
from pySDC.projects.DAE.run.piline import get_description
from pySDC.projects.DAE.problems.buck_dae import BuckConverterDAE
from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
from pySDC.projects.DAE.run.piline import get_description, controller_run, pack_solution_data
from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.projects.PinTSimE.battery_model import get_recomputed
from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Hooks import hooks
import pySDC.helpers.plot_helper as plt_helper


class event_data(hooks):
    """
    Hook class to add the state function of a discontinuous differential-algebraic equation after each step.
    Data of the state functions are problem dependent.
    """

    def __init__(self):
        """Initialization routine"""
        super(event_data, self).__init__()

    def post_step(self, step, level_number):
        """
        Default routine called after each step.

        Parameters
        ----------
        step : pySDC.core.Step
            Current step.
        level_number : pySDC.core.Level
            Current level number.
        """

        super(event_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]
        P = L.prob

        # TODO: is it really necessary to recompute the end point? Hasn't this been done already?
        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='state_function',
            value=P.V_ref - L.uend[1],
        )


def main():
    """
    Main function that executes all the stuff containing:
        - plotting the solution for one single time step size,
        - plotting the differences around a discrete event (i.e., the differences at the time before, at, and after the event)
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = [approx_solution_hook, event_data]

    nvars = 13
    problem_class = BuckConverterDAE
    V_ref = 4

    sweeper = fully_implicit_DAE
    nnodes = [2, 3, 5]
    newton_tol = 1e-3 #[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    maxiter = 7

    use_detection = [True] #[False, True]

    t0 = 0.0
    Tend = 0.20 #0.1324 #6.1

    dt_list = [5e-3]
    N_dt = int(Tend / dt_list[0])
    fac_tol_SE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] #[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    results_dt = dict()
    switching_dt = dict()
    results_simulations = dict()
    results_events = dict()

    res_norm_against_newton_tol = dict()
    tol_SE_against_switches = dict()
    tol_SE_against_switches_M = dict()
    #for newton_tol in newton_tolerances:
    for num_nodes in nnodes:
        for m in range(len(fac_tol_SE)):
            for use_SE in use_detection:
                for dt_item in dt_list:
                    print(f'Controller run -- Simulation for step size: {dt_item}')

                    tol_SE = fac_tol_SE[m] * dt_item
                    restol = -1 if use_SE else 1e-12
                    recomputed = False if use_SE else None

                    description, controller_params = get_description(
                        dt_item,
                        nvars,
                        problem_class,
                        hookclass,
                        sweeper,
                        num_nodes,
                        use_SE,
                        restol,
                        tol_SE,
                        maxiter,
                        newton_tol,
                    )

                    description['problem_params']['V_ref'] = V_ref

                    stats, nfev = controller_run(t0, Tend, controller_params, description)

                    #plot_solution(stats, use_SE)

                    vC1 = np.array([me[1][1] for me in get_sorted(stats, type='approx_solution', recomputed=recomputed)])
                    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution', recomputed=recomputed)])

                    results_dt[dt_item] = pack_solution_data(t, vC1)
                    res_array = pack_solution_data(t, vC1)

                    t_switches = (
                        np.array([me[1] for me in get_recomputed(stats, type='switch', sortby='time')])
                        if use_SE
                        else np.zeros(1)
                    )
                    switching_dt[dt_item] = t_switches

                    nfev = round(nfev / N_dt)  # average of nfev across all time steps
                    residual_post_step = get_sorted(stats, type='residual_post_step', sortby='time', recomputed=False)
                    res_norm = max([me[1] for me in residual_post_step])
                    res_norm_against_newton_tol[newton_tol] = [res_norm, nfev]

                    tol_SE_against_switches[tol_SE] = len(t_switches)

                results_simulations[use_SE] = results_dt
                results_events[use_SE] = switching_dt

            #diffs_over_time(V_ref, dt_list, use_detection, results_simulations, results_events)

        #plot_nfev_against_residual(res_norm_against_newton_tol)

        tol_SE_against_switches_M[num_nodes] = tol_SE_against_switches

    plot_tol_SE_against_switches(dt_list[0], tol_SE_against_switches_M, nnodes)


def plot_solution(stats, use_detection=False, cwd='./'):
    """
    Plots the solution of one simulation run (i.e., for one time step size).

    Parameters
    ----------
    stats : dict
        Raw statistics from a controller run.
    use_detection : bool, optional
        Indicate whether switch detection should be used or not.
    cwd : str
        Current working directory.
    """

    vC1 = np.array([me[1][1] for me in get_sorted(stats, type='approx_solution')])
    vC2 = np.array([me[1][4] for me in get_sorted(stats, type='approx_solution')])
    iLp = np.array([me[1][9] for me in get_sorted(stats, type='approx_solution')])
    t = np.array([me[0] for me in get_sorted(stats, type='approx_solution')])

    if use_detection:
        switches = get_recomputed(stats, type='switch', sortby='time')
        assert len(switches) >= 1, 'No switches found!'
        t_switch = [v[1] for v in switches]

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title(r'Solution /w switch detection', fontsize=8)
    ax.plot(t, vC1, label=r"$v_\mathrm{C_1}$")
    ax.plot(t, vC2, label=r"$v_\mathrm{C_2}$")
    ax.plot(t, iLp, label=r"$i_\mathrm{L_\pi}$")
    if use_detection:
        for m in range(len(t_switch)):
            if m == 0:
                ax.axvline(x=t_switch[m], linestyle='--', linewidth=0.8, color='g', label='{} Event(s) found'.format(len(t_switch)))
            else:
                ax.axvline(x=t_switch[m], linestyle='--', linewidth=0.8, color='g')
    ax.legend(frameon=False, fontsize=8, loc='lower right')

    ax.set_xlabel('Time[s]', fontsize=8)
    ax.set_ylabel('Voltage[V]', fontsize=8)
    fig.savefig('data/buck_solution.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def diffs_over_time(V_refmax, dt_list, use_detection, results_simulations, results_events):
    """
    Plots the state function over time. It can be investigated how large the error is.

    Parameters
    ----------
    V_refmax : float
        Value at which the switching states change (used to compute difference at event).
    dt_list : list
        Contains multiple time step sizes.
    use_detection : list
        Contains the iterable object for indicating whether a detection of events is used.
    results_simulations : dict
        Results of the solution for each time step size.
    results_events : dict
        Switching results for each time step size.
    """

    for dt_item in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(5, 5))
        ax.set_title(r'Evaluating state function over time for $\Delta t=%s$' % dt_item, fontsize=8)
        for use_SE in use_detection:
            u = results_simulations[use_SE][dt_item]

            t_switches = results_events[use_SE][dt_item]
            t_switch = t_switches[-1]

            if use_SE:
                ax.plot(u[0, :], V_refmax - u[1, :], 'r--', linewidth=0.8, label=r'Detection - {}'.format(use_SE))
                #ax.plot(u[0, :], u[1, :], 'r--', linewidth=0.8, label=r'Detection - {}'.format(use_SE))
            else:
                ax.plot(u[0, :], V_refmax - u[1, :], 'k-', linewidth=0.8, label=r'Detection - {}'.format(use_SE))
                #ax.plot(u[0, :], u[1, :], 'k-', linewidth=0.8, label=r'Detection - {}'.format(use_SE))


        for m in range(len(t_switches)):
            if m == 0:
                ax.axvline(x=t_switches[m], linestyle='--', linewidth=0.8, color='g', label='{} Event(s) found'.format(len(t_switches)))
            else:
                ax.axvline(x=t_switches[m], linestyle='--', linewidth=0.8, color='g')

        ax.legend(frameon=False, fontsize=8, loc='lower right')

        #ax.set_xlim(3.462769188733114-0.001, 3.462769188733114+0.001)
        ax.set_ylim(-1, 1)
        ax.set_yscale('symlog', linthresh=1e-11)
        ax.set_xlabel(r'Time[s]', fontsize=8)
        ax.set_ylabel(r'$V_\mathrm{refmax} - V_\mathrm{C_2}$', fontsize=8)

        fig.savefig('data/buck_diffs_over_time_{}.png'.format(dt_item), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_nfev_against_residual(res_norm_against_newton_tol):
    """
    Plots different newtol's against the residual (max) norm, where the residuals after one step is considered.

    Parameters
    ----------
    res_norm_against_newton_tol : dict
        Contains the residual norm and corresponding number of function evaluations for one specific newton_tol.
    """


    lists = sorted(res_norm_against_newton_tol.items())
    newton_tols, res_norm_against_newton_tol_list = zip(*lists)
    res_norms, nfev = [me[0] for me in res_norm_against_newton_tol_list], [me[1] for me in res_norm_against_newton_tol_list]

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(5, 3))
    ax.loglog(newton_tols, res_norms, color='firebrick', marker='.', linewidth=0.8)
    for m in range(len(newton_tols)):
        ax.annotate(nfev[m], (newton_tols[m], res_norms[m]), xytext=(-8.5, 10), textcoords="offset points", fontsize=8)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_ylim(1e-16, 1e+1)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Inner tolerance', fontsize=8)
    ax.set_ylabel(r'$||r_\mathrm{DAE}||_\infty$', fontsize=8)
    ax.grid(visible=True)
    ax.minorticks_off()
    fig.savefig('data/buck_residual_against_tolerances.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_tol_SE_against_switches(dt, tol_SE_against_switches_M, nnodes):
    """
    Plots the different tolerances for switch estimation against the number of switches found.

    Parameters
    ----------
    tol_SE_against_switches : dict
        Contains the number of switches for each tolerance considered.
    """

    colors = {2: 'limegreen', 3: 'firebrick', 5:'deepskyblue'}
    linestyles = {2: 'solid', 3: 'dashed', 5: 'dashdot'}
    markers = {2: 's', 3: 'o', 5: '*'}
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(5, 3))
    ax.set_title(r'Tolerances for switch detection using RAUDAU-RIGHT with $\Delta t={}$'.format(dt))
    for num_nodes in nnodes:
        lists = sorted(tol_SE_against_switches_M[num_nodes].items())
        tol_SE, nswitches = zip(*lists)
        ax.plot(
            tol_SE,
            nswitches,
            color=colors[num_nodes],
            linestyle=linestyles[num_nodes],
            marker=markers[num_nodes],
            linewidth=0.8,
            label=r'$M={}$'.format(num_nodes)
        )
        for m in range(len(nswitches)):
            ax.annotate(nswitches[m], (tol_SE[m], nswitches[m]), xytext=(-8.5, 10), textcoords="offset points", fontsize=8)
    ax.set_ylim(0, 40)
    ax.set_xlabel(r'Tolerance for SE', fontsize=8)
    ax.set_ylabel(r'Number of switches found', fontsize=8)
    ax.set_xticks(tol_SE)
    #ax.set_xticklabels([r'$0.7\Delta t$', r'$0.8\Delta t$', r'$0.9\Delta t$', r'$1.0\Delta t$', r'$1.1\Delta t$', r'$1.2\Delta t$', r'$1.3\Delta t$'])
    ax.set_xticklabels([r'$0.1\Delta t$', r'$0.2\Delta t$', r'$0.3\Delta t$', r'$0.4\Delta t$', r'$0.5\Delta t$', r'$0.6\Delta t$'])
    ax.grid(visible=True)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.savefig('data/buck_tol_SE_against_nswitches_dt{}.png'.format(dt), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    main()
