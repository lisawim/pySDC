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
import pySDC.helpers.plot_helper as plt_helper


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


def main():
    """
    Main function that executes all the stuff containing:
        - plotting the solution for one single time step size,
        - plotting the differences around a discrete event (i.e., the differences at the time before, at, and after the event)
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    hookclass = approx_solution_hook

    nvars = 13
    problem_class = BuckConverterDAE
    V_refmax = 8

    sweeper = fully_implicit_DAE
    newton_tol = 1e-7

    use_detection = [False, True]

    t0 = 0.0
    Tend = 6.1

    dt_list = [1e-2]

    results_dt = dict()
    switching_dt = dict()
    results_simulations = dict()
    results_events = dict()

    for use_SE in use_detection:
        for dt_item in dt_list:
            print(f'Controller run -- Simulation for step size: {dt_item}')

            if use_SE:
                restol = -1
                recomputed = False
            else:
                restol = 1e-12
                recomputed = None

            description, controller_params = get_description(dt_item, nvars, problem_class, hookclass, sweeper, use_SE, restol, newton_tol)

            description['problem_params']['V_refmax'] = V_refmax

            stats, nfev = controller_run(t0, Tend, controller_params, description)

            vC2 = np.array([me[1][4] for me in get_sorted(stats, type='approx_solution', recomputed=recomputed)])
            t = np.array([me[0] for me in get_sorted(stats, type='approx_solution', recomputed=recomputed)])

            results_dt[dt_item] = pack_solution_data(t, vC2)
            res_array = pack_solution_data(t, vC2)

            t_switches = (
                np.array([me[1] for me in get_recomputed(stats, type='switch', sortby='time')])
                if use_SE
                else np.zeros(1)
            )
            switching_dt[dt_item] = t_switches
            if use_SE:
                print('use_SE:', use_SE)
                for m in range(res_array.shape[1]):
                    if np.isclose(t_switches[-1], res_array[0, m]):
                        for l in range(m - 3, m + 3):
                            print(t_switches[-1], res_array[0, l], V_refmax - res_array[1, l])
            else:
                print('use_SE:', use_SE)
                for m in range(1, res_array.shape[1]):
                    t_switch = 3.2400029447686314
                    if res_array[0, m - 1] <= t_switch <= res_array[0, m]:
                        for l in range(m - 3, m + 3):
                            print(t_switch, res_array[0, l], V_refmax - res_array[1, l])

        results_simulations[use_SE] = results_dt
        results_events[use_SE] = switching_dt

    diffs_over_time(V_refmax, dt_list, use_detection, results_simulations, results_events)


if __name__ == "__main__":
    main()
