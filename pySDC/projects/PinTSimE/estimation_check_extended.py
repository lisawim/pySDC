import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery_2Condensators import battery_2condensators
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.PinTSimE.piline_model import setup_mpl
from pySDC.projects.PinTSimE.battery_2condensators_model import log_data, proof_assertions_description
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


def run(dt, problem, restol, sweeper, use_switch_estimator):

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = restol
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C1'] = 1.0
    problem_params['C2'] = 1.0
    problem_params['R'] = 1.0
    problem_params['L'] = 1.0
    problem_params['alpha'] = 5.0
    problem_params['V_ref'] = np.array([1.0, 1.0])  # [V_ref1, V_ref2]
    problem_params['set_switch'] = np.array([False, False], dtype=bool)
    problem_params['t_switch'] = np.zeros(np.shape(problem_params['V_ref'])[0])

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # convergence controllers
    convergence_controllers = dict()
    if use_switch_estimator:
        switch_estimator_params = {}
        convergence_controllers[SwitchEstimator] = switch_estimator_params

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = problem  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = sweeper  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class

    if use_switch_estimator:
        description['convergence_controllers'] = convergence_controllers

    proof_assertions_description(description, problem_params)

    # set time parameters
    t0 = 0.0
    Tend = 3.5

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    Path("data").mkdir(parents=True, exist_ok=True)
    fname = 'data/battery_2condensators_{}.dat'.format(sweeper.__name__)
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', recomputed=None, sortby='time')

    # compute and print statistics
    min_iter = 10
    max_iter = 0

    f = open('data/battery_2condensators_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])

    assert np.mean(niters) <= 12, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    return description, stats


def check(cwd='./'):
    """
    Routine to check the differences between using a switch estimator or not
    """

    dt_list = [1e-1, 1e-2, 1e-3]
    use_switch_estimator = [True, False]
    restarts_all = []
    restarts_dict = dict()

    problem_classes = [battery_2condensators]
    restolerances = [1e-12]
    sweeper_classes = [imex_1st_order]

    for problem, restol, sweeper in zip(problem_classes, restolerances, sweeper_classes):
        Path("data/{}".format(problem.__name__)).mkdir(parents=True, exist_ok=True)
        for dt_item in dt_list:
            for item in use_switch_estimator:
                description, stats = run(
                    dt=dt_item, problem=problem, restol=1e-12, sweeper=sweeper, use_switch_estimator=item
                )

                fname = 'data/battery_2condensators_dt{}_USE{}_{}.dat'.format(dt_item, item, sweeper.__name__)
                f = open(fname, 'wb')
                dill.dump(stats, f)
                f.close()

                if item:
                    restarts_dict[dt_item] = np.array(get_sorted(stats, type='restart', recomputed=None, sortby='time'))
                    restarts = restarts_dict[dt_item][:, 1]
                    restarts_all.append(np.sum(restarts))
                    print("Restarts for dt: ", dt_item, " -- ", np.sum(restarts))

        V_ref = description['problem_params']['V_ref']

        differences_around_switch(dt_list, problem.__name__, restarts_dict, sweeper.__name__, V_ref)

        differences_over_time(dt_list, problem.__name__, sweeper.__name__, V_ref)

        iterations_over_time(dt_list, description['step_params']['maxiter'], problem.__name__, sweeper.__name__)

        restarts_all = []


def differences_around_switch(dt_list, problem, restarts_dict, sweeper, V_ref, cwd='./'):
    """
    Routine to plot the differences before, at, and after the switch. Produces the
    diffs_estimation_<problem_classes>_<sweeper_class>.png file
    """

    diffs_true_switch1 = []
    diffs_false_before_switch1 = []
    diffs_false_after_switch1 = []
    diffs_true_switch2 = []
    diffs_false_before_switch2 = []
    diffs_false_after_switch2 = []
    restarts_switch1 = []
    restarts_switch2 = []

    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_2condensators_dt{}_USETrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_2condensators_dt{}_USEFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        val_switch1 = get_sorted(stats_true, type='switch1', sortby='time')
        val_switch2 = get_sorted(stats_true, type='switch2', sortby='time')

        t_switch1, t_switch2 = [v[0] for v in val_switch1], [v[0] for v in val_switch2]

        t_switch1, t_switch2 = t_switch1[0], t_switch2[0]

        vC1_true = get_sorted(stats_true, type='voltage C1', recomputed=False, sortby='time')
        vC2_true = get_sorted(stats_true, type='voltage C2', recomputed=False, sortby='time')

        vC1_false = get_sorted(stats_false, type='voltage C1', sortby='time')
        vC2_false = get_sorted(stats_false, type='voltage C2', sortby='time')

        diff_true1, diff_true2 = [v[1] - V_ref[0] for v in vC1_true], [v[1] - V_ref[1] for v in vC2_true]
        diff_false1, diff_false2 = [v[1] - V_ref[0] for v in vC1_false], [v[1] - V_ref[1] for v in vC2_false]

        times_true1, times_true2 = [v[0] for v in vC1_true], [v[0] for v in vC2_true]
        times_false1, times_false2 = [v[0] for v in vC1_false], [v[0] for v in vC2_false]

        # find time points before, after and at switch 1 to find correct differences
        for m in range(len(times_true1)):
            if np.round(times_true1[m], 15) == np.round(t_switch1, 15):
                diffs_true_switch1.append(diff_true1[m])

        for m in range(1, len(times_false1)):
            if times_false1[m - 1] < t_switch1 < times_false1[m]:
                diffs_false_before_switch1.append(diff_false1[m - 1])
                diffs_false_after_switch1.append(diff_false1[m])

        # find time points before, after and at switch 2 to find correct differences
        for m in range(len(times_true2)):
            if np.round(times_true2[m], 15) == np.round(t_switch2, 15):
                diffs_true_switch2.append(diff_true2[m])

        for m in range(1, len(times_false2)):
            if times_false2[m - 1] < t_switch2 < times_false2[m]:
                diffs_false_before_switch2.append(diff_false2[m - 1])
                diffs_false_after_switch2.append(diff_false2[m])

        # separate all restarts into restarts for switch1, and restarts for switch2
        restarts_dt = restarts_dict[dt_item]
        for i in range(len(restarts_dt[:, 0])):
            if restarts_dt[i, 0] == t_switch1:
                restarts_switch1.append(np.sum(restarts_dt[0 : i - 1, 1]))

            if restarts_dt[i, 0] == t_switch2:
                restarts_switch2.append(np.sum(restarts_dt[i - 2 :, 1]))

    setup_mpl()
    fig_around, ax_around = plt_helper.plt.subplots(1, 2, figsize=(6, 3), sharex='col', sharey='row')
    ax_around[0].set_title("Difference $v_{C_{1}}-V_{ref1}$")
    pos1 = ax_around[0].plot(dt_list, diffs_false_before_switch1, 'rs-', label='SE=False - before switch1')
    pos2 = ax_around[0].plot(dt_list, diffs_false_after_switch1, 'bd-', label='SE=False - after switch1')
    pos3 = ax_around[0].plot(dt_list, diffs_true_switch1, 'kd-', label='SE=True')
    ax_around[0].set_xticks(dt_list)
    ax_around[0].set_xticklabels(dt_list)
    ax_around[0].set_xscale('log', base=10)
    ax_around[0].set_yscale('symlog', linthresh=1e-10)
    ax_around[0].set_ylim(-2, 2)
    ax_around[0].set_xlabel(r'$\Delta t$')

    restart_ax1 = ax_around[0].twinx()
    restarts1 = restart_ax1.plot(dt_list, restarts_switch1, 'cs--', label='Restarts')

    lines = pos1 + pos2 + pos3 + restarts1
    labels = [l.get_label() for l in lines]
    ax_around[0].legend(lines, labels, frameon=False, fontsize=8, loc='center right')

    ax_around[1].set_title("Difference $v_{C_{2}}-V_{ref2}$")
    pos1 = ax_around[1].plot(dt_list, diffs_false_before_switch2, 'rs-', label='SE=False - before switch2')
    pos2 = ax_around[1].plot(dt_list, diffs_false_after_switch2, 'bd-', label='SE=False - after switch2')
    pos3 = ax_around[1].plot(dt_list, diffs_true_switch2, 'kd-', label='SE=True')
    ax_around[1].set_xticks(dt_list)
    ax_around[1].set_xticklabels(dt_list)
    ax_around[1].set_xscale('log', base=10)
    ax_around[1].set_yscale('symlog', linthresh=1e-10)
    ax_around[1].set_ylim(-2, 2)
    ax_around[1].set_xlabel(r'$\Delta t$')

    restart_ax2 = ax_around[1].twinx()
    restarts2 = restart_ax2.plot(dt_list, restarts_switch2, 'cs--', label='Restarts')
    restart_ax2.set_ylabel('Restarts')

    lines = pos1 + pos2 + pos3 + restarts2
    labels = [l.get_label() for l in lines]
    ax_around[1].legend(lines, labels, frameon=False, fontsize=8, loc='center right')

    fig_around.savefig('data/{}/diffs_estimation_{}.png'.format(problem, sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_around)


def differences_over_time(dt_list, problem, sweeper, V_ref, cwd='./'):
    """
    Routine to plot the differences in time using the switch estimator or not. Produces the
    difference_estimation_vC_<problem_classes>_<sweeper_class>.png file
    """

    if len(dt_list) > 1:
        setup_mpl()
        fig_diffs1, ax_diffs1 = plt_helper.plt.subplots(
            1, len(dt_list), figsize=(2 * len(dt_list), 2), sharex='col', sharey='row'
        )
        fig_diffs2, ax_diffs2 = plt_helper.plt.subplots(
            1, len(dt_list), figsize=(2 * len(dt_list), 2), sharex='col', sharey='row'
        )

    else:
        setup_mpl()
        fig_diffs1, ax_diffs1 = plt_helper.plt.subplots(
            1, 1, figsize=(3, 3), sharex='col', sharey='row'
        )
        fig_diffs2, ax_diffs2 = plt_helper.plt.subplots(
            1, 1, figsize=(3, 3), sharex='col', sharey='row'
        )

    count_ax = 0
    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_2condensators_dt{}_USETrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_2condensators_dt{}_USEFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        val_switch1 = get_sorted(stats_true, type='switch1', sortby='time')
        val_switch2 = get_sorted(stats_true, type='switch2', sortby='time')

        t_switch1, t_switch2 = [v[0] for v in val_switch1], [v[0] for v in val_switch2]

        t_switch1, t_switch2 = t_switch1[0], t_switch2[0]

        vC1_true = get_sorted(stats_true, type='voltage C1', recomputed=False, sortby='time')
        vC2_true = get_sorted(stats_true, type='voltage C2', recomputed=False, sortby='time')

        vC1_false = get_sorted(stats_false, type='voltage C1', sortby='time')
        vC2_false = get_sorted(stats_false, type='voltage C2', sortby='time')

        diff_true1, diff_true2 = [v[1] - V_ref[0] for v in vC1_true], [v[1] - V_ref[1] for v in vC2_true]
        diff_false1, diff_false2 = [v[1] - V_ref[0] for v in vC1_false], [v[1] - V_ref[1] for v in vC2_false]

        times_true1, times_true2 = [v[0] for v in vC1_true], [v[0] for v in vC2_true]
        times_false1, times_false2 = [v[0] for v in vC1_false], [v[0] for v in vC2_false]

        if len(dt_list) > 1:
            ax_diffs1[count_ax].plot(times_true1, diff_true1, label='SE=True', color='#ff7f0e')
            ax_diffs1[count_ax].plot(times_false1, diff_false1, label='SE=False', color='#1f77b4')
            ax_diffs1[count_ax].axvline(x=t_switch1, linestyle='--', color='k', label='Switch1')
            ax_diffs1[count_ax].legend(frameon=False, fontsize=6, loc='lower left')
            ax_diffs1[count_ax].set_yscale('symlog', linthresh=1e-5)
            ax_diffs1[count_ax].set_xlabel('Time', fontsize=6)

            ax_diffs2[count_ax].plot(times_true2, diff_true2, label='SE=True', color='#ff7f0e')
            ax_diffs2[count_ax].plot(times_false2, diff_false2, label='SE=False', color='#1f77b4')
            ax_diffs2[count_ax].axvline(x=t_switch2, linestyle='--', color='k', label='Switch1')
            ax_diffs2[count_ax].legend(frameon=False, fontsize=6, loc='lower left')
            ax_diffs2[count_ax].set_yscale('symlog', linthresh=1e-7)
            ax_diffs2[count_ax].set_xlabel('Time', fontsize=6)

            if count_ax == 0:
                ax_diffs1[count_ax].set_ylabel('Difference $v_{C_1}-V_{ref1}$', fontsize=6)
                ax_diffs2[count_ax].set_ylabel('Difference $v_{C_2}-V_{ref2}$', fontsize=6)

            count_ax += 1

        else:
            ax_diffs1.set_title('dt={}'.format(dt_item))
            ax_diffs1.plot(times_true1, diff_true1, label='SE=True', color='#ff7f0e')
            ax_diffs1.plot(times_false1, diff_false1, label='SE=False', color='#1f77b4')
            ax_diffs1.axvline(x=t_switch1, linestyle='--', color='k', label='Switch1')
            ax_diffs1.legend(frameon=False, fontsize=6, loc='lower left')
            ax_diffs1.set_yscale('symlog', linthresh=1e-5)
            ax_diffs1.set_xlabel('Time', fontsize=6)

            ax_diffs2.set_title('dt={}'.format(dt_item))
            ax_diffs2.plot(times_true2, diff_true2, label='SE=True', color='#ff7f0e')
            ax_diffs2.plot(times_false2, diff_false2, label='SE=False', color='#1f77b4')
            ax_diffs2.axvline(x=t_switch2, linestyle='--', color='k', label='Switch1')
            ax_diffs2.legend(frameon=False, fontsize=6, loc='lower left')
            ax_diffs2.set_yscale('symlog', linthresh=1e-7)
            ax_diffs2.set_xlabel('Time', fontsize=6)

            ax_diffs1.set_ylabel('Difference $v_{C_1}-V_{ref1}$', fontsize=6)
            ax_diffs2.set_ylabel('Difference $v_{C_2}-V_{ref2}$', fontsize=6)

    fig_diffs1.savefig(
        'data/{}/difference_estimation_vC1_{}.png'.format(problem, sweeper), dpi=300, bbox_inches='tight'
    )
    plt_helper.plt.close(fig_diffs1)

    fig_diffs2.savefig(
        'data/{}/difference_estimation_vC2_{}.png'.format(problem, sweeper), dpi=300, bbox_inches='tight'
    )
    plt_helper.plt.close(fig_diffs2)


def iterations_over_time(dt_list, maxiter, problem, sweeper, cwd='./'):
    """
    Routine  to plot the number of iterations over time using switch estimator or not. Produces the iters_<sweeper_class>.png file
    """

    iters_time_true = []
    iters_time_false = []
    times_true = []
    times_false = []
    t_switches1 = []
    t_switches2 = []

    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_2condensators_dt{}_USETrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_2condensators_dt{}_USEFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        iter_counts_true_val = get_sorted(stats_true, type='niter', recomputed=False, sortby='time')
        iter_counts_false_val = get_sorted(stats_false, type='niter', sortby='time')

        iters_time_true.append([v[1] for v in iter_counts_true_val])
        iters_time_false.append([v[1] for v in iter_counts_false_val])

        times_true.append([v[0] for v in iter_counts_true_val])
        times_false.append([v[0] for v in iter_counts_false_val])

        val_switch1 = get_sorted(stats_true, type='switch1', sortby='time')
        t_switch1 = [v[0] for v in val_switch1]
        t_switches1.append(t_switch1[0])

        val_switch2 = get_sorted(stats_true, type='switch2', sortby='time')
        t_switch2 = [v[0] for v in val_switch2]
        t_switches2.append(t_switch2[0])

    if len(dt_list) > 1:
        setup_mpl()
        fig_iter_all, ax_iter_all = plt_helper.plt.subplots(
            nrows=2, ncols=len(dt_list), figsize=(2 * len(dt_list) - 1, 3), sharex='col', sharey='row'
        )
        for row in range(2):
            for col in range(len(dt_list)):
                if row == 0:
                    # SE = False
                    ax_iter_all[row, col].plot(times_false[col], iters_time_false[col], label='SE=False')
                    ax_iter_all[row, col].set_title('dt={}'.format(dt_list[col]))
                    ax_iter_all[row, col].set_ylim(1, maxiter)

                else:
                    # SE = True
                    ax_iter_all[row, col].plot(times_true[col], iters_time_true[col], label='SE=True')
                    ax_iter_all[row, col].axvline(x=t_switches1[col], linestyle='--', color='r', label='Switch1')
                    ax_iter_all[row, col].axvline(x=t_switches2[col], linestyle='--', color='r', label='Switch2')
                    ax_iter_all[row, col].set_xlabel('Time', fontsize=6)
                    ax_iter_all[row, col].set_ylim(1, maxiter)

                if col == 0:
                    ax_iter_all[row, col].set_ylabel('Number iterations', fontsize=6)

                ax_iter_all[row, col].legend(frameon=False, fontsize=6, loc='upper right')
    else:
        setup_mpl()
        fig_iter_all, ax_iter_all = plt_helper.plt.subplots(
            nrows=2, ncols=1, figsize=(3, 3), sharex='col', sharey='row'
        )

        for row in range(2):
            if row == 0:
                # SE = False
                ax_iter_all[row].plot(times_false[0], iters_time_false[0], label='SE=False')
                ax_iter_all[row].set_title('dt={}'.format(dt_list[0]))
                ax_iter_all[row].set_ylim(1, maxiter)

            else:
                # SE = True
                ax_iter_all[row].plot(times_true[0], iters_time_true[0], label='SE=True')
                ax_iter_all[row].axvline(x=t_switches1[0], linestyle='--', color='r', label='Switch1')
                ax_iter_all[row, col].axvline(x=t_switches2[col], linestyle='--', color='r', label='Switch2')
                ax_iter_all[row].set_xlabel('Time', fontsize=6)
                ax_iter_all[row].set_ylim(1, maxiter)

            ax_iter_all[row].set_ylabel('Number iterations', fontsize=6)
            ax_iter_all[row].legend(frameon=False, fontsize=6, loc='upper right')

    plt_helper.plt.tight_layout()
    fig_iter_all.savefig('data/{}/iters_{}.png'.format(problem, sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_iter_all)


if __name__ == "__main__":
    check()
