import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery, battery_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.piline_model import setup_mpl
import pySDC.helpers.plot_helper as plt_helper
from pySDC.core.Hooks import hooks

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


class log_data(hooks):
    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='current L',
            value=L.uend[0],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C',
            value=L.uend[1],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restart',
            value=int(step.status.get('restart')),
        )


def main(dt, problem, restol, sweeper, use_switch_estimator):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = restol
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_maxiter'] = 200
    problem_params['newton_tol'] = 1e-08
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = 1.0
    problem_params['R'] = 1.0
    problem_params['L'] = 1.0
    problem_params['alpha'] = 1.2
    problem_params['V_ref'] = 1.0
    problem_params['set_switch'] = np.array([False], dtype=bool)
    problem_params['t_switch'] = np.zeros(1)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # convergence controllers
    switch_estimator_params = {}
    convergence_controllers = {SwitchEstimator: switch_estimator_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = problem  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = sweeper  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if use_switch_estimator:
        description['convergence_controllers'] = convergence_controllers

    proof_assertions_description(description, problem_params)

    # set time parameters
    t0 = 0.0
    Tend = 0.5

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', recomputed=False, sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0

    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/{}".format(problem.__name__)).mkdir(parents=True, exist_ok=True)
    fname = 'data/battery_{}_USE{}.dat'.format(sweeper.__name__, use_switch_estimator)
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    f = open('data/battery_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    f.write(out + '\n')
    print(out)
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %1i' % item
        f.write(out + '\n')
        print(out)
        min_iter = min(min_iter, item[1])
        max_iter = max(max_iter, item[1])
    times = [item[0] for item in iter_counts]
    for m in range(len(times)):
        if niters[m] == step_params['maxiter']:
            print(times[m])
    assert np.mean(niters) <= 9, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    return description


def run():
    """
    Executes the simulation for the battery model using two different sweepers and plot the results
    as <problem_class>_model_solution_<sweeper_class>.png
    """

    dt = 1e-4
    problem_classes = [battery_implicit]  # [battery, battery_implicit]
    restolerances = [5e-8]  # [1e-12, 1e-8]
    sweeper_classes = [generic_implicit]  # [imex_1st_order, generic_implicit]
    use_switch_estimator = [False]  # [True, False]

    for problem, restol, sweeper in zip(problem_classes, restolerances, sweeper_classes):
        for use_SE in use_switch_estimator:
            description = main(dt=dt, problem=problem, restol=restol, sweeper=sweeper, use_switch_estimator=use_SE)

            plot_voltages(description, problem.__name__, sweeper.__name__, use_SE)

        plot_comparison(description, problem.__name__, sweeper.__name__)


def plot_voltages(description, problem, sweeper, use_switch_estimator, cwd='./'):
    """
    Routine to plot the numerical solution of the model alone
    """

    f = open('data/battery_{}_USE{}.dat'.format(sweeper, use_switch_estimator), 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    cL_val = get_sorted(stats, type='current L', recomputed=True, sortby='time')
    vC_val = get_sorted(stats, type='voltage C', recomputed=True, sortby='time')

    times = [v[0] for v in vC_val]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title('Simulation of {} using {}'.format(problem, sweeper), fontsize=10)
    ax.plot(times, [v[1] for v in cL_val], label=r'$i_L$')
    ax.plot(times, [v[1] for v in vC_val], label=r'$v_C$')

    if use_switch_estimator:
        val_switch = get_sorted(stats, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        ax.axvline(x=t_switch[0], linestyle='--', color='k', label='Switch')
        print("t_switch=", t_switch)

    ax.axhline(y=1.0, linestyle='--', color='k', label='$V_{ref}$')
    ax.legend(frameon=False, fontsize=12, loc='upper right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('data/{}/{}_model_solution_{}.png'.format(problem, problem, sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_comparison(description, problem, sweeper, cwd='./'):
    """
    Routine to plot the numerical solutions of the model using SE or not together with error to V_ref
    """

    f = open('data/battery_{}_USETrue.dat'.format(sweeper), 'rb')
    stats_true = dill.load(f)
    f.close()

    f = open('data/battery_{}_USEFalse.dat'.format(sweeper), 'rb')
    stats_false = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    vC_true_val = get_sorted(stats_true, type='voltage C', recomputed=True, sortby='time')
    vC_false_val = get_sorted(stats_false, type='voltage C', sortby='time')

    val_switch = get_sorted(stats_true, type='switch1', sortby='time')
    t_switch = [v[0] for v in val_switch]

    times_true = [v[0] for v in vC_true_val]
    times_false = [v[0] for v in vC_false_val]

    vC_true = [v[1] for v in vC_true_val]
    vC_false = [v[1] for v in vC_false_val]

    diff_true = []
    for i in range(len(vC_true)):
        diff_true.append((1 - vC_true[i]))

    diff_false = []
    for i in range(len(vC_false)):
        diff_false.append((1 - vC_false[i]))

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(3, 1, figsize=(6, 6))

    ax[0].plot(times_true, vC_true, label='SE=True')
    ax[0].plot(times_false, vC_false, label='SE=False')
    ax[0].axhline(y=1.0, linestyle='--', color='k', label='$V_{ref}$')
    ax[0].set_ylim(0.995, 1.0025)
    ax[0].legend(frameon=False, fontsize=6, loc='upper right')
    ax[0].set_ylabel('Energy', fontsize=6)

    ax[1].plot(times_false, diff_false, label='SE=False')
    ax[1].set_yscale('symlog', linthresh=1e-5)
    ax[1].legend(frameon=False, fontsize=6, loc='upper right')
    ax[1].set_ylabel('Error', fontsize=6)

    ax[2].plot(times_true, diff_true, label='SE=True')
    ax[2].axvline(x=t_switch[0], linestyle='--', color='r', label='Switch')
    ax[2].set_yscale('symlog', linthresh=1e-5)
    ax[2].legend(frameon=False, fontsize=6, loc='upper right')
    ax[2].set_xlabel('Time', fontsize=6)
    ax[2].set_ylabel('Error', fontsize=6)

    plt_helper.plt.tight_layout()
    fig.savefig('data/{}/{}_model_comparison_{}.png'.format(problem, problem, sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def proof_assertions_description(description, problem_params):
    """
    Function to proof the assertions (function to get cleaner code)
    """

    assert problem_params['alpha'] > problem_params['V_ref'], 'Please set "alpha" greater than "V_ref"'
    assert problem_params['V_ref'] > 0, 'Please set "V_ref" greater than 0'
    assert type(problem_params['V_ref']) == float, '"V_ref" needs to be of type float'

    assert type(problem_params['set_switch'][0]) == np.bool_, '"set_switch" has to be an bool array'
    assert type(problem_params['t_switch']) == np.ndarray, '"t_switch" has to be an array'
    assert problem_params['t_switch'][0] == 0, '"t_switch" is only allowed to have entry zero'

    assert 'errtol' not in description['step_params'].keys(), 'No exact solution known to compute error'
    assert 'alpha' in description['problem_params'].keys(), 'Please supply "alpha" in the problem parameters'
    assert 'V_ref' in description['problem_params'].keys(), 'Please supply "V_ref" in the problem parameters'


if __name__ == "__main__":
    run()
