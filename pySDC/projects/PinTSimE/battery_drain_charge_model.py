import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import sort_stats, filter_stats, get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery, battery_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.battery_model import get_recomputed
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
            type='voltage C_pv',
            value=L.uend[0],
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='voltage C0',
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


def main(dt, problem, sweeper, use_switch_estimator):
    """
    A simple test program to do SDC/PFASST runs for the battery drain charge model

    Args:
        dt (float): time step for computation
        problem (problem_class.__name__): problem class that wants to be simulated
        sweeper (sweeper_class.__name__): sweeper class for solving the problem class numerically
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not

    Returns:
        stats (dict): Raw statistics from a controller run
        description (dict): contains all information for a controller run
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['I_pv'] =
    problem_params['R_pv'] =
    problem_params['C_pv'] =
    problem_params['R0'] =
    problem_params['C0'] =
    problem_params['Rline2'] =
    problem_params['Rload'] =
    problem_params['Vs'] =
    problem_params['V_ref'] =
    problem_params['alpha'] =

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data
    controller_params['mssdc_jac'] = False

    # convergence controllers
    convergence_controllers = dict()
    if use_switch_estimator:
        switch_estimator_params = {}
        convergence_controllers.update({SwitchEstimator: switch_estimator_params})

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

    # set time parameters
    t0 = 0.0
    Tend = 0.3

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    Path("data").mkdir(parents=True, exist_ok=True)
    fname = 'data/battery_drain_charge_{}_USE{}.dat'.format(sweeper.__name__, use_switch_estimator)
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    return stats, description


def run():
    """
    Executes the simulation for the battery drain chargemodel using the IMEX sweeper and plot the results
    as <problem_class>_model_solution_<sweeper_class>.png
    """

    dt = 1e-2
    problem_classes = [battery_drain_charge]
    sweeper_classes = [imex_1st_order]
    recomputed = False
    use_switch_estimator = [False]

    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for use_SE in use_switch_estimator:
            stats, description = main(
                dt=dt,
                problem=problem,
                sweeper=sweeper,
                use_switch_estimator=use_SE,
            )

        plot_voltages(description, problem.__name__, sweeper.__name__, recomputed, use_SE)


def plot_voltages(description, problem, sweeper, recomputed, use_switch_estimator, cwd='./'):
    """
    Routine to plot the numerical solution of the model

    Args:
        description(dict): contains all information for a controller run
        problem (problem_class.__name__): problem class that wants to be simulated
        sweeper (sweeper_class.__name__): sweeper class for solving the problem class numerically
        recomputed (bool): flag if the values after a restart are used or before
        use_switch_estimator (bool): flag if the switch estimator wants to be used or not
        cwd: current working directory
    """

    f = open(cwd + 'data/battery_drain_charge_{}_USE{}.dat'.format(sweeper, use_switch_estimator), 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    vC_pv = get_sorted(stats, type='voltage C_pv', recomputed=False, sortby='time')
    vC0 = get_sorted(stats, type='voltage C0', recomputed=False, sortby='time')

    times = [v[0] for v in vC0]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title('Simulation of {} using {}'.format(problem, sweeper), fontsize=10)
    ax.plot(times, [v[1] for v in vC_pv], label=r'$v_{C_{pv}}$')
    ax.plot(times, [v[1] for v in vC0], label=r'$v_{C_0}$')

    if use_switch_estimator:
        switches = get_recomputed(stats, type='switch', sortby='time')

        assert len(switches) >= 1, 'No switches found!'
        t_switch = [v[1] for v in switches]
        ax.axvline(x=t_switch[-1], linestyle='--', linewidth=0.8, color='r', label='Switch')

    ax.axhline(y=1.0, linestyle='--', linewidth=0.8, color='g', label='$V_{ref}$')

    ax.legend(frameon=False, fontsize=8, loc='upper right')

    ax.set_xlabel('Time', fontsize=8)
    ax.set_ylabel('Energy', fontsize=8)

    fig.savefig('data/{}_model_solution_{}.png'.format(problem, sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


if __name__ == "__main__":
    run()
