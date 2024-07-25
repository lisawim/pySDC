import numpy as np
from mpi4py import MPI

from pySDC.core.errors import ProblemError
from pySDC.helpers.stats_helper import get_sorted

from pySDC.projects.DAE import QI_SERIAL, QI_PARALLEL, getEndTime, computeSolution, getColor, getMarker, Plotter


def runTest(num_processes, global_comm, global_rank, problemName, t0, dt, Tend, QI, problemType, useMPI, eps):
    r"""
    In this function the speed-up test is done. Here, the communicator is then splitted. Number of collocation nodes
    is adapted as well.

    Patameters
    ----------
    num_processes : int
        Number of processes.
    global_comm : MPI.COMM_WORLD
        Global communicator to be split.
    global_rank : MPI.COMM_WORLD
        Current rank that passes this function

    Returns
    -------
    """

    if global_rank < num_processes:
        # Split the communicator to create a new communicator for this test
        sub_comm = global_comm.Split(color=1, key=global_rank)

        sub_nNodes = sub_comm.Get_size()

        # Perform the computation with the sub-communicator
        solutionStats = computeSolution(
            problemName=problemName,
            t0=t0,
            dt=dt,
            Tend=Tend,
            nNodes=sub_nNodes,
            QI=QI,
            problemType=problemType,
            useMPI=useMPI,
            eps=eps,
            comm=sub_comm,
        )
        
        sub_comm.Free()
        return solutionStats
    else:
        # Split the communicator to exclude this process
        sub_comm = global_comm.Split(color=MPI.UNDEFINED, key=global_rank)
        return None


def main():
    global_comm = MPI.COMM_WORLD
    global_rank = global_comm.Get_rank()
    global_size = global_comm.Get_size()

    if global_size < 2:
        if global_rank == 0:
            raise ProblemError("This test requires at least 2 processes!")

    nRuns = 5

    problemName = 'VAN-DER-POL'

    QIser = 'LU'
    QIpar = 'MIN-SR-S'

    # assert QIser in QI_SERIAL, f"Q_\Delta for serial test not suitable! Choose on contained in QI_SERIAL!"
    # assert QIpar in QI_PARALLEL, f"Q_\Delta cannot be executed in parallel! Choose on contained in QI_PARALLEL!"

    problemType = 'SPP'

    compute_one_step = False
    t0 = 0.0
    dt = 1e-5
    Tend = getEndTime(problemName) if not compute_one_step else t0 + dt

    epsList = [10 ** (-m) for m in range(1, 6)]

    resultsDict = {} if global_rank == 0 else None

    # Run serial tests
    for num_tasks in range(2, global_size + 1):
        # Number of processes/tasks is number of collocation nodes
        nNodes = num_tasks
        if global_rank == 0:
            resultsDict[num_tasks] = {} if global_rank == 0 else None

            for eps in epsList:
                resultsDict[num_tasks][eps] = {}
                resultsDict[num_tasks][eps]['serial'] = 0
                resultsDict[num_tasks][eps]['parallel'] = 0

        for _ in range(1, nRuns + 1):
            for eps in epsList:
                if global_rank == 0:
                    solutionStats = computeSolution(
                        problemName=problemName,
                        t0=t0,
                        dt=dt,
                        Tend=Tend if not compute_one_step else t0 + dt,
                        nNodes=nNodes,
                        QI=QIser,
                        problemType=problemType,
                        useMPI=False,
                        eps=eps,
                        comm=global_comm,
                    )

                    timingRun = np.array(get_sorted(solutionStats, type='timing_run', sortby='time'))
                    resultsDict[num_tasks][eps]['serial'] += timingRun[0][1]

        else:
            # Other processes wait for the serial test to complete
            global_comm.Barrier()

    # # Run tests for 2 until 10 processes
    for num_processes in range(2, global_size + 1):

        for _ in range(1, nRuns + 1):
            for eps in epsList:
                if global_rank == 0:
                    print(f"\nRunning test with {num_processes} processes...\n")
                global_comm.Barrier()  # Ensure all processes reach this point before continuing
                solutionStats = runTest(num_processes, global_comm, global_rank, problemName, t0, dt, Tend, QIpar, problemType, True, eps)
                global_comm.Barrier()  # Ensure all processes finish this test before continuing

                if global_rank == 0:
                    timingRun = np.array(get_sorted(solutionStats, type='timing_run', sortby='time'))
                    resultsDict[num_processes][eps]['parallel'] += timingRun[0][1]

    if global_rank == 0:
        for num_processes in range(2, global_size + 1):
            for eps in epsList:
                resultsDict[num_processes][eps]['serial'] /= nRuns
                resultsDict[num_processes][eps]['parallel'] /= nRuns

        plotSpeedup(dt, epsList, resultsDict, QIser, QIpar, compute_one_step, problemType, problemName)


def plotSpeedup(dt, epsList, resultsDict, QIser, QIpar, compute_one_step, problemType, problemName):
    MPIPlotter = Plotter(nrows=1, ncols=2, orientation='horizontal', figsize=(22, 10))

    # Prepare data for plotting
    speedups = {eps: [] for eps in epsList}
    efficiencies = {eps: [] for eps in epsList}
    num_processes_list = sorted(resultsDict.keys())

    for num_processes in num_processes_list:
        for eps in epsList:
            timings = resultsDict[num_processes][eps]
            speedup = timings['serial'] / timings['parallel']
            speedups[eps].append(speedup)

            efficiency = speedup / num_processes
            efficiencies[eps].append(efficiency)

    # Plot the data
    for i, eps in enumerate(epsList):
        color, marker = getColor(problemType, i), getMarker(problemType)
        MPIPlotter.plot(num_processes_list, speedups[eps], subplot_index=0, color=color, marker=marker, label=rf'$\varepsilon=${eps}')
        MPIPlotter.plot(num_processes_list, efficiencies[eps], subplot_index=1, color=color, marker=marker)

    MPIPlotter.set_xticks(num_processes_list, subplot_index=None)
    MPIPlotter.set_yticks(num_processes_list, subplot_index=0)
    MPIPlotter.set_xlabel('Number of Processes', subplot_index=None)

    MPIPlotter.set_legend(subplot_index=0, loc='best')
    MPIPlotter.set_grid(True, subplot_index=None)

    MPIPlotter.set_title(rf'Speedup for $\Delta t=${dt}', subplot_index=0)
    MPIPlotter.set_title(rf'Efficiency for $\Delta t=${dt}', subplot_index=1)

    MPIPlotter.set_ylabel('Speedup (Serial Time / Parallel Time)', subplot_index=0)
    MPIPlotter.set_ylabel('Efficiency (Speedup / Number of processes)', subplot_index=1)

    MPIPlotter.set_ylim((0, 1), subplot_index=1)

    filename = "data" + "/" + f"{problemName}" + "/" + f"MPITest_{QIser=}_{QIpar=}_{dt=}_{compute_one_step=}.png"
    MPIPlotter.save(filename)

    # for ax_wrapper in [ax[0], ax[1]]:
    #     ax_wrapper.tick_params(axis='both', which='major', labelsize=14)
    #     ax_wrapper.set_xticks(num_processes_list)
    #     ax_wrapper.set_xticklabels([i for i in num_processes_list])
    #     ax_wrapper.set_xlabel('Number of Processes', fontsize=20)
    #     ax_wrapper.legend(frameon=False, fontsize=12, loc='upper right')
    #     ax_wrapper.grid(True)

    # ax[0].set_title(rf'Speedup for $\Delta t=${dt}')
    # ax[1].set_title(rf'Efficiency for $\Delta t=${dt}')
    # ax[0].set_ylabel('Speedup (Serial Time / Parallel Time)', fontsize=20)
    # ax[1].set_ylabel('Efficiency (Speedup / Number of processes)', fontsize=20)

    # ax[1].set_ylim(0, 1)

    # ax[0].set_yticks(num_processes_list)
    # ax[0].set_yticklabels([i for i in num_processes_list])

    # fig.savefig(f"data/VanDerPol/plotMPITest_{QIser=}_{QIpar=}_{dt=}_{compute_one_step=}.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)


if __name__ == "__main__":
    main()
