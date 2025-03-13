from mpi4py import MPI
import numpy as np

from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep

from pySDC.projects.DAE.run.error import get_error_label, plot_result
from pySDC.projects.DAE.run.wallclocktime_error import run_serial_test, run_parallel_test
from pySDC.projects.DAE.run.utils import SDC_METHODS, RK_METHODS, COLLOCATION_METHODS
from pySDC.projects.DAE import getLabel, Plotter


def print_status(dt, eps, nNodes, problemType, QI, SDC_METHODS):
    msg_sdc = f"\n{QI} with {nNodes} nodes: Running test for {problemType} with {eps=} using {dt=}...\n"
    msg_rk_coll = f"\n{QI}: Running test for {problemType} with {eps=} using {dt=}...\n"

    msg = msg_sdc if QI in SDC_METHODS else msg_rk_coll
    print(msg)


def plot_err(time, err, err_plotter, QI, color, marker, markersize, linestyle, label):
    r"""
    Plots the results for numerical performance of the methods. Here, a decision is made where the results
    are plotted. Results of general SDC schemes were plotted in subplot 0, results for RK schemes and 
    collocation methods were plotted in subplot 1. Results for parallel SDC schemes were plotted in both
    subplots.

    Parameters
    ----------
    time : list
        Contains wallclock times.
    err : list
        List of errors achieved in simulation.
    err_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    QI : str
        Indicates the scheme.
    color : str
        Problem-specific marker.
    marker : str
        Problem-specific marker.
    markersize : float
        Size of marker.
    linestyle : str
        Problem-specific linestyle.
    label : str
        Problem-specific label.

    Returns
    -------
    err_plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class with updated plot.
    """

    if QI == "MIN-SR-S":
        err_plotter = plot_result(err_plotter, time, err, 0, color, marker, markersize, linestyle, label)
    elif QI == "MIN-SR-FLEX":
        err_plotter = plot_result(err_plotter, time, err, 1, color, marker, markersize, linestyle, label)
    else:
        err_plotter = plot_result(err_plotter, time, err, 0, color, marker, markersize, linestyle, label)
        err_plotter = plot_result(err_plotter, time, err, 1, color, marker, markersize, linestyle, label)

    return err_plotter


def get_schemes():
    schemes = [("MIN-SR-S", "constrainedDAE", 0.0), ("MIN-SR-FLEX", "constrainedDAE", 0.0), ("MIN-SR-S", "embeddedDAE", 0.0), ("MIN-SR-FLEX", "embeddedDAE", 0.0),
        ("MIN-SR-S", "fullyImplicitDAE", 0.0), ("MIN-SR-FLEX", "fullyImplicitDAE", 0.0),
        ("MIN-SR-S", "semiImplicitDAE", 0.0), ("MIN-SR-FLEX", "semiImplicitDAE", 0.0),
        ("BE", "fullyImplicitDAE", 0.0), ("RadauIIA5", "fullyImplicitDAE", 0.0), ("RadauIIA7", "fullyImplicitDAE", 0.0)]

    return schemes


def finalize_plot(err_plotter, num_nodes, problem_name):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    dt_list : list
        List containing different time step size.
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    num_nodes : int
        Number of collocation nodes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    subplot_indices : tuple
        Subplot indices as tuple for y and z.
    """

    err_plotter.set_title(rf"$Q_\Delta=$MIN-SR-S", subplot_index=0, fontsize=20)
    err_plotter.set_title(rf"$Q_\Delta=$MIN-SR-FLEX", subplot_index=1, fontsize=20)

    err_label = get_error_label(problem_name)

    err_plotter.set_xlabel("wall clock time")
    err_plotter.set_ylabel(err_label, subplot_index=0)
    err_plotter.set_ylabel(err_label, subplot_index=1)

    err_plotter.set_xscale(scale="log", subplot_index=0)
    err_plotter.set_xscale(scale="log", subplot_index=1)

    err_plotter.set_xlim((7e-4, 3e0), scale="log")
    err_plotter.set_ylim((1e-16, 1e0), scale="log")

    err_plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=20)

    filename = "data" + "/" + f"{problem_name}" + "/" + f"numerical_methods_performance_MPI_{num_nodes=}.png"
    err_plotter.save(filename)


def get_random_plot_stuff(scheme):
    colors = {
        ("MIN-SR-S", "constrainedDAE", 0.0): "mediumseagreen",
        ("MIN-SR-FLEX", "constrainedDAE", 0.0): "mediumseagreen",
        ("MIN-SR-S", "embeddedDAE", 0.0): "royalblue",
        ("MIN-SR-FLEX", "embeddedDAE", 0.0): "royalblue",
        ("MIN-SR-S", "fullyImplicitDAE", 0.0): "mediumorchid",
        ("MIN-SR-FLEX", "fullyImplicitDAE", 0.0): "mediumorchid",
        ("MIN-SR-S", "semiImplicitDAE", 0.0): "sandybrown",
        ("MIN-SR-FLEX", "semiImplicitDAE", 0.0): "sandybrown",
        ("BE", "fullyImplicitDAE", 0.0): "grey",
        ("RadauIIA5", "fullyImplicitDAE", 0.0): "indianred",
        ("RadauIIA7", "fullyImplicitDAE", 0.0): "black",
    }

    markers = {
        ("MIN-SR-S", "constrainedDAE", 0.0): "v",
        ("MIN-SR-FLEX", "constrainedDAE", 0.0): "v",
        ("MIN-SR-S", "embeddedDAE", 0.0): "^",
        ("MIN-SR-FLEX", "embeddedDAE", 0.0): "^",
        ("MIN-SR-S", "fullyImplicitDAE", 0.0): "<",
        ("MIN-SR-FLEX", "fullyImplicitDAE", 0.0): "<",
        ("MIN-SR-S", "semiImplicitDAE", 0.0): ">",
        ("MIN-SR-FLEX", "semiImplicitDAE", 0.0): ">",
        ("BE", "fullyImplicitDAE", 0.0): "s",
        ("RadauIIA5", "fullyImplicitDAE", 0.0): "D",
        ("RadauIIA7", "fullyImplicitDAE", 0.0): "P",
    }

    linestyles = {
        ("MIN-SR-S", "constrainedDAE", 0.0): "solid",
        ("MIN-SR-FLEX", "constrainedDAE", 0.0): "solid",
        ("MIN-SR-S", "embeddedDAE", 0.0): "solid",
        ("MIN-SR-FLEX", "embeddedDAE", 0.0): "solid",
        ("MIN-SR-S", "fullyImplicitDAE", 0.0): "dotted",
        ("MIN-SR-FLEX", "fullyImplicitDAE", 0.0): "dotted",
        ("MIN-SR-S", "semiImplicitDAE", 0.0): "dotted",
        ("MIN-SR-FLEX", "semiImplicitDAE", 0.0): "dotted",
        ("BE", "fullyImplicitDAE", 0.0): "dashed",
        ("RadauIIA5", "fullyImplicitDAE", 0.0): "dashed",
        ("RadauIIA7", "fullyImplicitDAE", 0.0): "dashed",
    }

    markersize = 13

    return colors[scheme], markers[scheme], linestyles[scheme], markersize


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    problem_name = "LINEAR-TEST"

    schemes = get_schemes()
    QI_list = [item[0] for item in schemes]

    num_nodes = comm.Get_size()

    kwargs = {"e_tol": 1e-13}

    t0 = 0.0
    dt_list = np.logspace(-2.0, 0.0, num=11)

    hook_class = [LogGlobalErrorPostStep]

    results = []

    for scheme in schemes:
        QI, problem_type, eps = scheme[0], scheme[1], scheme[2]

        newton_tol = 1e-12 if QI in SDC_METHODS or QI in COLLOCATION_METHODS else 1e-15
        kwargs["newton_tol"] = newton_tol

        if QI in ["MIN-SR-S", "MIN-SR-FLEX"]:
            err, time = run_parallel_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs)
        else:
            err, time = run_serial_test(t0, QI, dt_list, num_nodes, problem_name, problem_type, hook_class, eps, comm, rank, **kwargs)

        if rank == 0:
            # Define plotting-related stuff
            problem_label = getLabel(problem_type, eps, QI)

            color, marker, linestyle, markersize = get_random_plot_stuff(scheme)

            results.append((time, err, problem_type, QI, eps, color, marker, linestyle, markersize, problem_label))

        comm.Barrier()

    if rank == 0:
        err_plotter = Plotter(nrows=1, ncols=2, figsize=(12, 6))

        for time, err, problem_type, QI, eps, color, marker, linestyle, markersize, label in results:
            err_plotter = plot_err(time, err, err_plotter, QI, color, marker, markersize, linestyle, label)

        finalize_plot(err_plotter, num_nodes, problem_name)

    MPI.Finalize()