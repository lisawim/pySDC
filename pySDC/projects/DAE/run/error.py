import numpy as np

from pySDC.implementations.hooks.log_solution import LogSolutionAfterIteration
from pySDC.helpers.stats_helper import get_sorted
from pySDC.projects.DAE import computeSolution, getColor, getEndTime, getLabel, get_linestyle, getMarker, Plotter


def get_problem_cases(k: int, problem_name: str, epsList: list=None):
    r"""
    Returns the different problem cases. In these different cases different SDC schemes will be compared:

      * Case 1: SDC for SPP, embedded SDC and constrained SDC for DAEs in integral formulation.
      * Case 2: SDC for SPP and SDC and SDC in fully-implicit and semi-implicit formulation
        (schemes proposed by M. L. Minion) in yp-formulation.
      * Case 3: SDC for SPP in integral formulation and in yp-formulation.
      * Case 4: Embedded SDC, constrained SDC (in integral formulation) and SDC in fully-implicit and
        semi-implicit formulation (i.e. yp-formulation) to solve DAE.
      * Case 5: SDC in fully-implicit and semi-implicit formulation for DAEs.
      * Case 6: SDC for SPP, embedded SDC, constrained SDC (in integral formulation) and SDC in fully-implicit and
        semi-implicit formulation (i.e. yp-formulation) to solve DAE.

    Parameters
    ----------
    k : int
        Case number.
    problem_name : str
        Name of the problem (decides the list of perturbation parameters).
    epsList : list
        List of perturbation parameter :math:`\varepsilon` can be passed. The default is
        :math:`\varepsilon=10^{-1},..,10^{-11}`.

    Returns
    -------
    problems : dict
        Problem dictionary.
    """

    if epsList is None and problem_name not in ["ANDREWS-SQUEEZER"]:
        epsListProblems = {
            "LINEAR-TEST": [10 ** (-m) for m in range(1, 12)],
            "DPR": [10 ** (-m) for m in range(1, 12)],
            "MICHAELIS-MENTEN": [10 ** (-m) for m in range(1, 8)],
            "PROTHERO-ROBINSON": [10 ** (-m) for m in range(1, 12)],
        }

        epsList = epsListProblems[problem_name]

    if k == 1:
        problems = {
            "SPP": epsList,
            "embeddedDAE": [0.0],
            "constrainedDAE": [0.0],
        }

    elif k == 2:
        problems = {
            "SPP-yp": epsList,
            "fullyImplicitDAE": [0.0],
            "semiImplicitDAE": [0.0],
        }

    elif k == 3:
        problems = {
            "SPP": epsList,
            "SPP-yp": epsList,
        }

    elif k == 4:
        problems = {
            "constrainedDAE": [0.0],
            "embeddedDAE": [0.0],
            "fullyImplicitDAE": [0.0],
            "semiImplicitDAE": [0.0],
        }

    elif k == 5:
        problems = {
            "fullyImplicitDAE": [0.0],
            "semiImplicitDAE": [0.0],
        }

    elif k == 6:
        problems = {
            "SPP": epsList,
            "constrainedDAE": [0.0],
            "embeddedDAE": [0.0],
            "fullyImplicitDAE": [0.0],
            "semiImplicitDAE": [0.0],
        }
    elif k == 7:
        problems = {
            "SPP": epsList,
            "SPP-IMEX": epsList,
        }

    elif k == 8:  # TODO: Should be done in two figures!!
        problems = {
            "SPP": epsList,
            "SPP-yp": epsList,
            "embeddedDAE": [0.0],
            "constrainedDAE": [0.0],
            "fullyImplicitDAE": [0.0],
            "semiImplicitDAE": [0.0],
        }

    elif k == 9:
        problems = {
            "constrainedDAE": [0.0],
            "semiImplicitDAE": [0.0],
        }

    elif k == 10:
        problems = {"embeddedDAE": [0.0]}

    elif k == 11:
        problems = {"constrainedDAE": [0.0]}

    elif k == 12:
        problems = {"fullyImplicitDAE": [0.0]}

    elif k == 13:
        problems = {"semiImplicitDAE": [0.0]}

    elif k == 14:
        problems = {
            "constrainedDAE": [0.0],
            "embeddedDAE": [0.0],
        }
    elif k == 15:
        problems = {
                "constrainedDAE": [0.0],
                "fullyImplicitDAE": [0.0],
                "semiImplicitDAE": [0.0],
            }

    else:
        raise NotImplementedError(f"Case {k} is not implemented!")
    
    return problems

def get_hooks(k: int, hook_for: str):
    r"""
    Returns the hooks to log errors along iterations for different cases (see documentation of
    ``get_problem_cases_and_hooks`` function).

    Parameters
    ----------
    k : int
        Case number.
    hook_for : str
        Indicates which hook we want to have.

    Returns
    -------
    hook_class: list or dict
        Hook class(es) to log the quantities during simulation.
    """

    if hook_for == "iteration":
        from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostIter, LogGlobalErrorPostIterPerturbation
        from pySDC.projects.DAE.misc.hooksDAE import LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg

        hook_SPP = [LogGlobalErrorPostIter, LogGlobalErrorPostIterPerturbation]
        hook_DAE = [LogGlobalErrorPostIterDiff, LogGlobalErrorPostIterAlg]

    elif hook_for == "sweep":
        from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostSweep, LogGlobalErrorPostSweepPerturbation
        from pySDC.projects.DAE.misc.hooksDAE import LogGlobalErrorPostSweepDiff, LogGlobalErrorPostSweepAlg

        hook_SPP = [LogGlobalErrorPostSweep, LogGlobalErrorPostSweepPerturbation]
        hook_DAE = [LogGlobalErrorPostSweepDiff, LogGlobalErrorPostSweepAlg]

    elif hook_for == "step":
        from pySDC.projects.DAE.misc.hooksEpsEmbedding import LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation
        from pySDC.projects.DAE.misc.hooksDAE import LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable

        hook_SPP = [LogGlobalErrorPostStep, LogGlobalErrorPostStepPerturbation]
        hook_DAE = [LogGlobalErrorPostStepDifferentialVariable, LogGlobalErrorPostStepAlgebraicVariable]

    else:
        raise NotImplementedError(f"There is no hook implemented for {hook_for}!")

    if k == 1:
        hook_class = {
            "SPP": hook_SPP,
            "embeddedDAE": hook_DAE,
            "constrainedDAE": hook_DAE,
        }

    elif k == 2:
        hook_class = {
            "SPP-yp": hook_SPP,
            "fullyImplicitDAE": hook_DAE,
            "semiImplicitDAE": hook_DAE,
        }

    elif k in [3, 7]:
        hook_class = hook_SPP

    elif k in [4, 5, 9, 10, 11, 12, 13, 14]:
        hook_class = hook_DAE

    elif k == 6:
        hook_class = {
            "SPP": hook_SPP,
            "embeddedDAE": hook_DAE,
            "constrainedDAE": hook_DAE,
            "fullyImplicitDAE": hook_DAE,
            "semiImplicitDAE": hook_DAE,
        }

    elif k == 8:
        hook_class = {
            "SPP": hook_SPP,
            "SPP-yp": hook_SPP,
            "embeddedDAE": hook_DAE,
            "constrainedDAE": hook_DAE,
            "fullyImplicitDAE": hook_DAE,
            "semiImplicitDAE": hook_DAE,
        }

    else:
        raise NotImplementedError(f"Case {k} is not implemented!")
    
    return hook_class

def get_figsize_for_problem_case(k: int):
    r"""
    Returns figure size and number of rows and columns for plot for chosen case (see documentation of
    ``get_problem_cases_and_hooks`` function).

    Parameters
    ----------
    k : int
        Case number.

    Returns
    -------
    dict
        Contains quantities for plot size.
    """

    dict1 = {"figsize": (18, 12), "nrows": 2, "ncols": 3}
    dict2 = {"figsize": (30, 18), "nrows": 4, "ncols": 3} 

    plot_size = {1: dict1, 2: dict1, 3: dict2, 4: dict1, 5: dict1, 6: dict1, 7: dict2, 8: dict2, 9: dict1, 10: dict1, 11: dict1, 12: dict1, 13: dict1}

    return plot_size[k]

def get_subplot_indices(k: int):
    r"""
    Returns the indices for subplots. We decide between indices for error in variable y and in variable z. Each subplot
    corresponds to one :math:`Q_\Delta` which are the keys of ``subplot_indices_y[problemType]``,
    ``subplot_indices_z[problemType]``.

    Parameter
    ---------
    k : int
        Case number.

    Returns
    -------
    subplot_indices_y : dict
        Indices for subplots corresponding to variable y (differential variable).
    subplot_indices_z : dict
        Indices for subplots corresponding to variable z (algebraic variable).
    """

    if k == 1:
        subplot_indices_y = {
            "SPP": {0: 0, 1: 1, 2: 2},
            "embeddedDAE": {0: 0, 1: 1, 2: 2},
            "constrainedDAE": {0: 0, 1: 1, 2: 2},
        }

        subplot_indices_z = {
            "SPP": {0: 3, 1: 4, 2: 5},
            "embeddedDAE": {0: 3, 1: 4, 2: 5},
            "constrainedDAE": {0: 3, 1: 4, 2: 5},
        }

    elif k == 2:
        subplot_indices_y = {
            "SPP-yp": {0: 0, 1: 1, 2: 2},
            "fullyImplicitDAE": {0: 0, 1: 1, 2: 2},
            "semiImplicitDAE": {0: 0, 1: 1, 2: 2},
        }

        subplot_indices_z = {
            "SPP-yp": {0: 3, 1: 4, 2: 5},
            "fullyImplicitDAE": {0: 3, 1: 4, 2: 5},
            "semiImplicitDAE": {0: 3, 1: 4, 2: 5},
        }

    elif k == 3:
        subplot_indices_y = {
            "SPP" : {0: 0, 1: 1, 2: 2},
            "SPP-yp": {0: 3, 1: 4, 2: 5},
        }
        subplot_indices_z = {
            "SPP" : {0: 6, 1: 7, 2: 8},
            "SPP-yp": {0: 9, 1: 10, 2: 11},
        }

    elif k == 4:
        subplot_indices_y = {
            "embeddedDAE": {0: 0, 1: 1, 2: 2},
            "constrainedDAE": {0: 0, 1: 1, 2: 2},
            "fullyImplicitDAE": {0: 0, 1: 1, 2: 2},
            "semiImplicitDAE": {0: 0, 1: 1, 2: 2},
        }

        subplot_indices_z = {
            "embeddedDAE": {0: 3, 1: 4, 2: 5},
            "constrainedDAE": {0: 3, 1: 4, 2: 5},
            "fullyImplicitDAE": {0: 3, 1: 4, 2: 5},
            "semiImplicitDAE": {0: 3, 1: 4, 2: 5},
        }
    elif k == 5:
        subplot_indices_y = {
            "fullyImplicitDAE": {0: 0, 1: 1, 2: 2},
            "semiImplicitDAE": {0: 0, 1: 1, 2: 2},
        }

        subplot_indices_z = {
            "fullyImplicitDAE": {0: 3, 1: 4, 2: 5},
            "semiImplicitDAE": {0: 3, 1: 4, 2: 5},
        }
    elif k == 6:
        subplot_indices_y = {
            "SPP": {0: 0, 1: 1, 2: 2},
            "embeddedDAE": {0: 0, 1: 1, 2: 2},
            "constrainedDAE": {0: 0, 1: 1, 2: 2},
            "fullyImplicitDAE": {0: 0, 1: 1, 2: 2},
            "semiImplicitDAE": {0: 0, 1: 1, 2: 2},
        }

        subplot_indices_z = {
            "SPP": {0: 3, 1: 4, 2: 5},
            "embeddedDAE": {0: 3, 1: 4, 2: 5},
            "constrainedDAE": {0: 3, 1: 4, 2: 5},
            "fullyImplicitDAE": {0: 3, 1: 4, 2: 5},
            "semiImplicitDAE": {0: 3, 1: 4, 2: 5},
        }
    elif k == 7:
        subplot_indices_y = {
            "SPP" : {0: 0, 1: 1, 2: 2},
            "SPP-IMEX": {0: 3, 1: 4, 2: 5},
        }
        subplot_indices_z = {
            "SPP" : {0: 6, 1: 7, 2: 8},
            "SPP-IMEX": {0: 9, 1: 10, 2: 11},
        }
    elif k == 8:
        subplot_indices_y = {
            "SPP" : {0: 0, 1: 1, 2: 2},
            "embeddedDAE": {0: 0, 1: 1, 2: 2},
            "constrainedDAE": {0: 0, 1: 1, 2: 2},
            "SPP-yp": {0: 3, 1: 4, 2: 5},
            "fullyImplicitDAE": {0: 3, 1: 4, 2: 5},
            "semiImplicitDAE": {0: 3, 1: 4, 2: 5},
        }
        subplot_indices_z = {
            "SPP" : {0: 6, 1: 6, 2: 8},
            "embeddedDAE": {0: 6, 1: 6, 2: 8},
            "constrainedDAE": {0: 6, 1: 6, 2: 8},
            "SPP-yp": {0: 9, 1: 10, 2: 11},
            "fullyImplicitDAE": {0: 9, 1: 10, 2: 11},
            "semiImplicitDAE": {0: 9, 1: 10, 2: 11},
        }

    elif k == 9:
        subplot_indices_y = {
            "constrainedDAE": {0: 0, 1: 1, 2: 2},
            "semiImplicitDAEDAE": {0: 0, 1: 1, 2: 2},
        }

        subplot_indices_z = {
            "constrainedDAE": {0: 3, 1: 4, 2: 5},
            "semiImplicitDAEDAE": {0: 3, 1: 4, 2: 5},
        }

    else:
        raise NotImplementedError(f"No subplot indices implemented for case {k}!")
    
    return subplot_indices_y, subplot_indices_z

def plot_result(plotter: Plotter, x, y, subplot_index, color, marker, markersize, linestyle, problem_label, plot_type="semilogy", markevery=None, **kwargs):
    r"""
    Plots the results.

    Parameters
    ----------
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    x : list or numpy.1darray
        Values for x-axis.
    y : list or numpy.1darray
        Values for y-axis.
    subplot_index : int
        Number of subplot index where quantity is plotted in.
    color : str
        Problem-specific color.
    marker : str
        Problem-specific marker.
    markersize : int or float
        Problem-specific markersize.
    linestyle : str
        Problem-specific linestyle.
    problem_label : str
        Label for plot.
    plot_type : str
        Type of plot.

    Returns
    -------
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Updated plotter class.
    """

    plotter.plot(
        x,
        y,
        subplot_index=subplot_index,
        color=color,
        marker=marker,
        markersize=markersize,
        linestyle=linestyle,
        label=problem_label,
        plot_type=plot_type,
        markevery=markevery,
        **kwargs,
    )
    return plotter

def get_error_label(problem_name):
    r"""
    Returns label for y-axis.

    Parameters
    ----------
    problem_name : str
        Name of the problem.

    Returns
    -------
    err_label : str
        Label for plotting.
    """

    if problem_name in ["LINEAR-TEST", "DPR", "MICHAELIS-MENTEN", "PROTHERO-ROBINSON"]:
        err_label = "global error"
    else:
        raise NotImplementedError(f"No label implemented for {problem_name}!")
    
    return err_label

def finalize_plot(k: int, dt, plotter, num_nodes, problems, problem_name, QI_list, subplot_indices, hook_for, solver_type=""):
    r"""
    Finalizes the plot with labels, titles, limits for y-axis and scaling, and legend. The plot is
    then stored with fixed filename. 

    Parameters
    ----------
    k : int
        Case number
    dt : float
        Time step size.
    plotter : pySDC.projects.DAE.run.utils.Plotter
        Plotter class.
    num_nodes : int
        Number of collocation nodes.
    problems : dict
        Contains different problem classes.
    problem_name : str
        Contains the name of the problem considered.
    QI_list : list
        Contains different :math:`Q_\Delta`.
    subplot_indices : tuple
        Subplot indices as tuple for y and z.
    """

    # Unpack subplot indices
    subplot_indices_y, subplot_indices_z = subplot_indices[0], subplot_indices[1]

    # Shortcut
    problem_types = list(problems.keys())

    # We only need one dict from subplot indices to define the title
    subplot_indices_y_1 = subplot_indices_y[problem_types[0]]
    subplot_indices_z_1 = subplot_indices_z[problem_types[0]]

    xlabel = hook_for if hook_for in ["iteration", "sweep"] else "time"
    plotter.set_xlabel(xlabel, subplot_index=None)

    err_label = get_error_label(problem_name)

    for q, QI in enumerate(QI_list):
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=subplot_indices_y_1[q], fontsize=24)
        plotter.set_title(rf"$Q_\Delta=${QI}", subplot_index=subplot_indices_z_1[q], fontsize=24)

        plotter.set_ylabel(f"{err_label} in " + r"$y$", subplot_index=subplot_indices_y_1[q])
        plotter.set_ylabel(f"{err_label} in " + r"$z$", subplot_index=subplot_indices_z_1[q])

        # plotter.set_ylim((1e-15, 1e-6), subplot_index=subplot_indices_y_1[q])
        # plotter.set_ylim((1e-15, 1e-6), subplot_index=subplot_indices_z_1[q])

        plotter.set_yscale(scale="log", subplot_index=subplot_indices_y_1[q])
        plotter.set_yscale(scale="log", subplot_index=subplot_indices_z_1[q])

    plotter.sync_ylim(min_y_set=1e-15)

    plotter.set_shared_legend(loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=6, fontsize=22)

    solve = f"_{solver_type}" if solver_type == "direct" else ""
    filename = "data" + "/" + f"{problem_name}" + "/" + f"error_{hook_for}_{num_nodes=}_{dt=}_case{k}{solve}.png"
    plotter.save(filename)


"""Main routine"""
if __name__ == "__main__":
    problem_name = "LINEAR-TEST"

    QI_list = ["MIN-SR-NS"]#["IE", "LU", "MIN-SR-S"]
    num_nodes = 29

    solver_type = "direct"  # ""
    kwargs = {
        "e_tol": -1,
        "maxiter": 10000,
        "solver_type": solver_type,
        "newton_tol": 1e-12,
        # "logger_level": 15,
    }

    t0 = 0.0
    dt = 1e-2#1e0  # 1e-1

    case = 4

    problems = {"fullyImplicitDAE": [0.0]}#get_problem_cases(k=case, problem_name=problem_name)

    hook_for = "iteration"  # "iteration"  # "step"
    if hook_for == "iteration":
        sortby = "iter"
    elif hook_for == "sweep":
        sortby = "sweep" # ?
    elif hook_for == "step":
        sortby = "time"

    hook_class = get_hooks(k=case, hook_for=hook_for)

    plot_size = get_figsize_for_problem_case(k=case)

    figsize = plot_size["figsize"]
    nrows = plot_size["nrows"]
    ncols = plot_size["ncols"]

    err_plotter = Plotter(nrows=nrows, ncols=ncols, figsize=figsize)

    subplot_indices_y, subplot_indices_z = get_subplot_indices(k=case)
    subplot_indices = (subplot_indices_y, subplot_indices_z)

    for q, QI in enumerate(QI_list):
        for problem_type, eps_values in problems.items():
            for i, eps in enumerate(eps_values):
                print(f"\n{QI} with {num_nodes} nodes: Running test for {problem_type} with {eps=} using {dt=}...\n")

                hooks = hook_class[problem_type] if isinstance(hook_class, dict) else hook_class
                hooks += [LogSolutionAfterIteration]

                # Let's do the simulation to get results
                solution_stats = computeSolution(
                    problemName=problem_name,
                    t0=t0,
                    dt=dt,
                    Tend=t0 + dt if hook_for in ["iteration", "sweep"] else getEndTime(problem_name),
                    nNodes=num_nodes,
                    QI=QI,
                    problemType=problem_type,
                    hookClass=hooks,
                    eps=eps,
                    **kwargs,
                )

                # Get error values along iterations
                x = [me[0] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_{hook_for}", sortby=sortby)]
                err_diff_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_differential_post_{hook_for}", sortby=sortby)]
                err_alg_values = [me[1] for me in get_sorted(solution_stats, type=f"e_global_algebraic_post_{hook_for}", sortby=sortby)]
                print(err_diff_values)
                print(err_alg_values)
                # if problem_type in ["embeddedDAE"]:
                #     rates = [np.linalg.norm(err_alg_values[i+1]/err_alg_values[i]) for i in range(len(err_alg_values)-1)]
                #     # print(err_alg_values)
                #     print(rates)
                # Define plotting-related stuff and use shortcuts
                color, res = getColor(problem_type, i, QI), getMarker(problem_type, i, QI)
                problem_label, linestyle = getLabel(problem_type, eps, QI), get_linestyle(problem_type, QI)
                marker = res["marker"]
                markersize = res["markersize"]

                # Shortcuts for subplot indices
                subplot_index_y = subplot_indices_y[problem_type][q]
                subplot_index_z = subplot_indices_z[problem_type][q]

                # Get things done in plot
                # x_diff, x_alg = np.arange(1, len(err_diff_values) + 1), np.arange(1, len(err_alg_values) + 1)
                err_plotter = plot_result(
                    err_plotter,
                    x,
                    err_diff_values,
                    subplot_index_y,
                    color,
                    None,  # marker,
                    None,  # markersize,
                    linestyle,
                    problem_label,
                    # markevery=100,
                )
                err_plotter = plot_result(
                    err_plotter,
                    x,
                    err_alg_values,
                    subplot_index_z,
                    color,
                    None,  # marker,
                    None,  # markersize,
                    linestyle,
                    problem_label,
                    # markevery=100,
                )

    finalize_plot(case, dt, err_plotter, num_nodes, problems, problem_name, QI_list, subplot_indices, hook_for, solver_type=solver_type)
