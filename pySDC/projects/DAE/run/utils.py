from mpi4py import MPI
import time
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

from pySDC.projects.DAE.misc.methods_config import QI_SERIAL, QI_PARALLEL, RADAU_METHODS, RK_METHODS

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


def my_setup_mpl(fontsize=16):
    "Setting up my personal settings for plotting."

    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize

    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False

    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams["lines.solid_capstyle"] = "round"
    plt.rcParams["lines.markeredgewidth"] = 0.5
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["lines.markersize"] = 2.9

    # sets fig.tight_layout()
    plt.rcParams["figure.autolayout"] = True

    # plt.rcParams['mathtext.fontset'] = 'cm'
    # plt.rcParams['mathtext.rm'] = 'serif'


def my_plot_style_config():
    """Defines plot-specific stuff."""

    colors = {
        "constrainedDAE_EE": "forestgreen",
        "constrainedDAE_IE": "gold",
        "constrainedDAE_LU": "orange",
        "constrainedDAE_MIN-SR-NS": "firebrick",
        "constrainedDAE_MIN-SR-S": "purple",
        "constrainedDAE_Picard": "dodgerblue",
        "constrainedDAE_DOPRI5": "darkmagenta",
        "embeddedDAE_IE": "royalblue",
        "embeddedDAE_LU": "green",
        "embeddedDAE_MIN-SR-NS": "plum",
        "embeddedDAE_MIN-SR-S": "coral",
        "embeddedDAE_Picard": "darkcyan",
        "fullyImplicitDAE_IE": "limegreen",
        "fullyImplicitDAE_LU": "darkturquoise",
        "fullyImplicitDAE_MIN-SR-NS": "slategrey",
        "fullyImplicitDAE_MIN-SR-S": "pink",
        "fullyImplicitDAE_Picard": "sandybrown",
        "fullyImplicitDAE_RadauIIA5": "darkcyan",
        "fullyImplicitDAE_RadauIIA7": "black",
        "fullyImplicitDAE_RadauIIA9": "lightskyblue",
        "semiImplicitDAE_EE": "palevioletred",
        "semiImplicitDAE_IE": "yellow",
        "semiImplicitDAE_LU": "royalblue",
        "semiImplicitDAE_MIN-SR-NS": "mediumseagreen",
        "semiImplicitDAE_MIN-SR-S": "khaki",
        "semiImplicitDAE_Picard": "darkmagenta",
    }

    markers = {
        "constrainedDAE_EE": "D",
        "constrainedDAE_IE": "o",
        "constrainedDAE_LU": "s",
        "constrainedDAE_MIN-SR-NS": "^",
        "constrainedDAE_MIN-SR-S": "d",
        "constrainedDAE_Picard": "H",
        "constrainedDAE_DOPRI5": "p",
        "embeddedDAE_IE": "D",
        "embeddedDAE_LU": "<",
        "embeddedDAE_MIN-SR-NS": "H",
        "embeddedDAE_MIN-SR-S": "o",
        "embeddedDAE_Picard": "v",
        "fullyImplicitDAE_IE": "s",
        "fullyImplicitDAE_LU": "p",
        "fullyImplicitDAE_MIN-SR-NS": "X",
        "fullyImplicitDAE_MIN-SR-S": "*",
        "fullyImplicitDAE_Picard": "<",
        "fullyImplicitDAE_RadauIIA5": "D",
        "fullyImplicitDAE_RadauIIA7": "v",
        "fullyImplicitDAE_RadauIIA9": "H",
        "semiImplicitDAE_EE": "o",
        "semiImplicitDAE_IE": "d",
        "semiImplicitDAE_LU": "8",
        "semiImplicitDAE_MIN-SR-NS": "s",
        "semiImplicitDAE_MIN-SR-S": "^",
        "semiImplicitDAE_Picard": "D",
    }

    sweeper_labels = {
        "constrainedDAE": "SDC-C",
        "embeddedDAE": "SDC-E",
        "fullyImplicitDAE": "FI-SDC",
        "semiImplicitDAE": "SI-SDC",
    }

    return colors, markers, sweeper_labels


def newton_tol(dt, dt_ref=2.6e-3, tol_ref=8e-13):
    return tol_ref * (dt / dt_ref)


def set_correct_sweeper_type(sweeper_type, QI):
    if QI in RADAU_METHODS and sweeper_type != "fullyImplicitDAE":
        return "fullyImplicitDAE"
    elif QI in RK_METHODS and sweeper_type != "constrainedDAE":
        return "constrainedDAE"
    else:
        return sweeper_type


def setup_convergence_controllers(description, use_mpi, use_mpi_grouped):
    if use_mpi and use_mpi_grouped:
        from pySDC.projects.DAE.misc.estimate_embedded_error_mpi_grouped import EstimateEmbeddedErrorMPIGrouped

        description["convergence_controllers"] = {EstimateEmbeddedErrorMPIGrouped: {}}

    return description


def setup_problem(problem_name, QI, description, sweeper_type, **kwargs):
    """Sets up the problem with certain parameters."""

    dt = description["level_params"]["dt"]

    if problem_name == "ANDREWS-SQUEEZER":
        if sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import (
                AndrewsSqueezingMechanismDAEConstrained as problem,
            )
        elif sweeper_type == "embeddedDAE":
            from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import (
                AndrewsSqueezingMechanismDAEEmbedded as problem,
            )
        elif sweeper_type == "fullyImplicitDAE":
            if QI.startswith("RadauIIA"):
                from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import (
                    AndrewsSqueezingMechanismDAE_Radau as problem,
                )
            else:
                from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import (
                    AndrewsSqueezingMechanismDAE as problem,
                )
        elif sweeper_type == "semiImplicitDAE":
            from pySDC.projects.DAE.problems.andrewsSqueezingMechanism import (
                SemiImplicitAndrewsSqueezingMechanismDAE as problem,
            )

        description["level_params"]["e_tol"] = kwargs.get("e_tol", 1e-9)
        description["step_params"] = {"maxiter": kwargs.get("maxiter", 15)}
        description["problem_params"] = {"index": 1, "solver_type": "newton"}

    elif problem_name == "LINEAR-TEST":
        if sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAEConstrained as problem
        elif sweeper_type == "embeddedDAE":
            from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAEEmbedded as problem
        elif sweeper_type == "fullyImplicitDAE":
            if QI.startswith("RadauIIA"):
                from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAE_Radau as problem
            else:
                from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAE as problem
        elif sweeper_type == "semiImplicitDAE":
            from pySDC.projects.DAE.problems.linearTestDAE import SemiImplicitLinearTestDAE as problem

        description["level_params"]["e_tol"] = kwargs.get("e_tol", 1e-12)
        description["step_params"] = {"maxiter": kwargs.get("maxiter", 12)}
        description["problem_params"] = {"solver_type": "direct"}

    elif problem_name == "REACTION-DIFFUSION":
        if sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAEConstrained as problem
        elif sweeper_type == "fullyImplicitDAE":
            if QI.startswith("RadauIIA"):
                from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAE_Radau as problem
            else:
                from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAE as problem
        elif sweeper_type == "semiImplicitDAE":
            from pySDC.projects.DAE.problems.reactionDiffusionPDAE import SemiImplicitReactionDiffusionPDAE as problem

        description["level_params"]["e_tol"] = kwargs.get("e_tol", 1e-12)  # for M > 5 we need to set e_tol = 1e-12!
        description["step_params"] = {"maxiter": kwargs.get("maxiter", 15)}

        tol = newton_tol(dt)
        description["problem_params"] = {
            "nvars": kwargs.get("nvars", 256),
            "newton_tol": tol,
            "newton_maxiter": 7,
        }
        if not QI.startswith("RadauIIA"):
            description["problem_params"]["spectral"] = kwargs.get("spectral", True)

    description["problem_class"] = problem

    return description


def get_sweeper_class_coll_method(QI: str):
    """Import the collocation sweeper class."""

    if QI == "RadauIIA5":
        from pySDC.projects.DAE.sweepers.collocationDAE import RadauIIA5DAE as sweeper
    elif QI == "RadauIIA7":
        from pySDC.projects.DAE.sweepers.collocationDAE import RadauIIA7DAE as sweeper
    elif QI == "RadauIIA9":
        from pySDC.projects.DAE.sweepers.collocationDAE import RadauIIA9DAE as sweeper

    return sweeper


def get_sweeper_class_rk_method(QI: str):
    """Import the Runge-Kutta sweeper class."""

    if QI == "DOPRI5":
        from pySDC.projects.DAE.sweepers.rungeKuttaAllowingExplicitSolve import DOPRI5 as sweeper

    return sweeper


def get_sweeper_class_sdc(use_mpi: bool, sweeper_type: str, use_mpi_grouped: bool = False):
    """Import the SDC sweeper class."""

    if use_mpi:
        if not use_mpi_grouped:
            if sweeper_type == "constrainedDAE":
                from pySDC.projects.DAE.sweepers.genericImplicitDAEMPI import genericImplicitConstrainedMPI as sweeper
            elif sweeper_type == "embeddedDAE":
                from pySDC.projects.DAE.sweepers.genericImplicitDAEMPI import genericImplicitEmbeddedMPI as sweeper
            elif sweeper_type == "fullyImplicitDAE":
                from pySDC.projects.DAE.sweepers.fullyImplicitDAEMPI import FullyImplicitDAEMPI as sweeper
            elif sweeper_type == "semiImplicitDAE":
                from pySDC.projects.DAE.sweepers.semiImplicitDAEMPI import SemiImplicitDAEMPI as sweeper
            else:
                NotImplementedError(f"No MPI sweeper implemented for {sweeper_type}!")
        else:
            if sweeper_type == "constrainedDAE":
                from pySDC.projects.DAE.sweepers.genericImplicitConstrainedMPIGrouped import (
                    genericImplicitConstrainedMPIGrouped as sweeper,
                )
            elif sweeper_type == "semiImplicitDAE":
                from pySDC.projects.DAE.sweepers.semiImplicitDAEMPIGrouped import SemiImplicitDAEMPIGrouped as sweeper
            else:
                NotImplementedError(f"No grouped MPI sweeper implemented for {sweeper_type}!")

    else:
        if sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained as sweeper
        elif sweeper_type == "embeddedDAE":
            from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded as sweeper
        elif sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE as sweeper
        elif sweeper_type == "semiImplicitDAE":
            from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE as sweeper
        else:
            NotImplementedError(f"No sweeper implemented for {sweeper_type}!")

    return sweeper


def setup_sweeper_sdc(
    description,
    num_nodes=3,
    sweeper_type="constrainedDAE",
    QI="LU",
    use_mpi=False,
    use_mpi_grouped=False,
    **kwargs,
):
    """Sets up the SDC sweeper with certain parameters."""

    skip_residual_computation_default = ("IT_DOWN", "IT_UP", "IT_COARSE", "IT_FINE", "IT_CHECK")

    sdc_sweeper = get_sweeper_class_sdc(use_mpi, sweeper_type, use_mpi_grouped)
    description["sweeper_class"] = sdc_sweeper

    description["sweeper_params"] = {
        "quad_type": "RADAU-RIGHT",
        "num_nodes": num_nodes,
        "QI": QI,
        "initial_guess": kwargs.get("initial_guess", "spread"),
        "skip_residual_computation": kwargs.get("skip_residual_computation", skip_residual_computation_default),
    }

    description["level_params"].update({"nsweeps": 1, "restol": -1})

    # MPI-related checks
    if use_mpi and "comm" in kwargs:
        comm = kwargs["comm"]
        description["sweeper_params"]["comm"] = comm
        size = comm.Get_size()
        if not use_mpi_grouped:
            assert size == num_nodes, f"Mismatch: {num_nodes} nodes, but {size} MPI processes."
        else:
            assert size <= num_nodes, f"Need comm.size <= num_nodes. Got {size} MPI processes and {num_nodes=}."

    return description


def setup_sweeper_coll_method(description, sweeper_type="fullyImplicitDAE", QI="RadauIIA5"):
    """Sets up the RadauIIA sweeper with certain parameters."""

    if sweeper_type != "fullyImplicitDAE":
        sweeper_type == "fullyImplicitDAE"
        logger.warning(f"For {QI} sweeper_type is set to 'fullyImplicitDAE'")

    skip_residual_computation_default = ("IT_DOWN", "IT_UP", "IT_COARSE", "IT_FINE", "IT_CHECK")

    coll_sweeper = get_sweeper_class_coll_method(QI)
    description["sweeper_class"] = coll_sweeper

    description["level_params"].update({"restol": -1, "e_tol": -1, "nsweeps": 1})
    description["step_params"]["maxiter"] = 1
    description["sweeper_params"] = {"skip_residual_computation": skip_residual_computation_default}

    return description


def setup_sweeper_rk_method(description, sweeper_type="fullyImplicitDAE", QI="RadauIIA5"):
    """Sets up the RadauIIA sweeper with certain parameters."""

    if sweeper_type != "constrainedDAE":
        sweeper_type == "constrainedDAE"
        logger.warning(f"For {QI} sweeper_type is set to 'constrainedDAE'")

    skip_residual_computation_default = ("IT_DOWN", "IT_UP", "IT_COARSE", "IT_FINE", "IT_CHECK")

    coll_sweeper = get_sweeper_class_rk_method(QI)
    description["sweeper_class"] = coll_sweeper

    description["level_params"].update({"restol": -1, "e_tol": -1, "nsweeps": 1})
    description["step_params"]["maxiter"] = 1
    description["sweeper_params"] = {"skip_residual_computation": skip_residual_computation_default}

    return description


def compute_solution(
    problem_name,
    t0,
    dt,
    Tend,
    num_nodes,
    QI,
    sweeper_type,
    use_mpi=False,
    use_mpi_grouped=False,
    hook_class=[],
    measure=True,
    **kwargs,
):
    comm = kwargs.get("comm", None)

    description = {}
    description["level_params"] = {"dt": dt}

    description = setup_convergence_controllers(description, use_mpi, use_mpi_grouped)

    corrected_sweeper_type = set_correct_sweeper_type(sweeper_type, QI)

    description = setup_problem(problem_name, QI, description, corrected_sweeper_type, **kwargs)

    if QI in QI_SERIAL + QI_PARALLEL:
        description = setup_sweeper_sdc(
            description, num_nodes, corrected_sweeper_type, QI, use_mpi, use_mpi_grouped, **kwargs
        )
    elif QI in RADAU_METHODS:
        description = setup_sweeper_coll_method(description, corrected_sweeper_type, QI)
    elif QI in RK_METHODS:
        description = setup_sweeper_rk_method(description, corrected_sweeper_type, QI)

    # instantiate controller
    logger_level = kwargs.get("logger_level", 30)
    controller_params = {"logger_level": logger_level, "hook_class": hook_class}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # Using MPI does imply adding a communicator
    if use_mpi:
        if comm is None:
            comm = MPI.COMM_WORLD
        comm.Barrier()
        t_start = MPI.Wtime()
    elif measure:
        t_start = time.time()

    # call main function to get things done...
    _, solution_stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    if use_mpi:
        comm.Barrier()
        t_end = MPI.Wtime()
        return (t_end - t_start), solution_stats
    elif measure:
        t_end = time.time()
        return (t_end - t_start), solution_stats

    return solution_stats
