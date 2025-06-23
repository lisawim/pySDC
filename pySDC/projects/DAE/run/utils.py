import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


QDELTAS = ["IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard"]

QI_PARALLEL = ["MIN-SR-NS", "MIN-SR-S"]
QI_SERIAL  = ["IE", "LU", "Picard"]

COLL_METHODS = ["RadauIIA5", "RadauIIA7", "RadauIIA9"]

class ExperimentConfig:
    qDelta_list = [
        "IE", "LU", "MIN-SR-NS", "MIN-SR-S", "Picard", "RadauIIA5", "RadauIIA7"
    ]
    sweeper_type_list = [
        "constrainedDAE", "embeddedDAE", "fullyImplicitDAE", "semiImplicitDAE"
    ]
    num_nodes = 6


def my_setup_mpl():
    "Setting up my personal settings for plotting."

    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams['legend.fontsize'] = 16

    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False

    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams["lines.solid_capstyle"] = "round"
    plt.rcParams["lines.markeredgewidth"] = 1.2
    plt.rcParams["lines.markeredgecolor"] = "black"

    # sets fig.tight_layout()
    plt.rcParams["figure.autolayout"] = True

    # plt.rcParams['mathtext.fontset'] = 'cm'
    # plt.rcParams['mathtext.rm'] = 'serif'


def my_plot_style_config():
    """Defines plot-specific stuff."""

    colors = {
        "constrainedDAE_IE": "gold",
        "constrainedDAE_LU": "orange",
        "constrainedDAE_MIN-SR-NS": "firebrick",
        "constrainedDAE_MIN-SR-S": "purple",
        "constrainedDAE_Picard": "dodgerblue",
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
        "fullyImplicitDAE_RadauIIA5": "palegreen",
        "fullyImplicitDAE_RadauIIA7": "black",
        "fullyImplicitDAE_RadauIIA9": "lightskyblue",
        "semiImplicitDAE_IE": "yellow",
        "semiImplicitDAE_LU": "darkmagenta",
        "semiImplicitDAE_MIN-SR-NS": "mediumseagreen",
        "semiImplicitDAE_MIN-SR-S": "khaki",
        "semiImplicitDAE_Picard": "red",
    }

    markers = {
        "constrainedDAE_IE": "o",
        "constrainedDAE_LU": "s",
        "constrainedDAE_MIN-SR-NS": "^",
        "constrainedDAE_MIN-SR-S": "d",
        "constrainedDAE_Picard": "*",
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

def setup_problem(problem_name, description, sweeper_type):
    """Sets up the problem with certain parameters."""

    if problem_name == "LINEAR-TEST":
        if sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAEConstrained as problem
        elif sweeper_type == "embeddedDAE":
            from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAEEmbedded as problem
        elif sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.problems.linearTestDAE import LinearTestDAE as problem
        elif sweeper_type == "semiImplicitDAE":
            from pySDC.projects.DAE.problems.linearTestDAE import SemiImplicitLinearTestDAE as problem

        description["problem_class"] = problem

        description["level_params"]["e_tol"] = 1e-13
        description["step_params"] = {"maxiter": 120}
        description["problem_params"] = {"solver_type": "direct"}

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
    

def get_sweeper_class_sdc(use_mpi: bool, sweeper_type: str):
    """Import the SDC sweeper class."""

    if use_mpi:
        if sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.sweepers.genericImplicitDAEMPI import genericImplicitConstrainedMPI as sweeper
        elif sweeper_type == "embeddedDAE":
            from pySDC.projects.DAE.sweepers.genericImplicitDAEMPI import genericImplicitEmbeddedMPI as sweeper
        elif sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.sweepers.fullyImplicitDAEMPI import FullyImplicitDAEMPI as sweeper
        elif sweeper_type == "semiImplicitDAE":
            from pySDC.projects.DAE.sweepers.semiImplicitDAEMPI import SemiImplicitDAEMPI as sweeper
    else:
        if sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained as sweeper
        elif sweeper_type == "embeddedDAE":
            from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded as sweeper
        elif sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE as sweeper
        elif sweeper_type == "semiImplicitDAE":
            from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE as sweeper

    return sweeper

def setup_sweeper_sdc(
        description,
        num_nodes=3,
        sweeper_type="constrainedDAE",
        QI="LU",
        use_mpi=False,
        **kwargs,
    ):
    """Sets up the SDC sweeper with certain parameters."""

    skip_residual_computation_default = ("IT_DOWN", "IT_UP", "IT_COARSE", "IT_FINE", "IT_CHECK")

    sdc_sweeper = get_sweeper_class_sdc(use_mpi, sweeper_type)
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
        assert num_nodes == comm.Get_size(), (
            f"Mismatch: {num_nodes} nodes, but {comm.Get_size()} MPI processes."
        )

    return description

def setup_sweeper_coll_method(description, sweeper_type="fullyImplicitDAE", num_nodes=3, QI="RadauIIA5"):
    """Sets up the RadauIIA sweeper with certain parameters."""

    assert sweeper_type == "fullyImplicitDAE", (
        "Incorrect problem type! For collocation method use 'fullyImplicitDAE'!"
    )

    skip_residual_computation_default = ("IT_DOWN", "IT_UP", "IT_COARSE", "IT_FINE", "IT_CHECK")

    coll_sweeper = get_sweeper_class_coll_method(QI)
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
        hook_class=[],
        **kwargs,
    ):

    description = {}
    description["level_params"] = {"dt": dt}

    description = setup_problem(problem_name, description, sweeper_type)

    if QI in QDELTAS:
        description = setup_sweeper_sdc(
            description, num_nodes, sweeper_type, QI, use_mpi, **kwargs
        )
    elif QI in COLL_METHODS:
        description = setup_sweeper_coll_method(
            description, sweeper_type, num_nodes, QI
        )

    # instantiate controller
    logger_level = kwargs.get("logger_level", 30)
    controller_params = {"logger_level": logger_level, "hook_class": hook_class}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    _, solution_stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return solution_stats
