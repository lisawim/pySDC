from mpi4py import MPI
import time
import matplotlib.pyplot as plt
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

from pySDC.core.hooks import Hooks
from pySDC.core.sweeper import Sweeper
from pySDC.projects.DAE.misc.methods_config import QI_SERIAL, QI_PARALLEL, RADAU_METHODS, RK_METHODS

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


def my_setup_mpl(fontsize: int = 16) -> None:
    """
    Setting up my personal settings for plotting.

    Parameters
    ----------
    fontsize : int, optional
        Fontsize used in plot.
    """

    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize
    plt.rcParams['axes.linewidth'] = 0.5

    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6
    plt.rcParams["xtick.minor.width"] = 0.6
    plt.rcParams["ytick.minor.width"] = 0.6
    plt.rcParams["xtick.major.size"] = 2.5
    plt.rcParams["ytick.major.size"] = 2.5
    plt.rcParams["xtick.minor.size"] = 1
    plt.rcParams["ytick.minor.size"] = 1

    plt.rcParams['lines.linewidth'] = 0.9
    plt.rcParams["lines.solid_capstyle"] = "round"
    plt.rcParams["lines.markeredgewidth"] = 0.5
    plt.rcParams["lines.markeredgecolor"] = "black"
    plt.rcParams["lines.markersize"] = 2.9

    # sets fig.tight_layout()
    plt.rcParams["figure.autolayout"] = True


def my_plot_style_config() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    r"""
    Defines plot-specific stuff. Keys have the form ``f"{sweeper_type}_{QI}"``.

    Returns
    -------
    colors : dict
        Sweeper specific colors.
    markers : dict
        Sweeper specific markers.
    sweeper_labels : dict
        Labels that denote the sweepers.
    """

    colors = {
        "imexConstrainedDAE_EE": "green",
        "imexConstrainedDAE_IE": "lightskyblue",
        "imexConstrainedDAE_LU": "pink",
        "imexConstrainedDAE_MIN-SR-NS": "coral",
        "imexConstrainedDAE_MIN-SR-S": "plum",
        "constrainedDAE_EE": "forestgreen",
        "constrainedDAE_IE": "gold",
        "constrainedDAE_LU": "orange",
        "constrainedDAE_MIN-SR-NS": "firebrick",
        "constrainedDAE_MIN-SR-S": "purple",
        "constrainedDAE_MIN-SR-FLEX": "darkgrey",
        "constrainedDAE_Picard": "dodgerblue",
        "constrainedDAE_DOPRI5": "darkmagenta",
        "embeddedDAE_EE": "black",
        "embeddedDAE_IE": "firebrick",
        "embeddedDAE_LU": "green",
        "embeddedDAE_MIN-SR-NS": "plum",
        "embeddedDAE_MIN-SR-S": "coral",
        "embeddedDAE_Picard": "darkturquoise",
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
        "semiImplicitDAE_MIN-SR-S": "gold",
        "semiImplicitDAE_MIN-SR-FLEX": "lightskyblue",
        "semiImplicitDAE_Picard": "darkmagenta",
    }

    markers = {
        "imexConstrainedDAE_EE": "*",
        "imexConstrainedDAE_IE": "s",
        "imexConstrainedDAE_LU": "^",
        "imexConstrainedDAE_MIN-SR-NS": "o",
        "imexConstrainedDAE_MIN-SR-S": "d",
        "constrainedDAE_EE": "D",
        "constrainedDAE_IE": "o",
        "constrainedDAE_LU": "s",
        "constrainedDAE_MIN-SR-NS": "^",
        "constrainedDAE_MIN-SR-S": "d",
        "constrainedDAE_MIN-SR-FLEX": "*",
        "constrainedDAE_Picard": "H",
        "constrainedDAE_DOPRI5": "p",
        "embeddedDAE_EE": "s",
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
        "semiImplicitDAE_MIN-SR-S": "*",
        "semiImplicitDAE_MIN-SR-FLEX": "H",
        "semiImplicitDAE_Picard": "D",
    }

    sweeper_labels = {
        "imexConstrainedDAE": "IMEX-SDC-C",
        "constrainedDAE": "SDC-C",
        "embeddedDAE": "SDC-E",
        "fullyImplicitDAE": "FI-SDC",
        "semiImplicitDAE": "SI-SDC",
        "SPP": "SDC-SPP",
    }

    return colors, markers, sweeper_labels


def newton_tol(dt: float, dt_ref: float = 2.6e-3, tol_ref: float = 8e-13) -> float:
    r"""
    Newton tolerance is coupled to time step size ``dt`` with reference step size ``dt_ref``
    and ``tol_ref``. The Newton tolerance is then defined by

    .. math::
        tol_{\mathrm{newton}} = \frac{tol_{\mathrm{ref}} \Delta t}{\Delta t_{\mathrm{ref}}}

    Parameters
    ----------
    dt : float
        Time step size.
    dt_ref : float, optional
        Reference time step size.
    tol_ref : float, optional
        Reference tolerance that is obtained if :math:`\Delta t_{\mathrm{ref}} = \Delta t`.

    Returns
    -------
    : float
        Newton tolerance.
    """
    return tol_ref * (dt / dt_ref)


def set_correct_sweeper_type(sweeper_type: str, QI: str) -> str:
    """
    Depending on ``QI`` the sweeper_type is adjusted. This is needed when ``QI`` refers
    to a Runge-Kutta method and avoid raising an error.

    Parameters
    ----------
    sweeper_type : str
        Original sweeper type.
    QI : str
        Indicates the method to choose.

    Returns
    -------
    : str
        Adjusted sweeper type.
    """

    if QI in RADAU_METHODS and sweeper_type != "fullyImplicitDAE":
        return "fullyImplicitDAE"
    elif QI in RK_METHODS and sweeper_type != "constrainedDAE":
        return "constrainedDAE"
    else:
        return sweeper_type


def setup_convergence_controllers(
    description: dict[str, dict[str, Any]], stop_at_accuracy_for_speedup: bool, use_mpi: bool, use_mpi_grouped: bool
) -> dict[str, dict[str, Any]]:
    r"""
    Convergence controllers for run are added to the ``description`` dictionary. For the
    MPI grouped sweepers an alternative variant to correctly compute the increment in each iteration
    is added.

    Parameters
    ----------
    description : dict
        Description of all parameters for the run.
    stop_at_accuracy_for_speedup : bool
        If True, the CheckExactError convergence controller is added. The convergence controller checks
        if the numerical solution achieves a certain accuracy and if so, step is converged. Note that method
        ``u_exact`` needs to be implemented for a problem.
    use_mpi : bool
        Indicate the usage of MPI.
    use_mpi_grouped : bool
        Indicates the usage of grouped MPI.

    Returns
    -------
    description : dict
        Updated description.
    """

    if use_mpi and use_mpi_grouped:
        from pySDC.projects.DAE.misc.estimate_embedded_error_mpi_grouped import EstimateEmbeddedErrorMPIGrouped

        description["convergence_controllers"] = {EstimateEmbeddedErrorMPIGrouped: {}}

    if stop_at_accuracy_for_speedup:
        from pySDC.projects.DAE.misc.check_exact_error import CheckExactError

        description["convergence_controllers"] = {CheckExactError: {}}

    return description


def setup_problem(
    problem_name: str,
    QI: str,
    description: dict[str, dict[str, Any]],
    sweeper_type: str,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """
    Sets up the problem with certain parameters.

    Parameters
    ----------
    problem_name : str
        Name of the problem.
    QI : str
        Indicates the method to use.
    description : dict
        Description of all parameters for the run.
    sweeper_type : str
        Sweeper type.

    Returns
    -------
    description : dict
        Updated description.
    """

    dt = description["level_params"]["dt"]
    stop_at_accuracy_for_speedup = kwargs.get("stop_at_accuracy_for_speedup", False)

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

        maxiter = kwargs.get("maxiter", 20)
        e_tol = kwargs.get("e_tol", 1e-4) if stop_at_accuracy_for_speedup else kwargs.get("e_tol", 1e-9)
        description["level_params"]["e_tol"] = e_tol
        description["step_params"] = {"maxiter": maxiter}
        description["problem_params"] = {"index": 1, "solver_type": "newton"}

    elif problem_name == "BATTERY":
        if sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.problems.batteryDAE import BatteryDAE as problem
        else:
            raise NotImplementedError("Battery problem only implemented for fully implicit DAE sweeper.")

        maxiter = kwargs.get("maxiter", 10)
        e_tol = kwargs.get("e_tol", 1e-12)
        description["level_params"]["e_tol"] = e_tol
        description["step_params"] = {"maxiter": maxiter}
        description["problem_params"] = {}
    
    elif problem_name == "BUCK-CONVERTER":
        if sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.problems.buckConverterDAE import BuckConverterDAE as problem
        else:
            raise NotImplementedError("Buck converter problem only implemented for fully implicit DAE sweeper.")

        maxiter = kwargs.get("maxiter", 10)
        e_tol = kwargs.get("e_tol", 1e-12)
        description["level_params"]["e_tol"] = e_tol
        description["step_params"] = {"maxiter": maxiter}
        description["problem_params"] = {"duty": kwargs.get("duty", 0.5)}

    elif problem_name == "DISC-TEST":
        if sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.problems.discontinuousTestDAE import DiscontinuousTestDAE as problem
        else:
            raise NotImplementedError("Discontinuous test problem only implemented for fully implicit DAE sweeper.")

        maxiter = kwargs.get("maxiter", 12)
        e_tol = kwargs.get("e_tol", 1e-12)
        description["level_params"]["e_tol"] = e_tol
        description["step_params"] = {"maxiter": maxiter}
        description["problem_params"] = {}

    elif problem_name == "LINEAR-TEST":
        eps = kwargs.get("eps", 0.0)
        if eps > 0.0:
            from pySDC.implementations.problem_classes.linearTestSPP import LinearTestSPP as problem
            # from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
        elif sweeper_type == "constrainedDAE":
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

        maxiter = kwargs.get("maxiter", 12)
        e_tol = kwargs.get("e_tol", 1e-4) if stop_at_accuracy_for_speedup else kwargs.get("e_tol", 1e-12)
        description["level_params"]["e_tol"] = e_tol
        description["step_params"] = {"maxiter": maxiter}
        description["problem_params"] = {"solver_type": "direct"}
        if eps > 0.0:
            description["problem_params"]["eps"] = eps

    elif problem_name == "PILINE":
        if sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.problems.pilineDAE import PilineDAE as problem
        else:
            raise NotImplementedError("Piline problem only implemented for fully implicit DAE sweeper.")

        maxiter = kwargs.get("maxiter", 10)
        e_tol = kwargs.get("e_tol", 1e-12)
        description["level_params"]["e_tol"] = e_tol
        description["step_params"] = {"maxiter": maxiter}
        description["problem_params"] = {}

    elif problem_name == "REACTION-DIFFUSION":
        if sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAEConstrained as problem
        elif sweeper_type == "imexConstrainedDAE":
            from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAE_IMEX as problem
        elif sweeper_type == "fullyImplicitDAE":
            if QI.startswith("RadauIIA"):
                from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAE_Radau as problem
            else:
                from pySDC.projects.DAE.problems.reactionDiffusionPDAE import ReactionDiffusionPDAE as problem
        elif sweeper_type == "semiImplicitDAE":
            from pySDC.projects.DAE.problems.reactionDiffusionPDAE import SemiImplicitReactionDiffusionPDAE as problem

        maxiter = kwargs.get("maxiter", 25)
        e_tol = kwargs.get("e_tol", 1e-5) if stop_at_accuracy_for_speedup else kwargs.get("e_tol", 1e-13)
        description["level_params"]["e_tol"] = e_tol
        description["step_params"] = {"maxiter": maxiter}

        tol = newton_tol(dt)
        description["problem_params"] = {
            "nvars": kwargs.get("nvars", 256),
            "newton_tol": tol,
            "newton_maxiter": 7,
        }
        if not QI.startswith("RadauIIA"):
            description["problem_params"]["spectral"] = kwargs.get("spectral", True)

    elif problem_name == "WSCC9":
        if sweeper_type == "fullyImplicitDAE":
            from pySDC.projects.DAE.problems.wscc9BusSystem import WSCC9BusSystem as problem
        else:
            raise NotImplementedError("WSCC9 problem only implemented for fully implicit DAE sweeper.")

        maxiter = kwargs.get("maxiter", 10)
        e_tol = kwargs.get("e_tol", 1e-12)
        description["level_params"]["e_tol"] = e_tol
        description["step_params"] = {"maxiter": maxiter}
        description["problem_params"] = {}

    description["problem_class"] = problem

    return description


def get_sweeper_class_coll_method(QI: str) -> type[Sweeper]:
    """
    Import of the collocation sweeper class.

    Parameters
    ----------
    QI : str
        Indicates the Radau method.

    Returns
    -------
    : Sweeper
        Imported Radau sweeper class.
    """

    if QI == "RadauIIA5":
        from pySDC.projects.DAE.sweepers.collocationDAE import RadauIIA5DAE as sweeper
    elif QI == "RadauIIA7":
        from pySDC.projects.DAE.sweepers.collocationDAE import RadauIIA7DAE as sweeper
    elif QI == "RadauIIA9":
        from pySDC.projects.DAE.sweepers.collocationDAE import RadauIIA9DAE as sweeper

    return sweeper


def get_sweeper_class_rk_method(QI: str) -> type[Sweeper]:
    """
    Import of the Runge-Kutta sweeper class.

    Parameters
    ----------
    QI : str
        Indicates the Runge-Kutta method.

    Returns
    -------
    : Sweeper
        Imported Runge-Kutta sweeper class.
    """

    if QI == "DOPRI5":
        from pySDC.projects.DAE.sweepers.rungeKuttaAllowingExplicitSolve import DOPRI5 as sweeper

    return sweeper


def get_sweeper_class_sdc(use_mpi: bool, sweeper_type: str, use_mpi_grouped: bool = False, **kwargs: Any) -> type[Sweeper]:
    """
    Import of the SDC sweeper class.

    Parameters
    ----------
    use_mpi : bool
        Indicates usage of MPI.
    sweeper_type : str
        Sweeper type.
    use_mpi_grouped : bool
        Indicates usage of grouped MPI.

    Returns
    -------
    : Sweeper
        Imported sweeper class.
    """

    eps = kwargs.get("eps", 0.0)
    if use_mpi:
        if not use_mpi_grouped:
            if eps > 0.0:
                from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper
            elif sweeper_type == "constrainedDAE":
                from pySDC.projects.DAE.sweepers.genericImplicitDAEMPI import genericImplicitConstrainedMPI as sweeper
            # elif sweeper_type == "imexConstrainedDAE":
            #     from pySDC.projects.DAE.sweepers.imex_sdc_c import imex_sdc_c as sweeper
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
        if eps > 0.0:
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper
        elif sweeper_type == "constrainedDAE":
            from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained as sweeper
        elif sweeper_type == "imexConstrainedDAE":
            from pySDC.projects.DAE.sweepers.imex_sdc_c import imex_sdc_c as sweeper
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
    description: dict[str, dict[str, Any]],
    num_nodes: int = 3,
    sweeper_type: str = "constrainedDAE",
    QI: str = "LU",
    use_mpi: bool = False,
    use_mpi_grouped: bool = False,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """
    Sets up the SDC sweeper with certain parameters.

    Parameters
    ----------
    description : dict
        Description of all parameters for the run.
    num_nodes : int, optional
        Number of collocation nodes.
    sweeper_type : str, optional
        Sweeper type.
    QI : str, optional
        Indicates the choice of the preconditioner.
    use_mpi : bool, optional
        Indicates usage of MPI.
    use_mpi_grouped : bool, optional
        Indicates usage of grouped MPI.

    Returns
    -------
    description : dict
        Updated description.
    """

    skip_residual_computation_default = ("IT_DOWN", "IT_UP", "IT_COARSE", "IT_FINE", "IT_CHECK")

    sdc_sweeper = get_sweeper_class_sdc(use_mpi, sweeper_type, use_mpi_grouped, **kwargs)
    description["sweeper_class"] = sdc_sweeper

    description["sweeper_params"] = {
        "quad_type": "RADAU-RIGHT",
        "num_nodes": num_nodes,
        "QI": QI,
        "QE": kwargs.get("QE", "EE"),
        "initial_guess": kwargs.get("initial_guess", "spread"),
        "skip_residual_computation": kwargs.get("skip_residual_computation", skip_residual_computation_default),
    }

    nsweeps = kwargs.get("nsweeps", 1)
    description["level_params"].update({"nsweeps": nsweeps, "restol": -1})

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


def setup_sweeper_coll_method(
    description: dict[str, dict[str, Any]], sweeper_type: str = "fullyImplicitDAE", QI: str = "RadauIIA5"
) -> dict[str, dict[str, Any]]:
    r"""
    Sets up the RadauIIA sweeper with certain parameters.

    Parameters
    ----------
    description : dict
        Description of all parameters for the run.
    sweeper_type : str, optional
        Sweeper type. Default is ``"fullyImplicitDAE"``.
    QI : str, optional
        Indicates the choice of the Radau method. Default is ``"RadauIIA5"`` that
        defines the Radau IIA method of order 5.

    Returns
    -------
    description : dict
        Updated description.
    """

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


def setup_sweeper_rk_method(
    description: dict[str, dict[str, Any]], sweeper_type: str = "constrainedDAE", QI: str = "DOPRI5"
) -> dict[str, dict[str, Any]]:
    r"""
    Sets up the Runge-Kutta sweeper with certain parameters.

    Parameters
    ----------
    description : dict
        Description of all parameters for the run.
    sweeper_type : str, optional
        Sweeper type. Default is ``"constrainedDAE"``.
    QI : str, optional
        Indicates the choice of the Runge-Kutta method. Default is ``"DOPRI5"`` that
        defines the half-explicit Runge-Kutta method using coefficients of Dormand & Prince.

    Returns
    -------
    description : dict
        Updated description.
    """

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
    problem_name: str,
    t0: float,
    dt: float,
    Tend: float,
    num_nodes: int,
    QI: str,
    sweeper_type: str,
    use_mpi: bool = False,
    use_mpi_grouped: bool = False,
    hook_class: list[Hooks] = [],
    measure: bool = True,
    return_uend: bool = False,
    **kwargs: Any,
) -> tuple[Optional[float], dict]:
    r"""
    Computes the numerical solution. Main things are done in this routine.

    Parameters
    ----------
    problem_name : str
        Name of the problem.
    t0 : float
        Initial time.
    dt : float
        Time step size.
    Tend : float
        End time.
    num_nodes : int
        Number of collocation nodes.
    QI : str
        Indicates the method.
    sweeper_type : str
        Sweeper type.
    use_mpi : bool, optional
        Indicates usage of MPI.
    use_mpi_grouped : bool, optional
        Indicates usage of grouped MPI.
    hook_class : list of Hooks, optional
        Contains the hook classes to log specific data during the run. Default
        is an empty list.
    measure : bool, optional
        If True, runtime is measured.

    Returns
    -------
    runtime : float
        Runtime of the simulation.
    uend : dtype_u, optional
        Numerical solution at time ``Tend``. Only returned if ``return_uend`` is True.
    solution_stats : dict
        Statistics of the run.

    Notes
    -----
    The returned tuple depends on the flags ``use_mpi``, ``measure`` and
    ``return_uend``:

    - If ``use_mpi`` or ``measure`` is True:

    - If ``return_uend`` is False:
        ``(runtime, solution_stats)``
    - If ``return_uend`` is True:
        ``(runtime, uend, solution_stats)``

    - If neither ``use_mpi`` nor ``measure`` is True:

    - If ``return_uend`` is False:
        ``solution_stats``
    - If ``return_uend`` is True:
        ``(uend, solution_stats)``
    """

    comm = kwargs.get("comm", None)

    description = {}
    description["level_params"] = {"dt": dt}

    stop_at_accuracy_for_speedup = kwargs.get("stop_at_accuracy_for_speedup", False)
    description = setup_convergence_controllers(description, stop_at_accuracy_for_speedup, use_mpi, use_mpi_grouped)

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
    uend, solution_stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    runtime = None

    if use_mpi:
        comm.Barrier()
        runtime = MPI.Wtime() - t_start
    elif measure:
        runtime = time.time() - t_start

    if runtime is not None:
        if return_uend:
            return runtime, uend, solution_stats
        return runtime, solution_stats

    # no runtime measured
    if return_uend:
        return uend, solution_stats

    return solution_stats
