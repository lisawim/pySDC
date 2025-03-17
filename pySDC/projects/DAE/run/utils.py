import numpy as np
import importlib

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

SDC_METHODS = ['IE', 'LU', 'MIN-SR-NS', 'MIN-SR-S', 'MIN-SR-FLEX', 'MIN', 'MIN3', 'Picard']
RK_METHODS = ["BE", "DIRK43", "EDIRK4", "DIRK", "DIRK5", "DIRK5_2", "ESDIRK53", "SDIRK3"]
COLLOCATION_METHODS = ["RadauIIA5", "RadauIIA7", "RadauIIA9"]


problemMapping = {
    "ANDREWS-SQUEEZER": {
        "SPP": {
            "NotImplemented": True,  # Indicate this case is not implemented
        },
        "embeddedDAE": {
            "module": "pySDC.projects.DAE.problems.andrewsSqueezingMechanism",
            "class": "AndrewsSqueezingMechanismDAEEmbedded",
        },
        "constrainedDAE": {
            "module": "pySDC.projects.DAE.problems.andrewsSqueezingMechanism",
            "class": "AndrewsSqueezingMechanismDAEConstrained",
        },
        "description": {
            "index": 3,
            "e_tol": 1e-13,
            "maxiter": 80,
            "newton_tol": 1e-12,
            "newton_maxiter": 100,
        },
    },
    "LINEAR-TEST": {
        "SPP": {
            "module": "pySDC.implementations.problem_classes.singularPerturbed",
            "class": "LinearTestSPP",
        },
        "SPP-IMEX": {
            "module": "pySDC.implementations.problem_classes.singularPerturbed",
            "class": "LinearTestSPPIMEX",
        },
        "gmresSPP": {
            "module": "pySDC.implementations.problem_classes.singularPerturbed",
            "class": "LinearTestSPP",
        },
        "SPP-yp": {
            "module": "pySDC.implementations.problem_classes.singularPerturbed",
            "class": "LinearTestSPP_YP",
        },
        "embeddedDAE": {
            "module": "pySDC.projects.DAE.problems.linearTestDAE",
            "class": "LinearTestDAEEmbedded",
        },
        "constrainedDAE": {
            "module": "pySDC.projects.DAE.problems.linearTestDAE",
            "class": "LinearTestDAEConstrained",
        },
        "fullyImplicitDAE": {
            "module": "pySDC.projects.DAE.problems.linearTestDAE",
            "class": "LinearTestDAE",
        },
        "semiImplicitDAE": {
            "module": "pySDC.projects.DAE.problems.linearTestDAE",
            "class": "SemiImplicitLinearTestDAE",
        },
        "description": {
            "e_tol": 1e-13,
            "maxiter": 120,
            "newton_tol": 1e-12,
            "newton_maxiter": 20,
            "solver_type": "newton",
        },
    },
    "MICHAELIS-MENTEN": {
        "SPP": {
            "module": "pySDC.implementations.problem_classes.singularPerturbed",
            "class": "MichaelisMentenSPP",
        },
        "SPP-yp": {
            "module": "pySDC.implementations.problem_classes.singularPerturbed",
            "class": "MichaelisMentenSPP_YP",
        },
        "embeddedDAE": {
            "module": "pySDC.projects.DAE.problems.michaelisMentenDAE",
            "class": "MichaelisMentenEmbedded",
        },
        "constrainedDAE": {
            "module": "pySDC.projects.DAE.problems.michaelisMentenDAE",
            "class": "MichaelisMentenConstrained",
        },
        "fullyImplicitDAE": {
            "module": "pySDC.projects.DAE.problems.michaelisMentenDAE",
            "class": "MichaelisMentenDAE",
        },
        "semiImplicitDAE": {
            "module": "pySDC.projects.DAE.problems.michaelisMentenDAE",
            "class": "SemiImplicitMichaelisMentenDAE",
        },
        "description": {
            "e_tol": 1e-13,
            "maxiter": 100,
            "newton_tol": 1e-12,
            "newton_maxiter": 20,
        },
    },
    "PROTHERO-ROBINSON": {
        "SPP": {
            "module": "pySDC.implementations.problem_classes.singularPerturbed",
            "class": "ProtheroRobinsonSPP",
        },
        "SPP-yp": {
            "NotImplemented": True,  # Indicate this case is not implemented
        },
        "embeddedDAE": {
            "module": "pySDC.projects.DAE.problems.protheroRobinsonDAE",
            "class": "ProtheroRobinsonDAEEmbedded",
        },
        "constrainedDAE": {
            "module": "pySDC.projects.DAE.problems.protheroRobinsonDAE",
            "class": "ProtheroRobinsonDAEConstrained",
        },
        "fullyImplicitDAE": {
            "module": "pySDC.projects.DAE.problems.protheroRobinsonDAE",
            "class": "ProtheroRobinsonDAE",
        },
        "semiImplicitDAE": {
            "module": "pySDC.projects.DAE.problems.protheroRobinsonDAE",
            "class": "ProtheroRobinsonDAE",
        },
        "description": {
            "e_tol": 1e-13,
            "maxiter": 40,
            "newton_tol": 1e-12,
            "newton_maxiter": 20,
        },
    },
    "VAN-DER-POL": {
        "SPP": {
            "module": "pySDC.implementations.problem_classes.singularPerturbed",
            "class": "VanDerPol",
        },
        "embeddedDAE": {
            "module": "pySDC.projects.DAE.problems.vanDerPolDAE",
            "class": "VanDerPolEmbedded",
        },
        "constrainedDAE": {
            "module": "pySDC.projects.DAE.problems.vanDerPolDAE",
            "class": "VanDerPolConstrained",
        },
        "description": {
            "e_tol": 5e-13,
            "maxiter": 20,
            "newton_tol": 1e-12,
            "newton_maxiter": 100,
        },
    },
}


# Non-MPI SDC sweeper mappings
SDC_SWEEPER_MAPPING = {
    "SPP": {
        "module": "pySDC.implementations.sweeper_classes.generic_implicit",
        "class": "generic_implicit",
    },
    "SPP-IMEX": {
        "module": "pySDC.implementations.sweeper_classes.imex_1st_order",
        "class": "imex_1st_order",
    },
    "SPP-yp": {
        "module": "pySDC.projects.DAE.sweepers.fullyImplicitDAE",
        "class": "FullyImplicitDAE",
    },
    "gmresSPP": {
        "module": "pySDC.implementations.sweeper_classes.gmres_sdc",
        "class": "GMRESSDC",
    },
    "embeddedDAE": {
        "module": "pySDC.projects.DAE.sweepers.genericImplicitDAE",
        "class": "genericImplicitEmbedded",
    },
    "constrainedDAE": {
        "module": "pySDC.projects.DAE.sweepers.genericImplicitDAE",
        "class": "genericImplicitConstrained",
    },
    "fullyImplicitDAE": {
        "module": "pySDC.projects.DAE.sweepers.fullyImplicitDAE",
        "class": "FullyImplicitDAE",
    },
    "semiImplicitDAE": {
        "module": "pySDC.projects.DAE.sweepers.semiImplicitDAE",
        "class": "SemiImplicitDAE",
    },
}

# MPI SDC sweeper mappings
SDC_SWEEPER_MAPPING_MPI = {
    "SPP": {
        "module": "pySDC.implementations.sweeper_classes.generic_implicit_MPI",
        "class": "generic_implicit_MPI",
    },
    "SPP-IMEX": {
        "module": "pySDC.implementations.sweeper_classes.imex_1st_order_MPI",
        "class": "imex_1st_order_MPI",
    },
    "SPP-yp": {
        "module": "pySDC.projects.DAE.sweepers.fullyImplicitDAEMPI",
        "class": "FullyImplicitDAEMPI",
    },
    "embeddedDAE": {
        "module": "pySDC.playgrounds.DAE.genericImplicitDAEMPI",
        "class": "genericImplicitEmbeddedMPI",
    },
    "constrainedDAE": {
        "module": "pySDC.playgrounds.DAE.genericImplicitDAEMPI",
        "class": "genericImplicitConstrainedMPI",
    },
    "fullyImplicitDAE": {
        "module": "pySDC.projects.DAE.sweepers.fullyImplicitDAEMPI",
        "class": "FullyImplicitDAEMPI",
    },
    "semiImplicitDAE": {
        "module": "pySDC.projects.DAE.sweepers.semiImplicitDAEMPI",
        "class": "SemiImplicitDAEMPI",
    },
}

# RK method mappings based on problem type (No MPI for RK)
RK_SWEEPER_MAPPING = {
    "constrained": {
        "module": "pySDC.playgrounds.DAE.RungeKuttaConstrainedDAE",
        "class": {
            "BE": "BackwardEulerConstrained",
            "DIRK43": "DIRK43_2Constrained",
            "EDIRK4": "EDIRK4Constrained",
            "DIRK": "DIRKConstrained",
            "DIRK5": "DIRK5Constrained",
            "DIRK5_2": "DIRK5_2Constrained",
            "ESDIRK53": "ESDIRK53Constrained",
            "SDIRK3": "SDIRK3",
        },
    },
    "fullyImplicit": {
        "module": "pySDC.projects.DAE.sweepers.rungeKuttaDAE",
        "class": {
            "BE": "BackwardEulerDAE",
            "RadauIIA5": "RadauIIA5DAE",
            "RadauIIA7": "RadauIIA7DAE",
            "RadauIIA9": "RadauIIA9DAE",
        },
    },
    "semiImplicit": {
        "module": "pySDC.projects.DAE.sweepers.rungeKuttaDAE",
        "class": {
            "BE": "BackwardEulerDAE",
            "RadauIIA5": "RadauIIA5DAE",
            "RadauIIA7": "RadauIIA7DAE",
            "RadauIIA9": "RadauIIA9DAE",
        },
    },
}

# Collocation method mappings based on problem type
COLLOCATION_SWEEPER_MAPPING = {
    "constrained": {
        "module": "pySDC.projects.DAE.sweepers.collocationConstrainedDAE",
        "class": {
            "RadauIIA5": "RadauIIA5Constrained",
            "RadauIIA7": "RadauIIA7Constrained",
        },
    },
    "fullyImplicit": {
        "module": "pySDC.projects.DAE.sweepers.collocationDAE",
        "class": {
            "RadauIIA5": "RadauIIA5DAE",
            "RadauIIA7": "RadauIIA7DAE",
            "RadauIIA9": "RadauIIA9DAE",
        },
    },
    "semiImplicit": {
        "module": "pySDC.projects.DAE.sweepers.collocationDAE",
        "class": {
            "RadauIIA5": "RadauIIA5DAE",
            "RadauIIA7": "RadauIIA7DAE",
            "RadauIIA9": "RadauIIA9DAE",
        },
    },
}


def getCollocationMatrix(dt, eps, M, Q, problemName, problemType):
    r"""
    Returns matrix of the collocation problem.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    eps : float
        Perturbation parameter :math:`\varepsilon` of singular perturbed problems.
    M : int
        Number of quadrature nodes.
    Q : np.2darray
        Spectral quadrature matrix.
    problemName : str
        Name of the problem
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.

    Returns
    -------
    C : np.2darray
        Matrix of collocation problem.
    """

    if problemName == 'LINEAR-TEST':
        lamb_diff = -2.0
        lamb_alg = 1.0

        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        Ieps = np.identity(2)
        Ieps[-1, -1] = 0

        if problemType == 'SPP':
            A[1, :] *= 1 / eps

            C = np.kron(np.identity(M), np.identity(2)) - dt * np.kron(Q, A)

        elif problemType == 'embeddedDAE':
            C = np.kron(np.identity(M), Ieps) - dt * np.kron(Q, A)

        elif problemType == 'constrainedDAE':
            A_d = np.array([[lamb_diff, lamb_alg], [0, 0]])
            A_a = np.array([[0, 0], [lamb_diff, -lamb_alg]])

            C = np.kron(np.identity(M), Ieps) - dt * np.kron(Q, A_d) + np.kron(np.identity(M), A_a)

        else:
            raise NotImplementedError(
                f"Matrix of collocation problem for {problemType} of {problemName} problem is not implemented!"
            )

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {problemName}!")

    return C


def getDescription(dt, QI, **kwargs):
    """
    Get a first description and initialise dictionaries for other setting such as problem
    and sweeper.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.

    Returns
    -------
    description : dict
        Description dictionary.
    """

    restol = kwargs.get("restol", -1)
    restol = -1 if kwargs.get("use_adaptivity", False) else restol
    level_params = {
        "dt": dt,
        "restol": restol,
    }

    convergence_controllers = setupConvergenceControllers(dt, QI, **kwargs)

    description = {
        "problem_class": None,
        "problem_params": {},
        "sweeper_class": None,
        "sweeper_params": {},
        "level_params": level_params,
        "step_params": {},
        "convergence_controllers": convergence_controllers,
    }

    return description


def getEndTime(problemName):
    """
    Returns end time for specific problem.

    Parameters
    ----------
    problemName : str
        Name of the problem.

    Returns
    -------
    float :
        End time for simulation.
    """

    Tprob = {
        "ANDREWS-SQUEEZER": 0.03,
        "LINEAR-TEST": 1.0,
        "MICHAELIS-MENTEN": 0.02,  # 1.2
        "VAN-DER-POL": 1.5,
    }
    return Tprob[problemName]


def getErrors(problemType, solutionStats, extract_for='step'):
    if extract_for == 'step':
        err_type_y = f'e_global_differential_post_{extract_for}' if not problemType == 'SPP' else f'e_global_post_{extract_for}'
        errDiffValues = [me[1] for me in get_sorted(solutionStats, type=err_type_y, sortby='time')]
        errAlgValues = [me[1] for me in get_sorted(solutionStats, type=f'e_global_algebraic_post_{extract_for}', sortby='time')]
    elif extract_for == 'iter':
        errDiffValues = [me[1] for me in get_sorted(solutionStats, type=f'e_global_post_{extract_for}', sortby='iter')]
        errAlgValues = [me[1] for me in get_sorted(solutionStats, type=f'e_global_algebraic_post_{extract_for}', sortby='iter')]
    else:
        raise NotImplementedError(f"No unpacking implemented for {extract_for}!")

    return errDiffValues, errAlgValues


def getIterationMatrix(dt, eps, M, Q, QI, problemName, problemType):
    r"""
    Returns iteration matrix for SDC written as Richardson iteration.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    eps : float
        Perturbation parameter :math:`\varepsilon` of singular perturbed problems.
    M : int
        Number of quadrature nodes.
    Q : np.2darray
        Spectral quadrature matrix.
    QI : np.2darray
        Lower triangular matrix (for the preconditioner).
    problemName : str
        Name of the problem
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.

    Returns
    -------
    K : np.2darray
        Iteration matrix of linear solver.
    """

    if problemName == 'LINEAR-TEST':
        lamb_diff = -2.0
        lamb_alg = 1.0

        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        Ieps = np.identity(2)
        Ieps[-1, -1] = 0

        if problemType == 'SPP':
            A[1, :] *= 1 / eps

            inv = np.linalg.inv(np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QI, A))
            K = np.matmul(inv, dt * np.kron(Q - QI, A))

        elif problemType == 'embeddedDAE':
            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QI, A))
            K = np.matmul(inv, dt * np.kron(Q - QI, A))

        elif problemType == 'constrainedDAE':
            A_d = np.array([[lamb_diff, lamb_alg], [0, 0]])
            A_a = np.array([[0, 0], [lamb_diff, -lamb_alg]])

            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QI, A_d) + np.kron(np.identity(M), A_a))
            K = np.matmul(inv, dt * np.kron(Q - QI, A_d))

        elif problemType == 'difference':
            ## SDC-E expressed in terms of SDC-C
            A_d = np.array([[lamb_diff, lamb_alg], [0, 0]])
            A_a = np.array([[0, 0], [lamb_diff, -lamb_alg]])

            diffL = np.kron(dt * QI - np.identity(M), A_a)
            diffR = dt * np.kron(Q - QI, A_a)
            # inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QI, A_d) - np.kron(np.identity(M), A_a))# - diffL)
            # K = np.matmul(inv, dt * np.kron(Q - QI, A_d))# + diffR)

            ## SDC-C expressed in terms of SDC-E
            inv = np.linalg.inv(np.kron(np.identity(M), Ieps) - dt * np.kron(QI, A))# + diffL)
            K = np.matmul(inv, dt * np.kron(Q - QI, A) - diffR)

        else:
            raise NotImplementedError(
                f"No iteration matrix for {problemType} of {problemName} problem is not implemented!"
            )

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {problemName}!")

    return K


def getJacobianMatrix(dt, eps, M, Q, QI, problemName, problemType):
    r"""
    Returns Jacobian matrix for SDC which is used to solve the linear system via Newton.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    eps : float
        Perturbation parameter :math:`\varepsilon` of singular perturbed problems.
    M : int
        Number of quadrature nodes.
    Q : np.2darray
        Spectral quadrature matrix.
    QI : np.2darray
        Lower triangular matrix (for the preconditioner).
    problemName : str
        Name of the problem
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.

    Returns
    -------
    J : np.2darray
        Iteration matrix of linear solver.
    """

    if problemName == 'LINEAR-TEST':
        lamb_diff = -2.0
        lamb_alg = 1.0

        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        Ieps = np.identity(2)
        Ieps[-1, -1] = 0

        if problemType == 'SPP':
            A[1, :] *= 1 / eps

            J = np.kron(np.identity(M), np.identity(2)) - dt * np.kron(QI, A)

        elif problemType == 'embeddedDAE':
            J = np.kron(np.identity(M), Ieps) - dt * np.kron(QI, A)

        elif problemType == 'constrainedDAE':
            A_d = np.array([[lamb_diff, lamb_alg], [0, 0]])
            A_a = np.array([[0, 0], [lamb_diff, -lamb_alg]])

            J = np.kron(np.identity(M), Ieps) - dt * np.kron(QI, A_d) + np.kron(np.identity(M), A_a)
        elif problemType == 'difference':
            A_d = np.array([[lamb_diff, lamb_alg], [0, 0]])
            A_a = np.array([[0, 0], [lamb_diff, -lamb_alg]])

            diffL = np.kron(dt * QI + np.identity(M), A_a)
            # diffR = dt * np.kron(Q - QI, A_a)
            J = np.kron(np.identity(M), Ieps) - dt * np.kron(QI, A_d) + np.kron(np.identity(M), A_a) - diffL

        else:
            raise NotImplementedError(
                f"No iteration matrix for {problemType} of {problemName} problem is not implemented!"
            )

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {problemName}!")

    return J


def getRHS(dt, eps, M, Q, QI, problemName, problemType):
    r"""
    Returns iteration matrix for SDC written as Richardson iteration.

    Parameters
    ----------
    dt : float
        Time step size for the simulation.
    eps : float
        Perturbation parameter :math:`\varepsilon` of singular perturbed problems.
    M : int
        Number of quadrature nodes.
    Q : np.2darray
        Spectral quadrature matrix.
    QI : np.2darray
        Lower triangular matrix (for the preconditioner).
    problemName : str
        Name of the problem
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.

    Returns
    -------
    K : np.2darray
        Iteration matrix of linear solver.
    """

    if problemName == 'LINEAR-TEST':
        lamb_diff = -2.0
        lamb_alg = 1.0

        A = np.array([[lamb_diff, lamb_alg], [lamb_diff, -lamb_alg]])

        Ieps = np.identity(2)
        Ieps[-1, -1] = 0

        if problemType == 'SPP':
            A[1, :] *= 1 / eps

            u0 = np.array([np.exp(2 * lamb_diff * 0.0), (lamb_diff / lamb_alg) * np.exp(2 * lamb_diff * 0.0)])

            RHS = np.kron(np.identity(M), np.diag(u0)) + dt * np.kron(Q - QI, A)

        elif problemType == 'embeddedDAE':
            u0 = np.array([np.exp(2 * lamb_diff * 0.0), 0.0])

            RHS = np.kron(np.identity(M), np.diag(u0)) + dt * np.kron(Q - QI, A)

        elif problemType == 'constrainedDAE':
            A_d = np.array([[lamb_diff, lamb_alg], [0, 0]])
            A_a = np.array([[0, 0], [lamb_diff, -lamb_alg]])

            u0 = np.array([np.exp(2 * lamb_diff * 0.0), 0.0])

            RHS = np.kron(np.identity(M), np.diag(u0)) + dt * np.kron(Q - QI, A_d)

        else:
            raise NotImplementedError(
                f"No iteration matrix for {problemType} of {problemName} problem is not implemented!"
            )

    else:
        raise NotImplementedError(f"No iteration matrix implemented for {problemName}!")

    return RHS


def getMaxVal(oldMax, newMax):
    if oldMax > newMax:
        newMax = oldMax
    return newMax


def getMinVal(oldMin, newMin):
    if oldMin < newMin:
        newMin = oldMin
    return newMin


# Function to get the appropriate class and description
def getProblemDetails(problemName, problemType):
    r"""
    Parameters
    ----------
    problemName : str
        Name of the problem
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.

    Returns
    -------
    problem_class, description_info
        Problem class module and corresponding description info.

    Raises
    ------
    NotImplementedError
        Possibly for some combination of ``problemName`` and ``problemType`` there is no setup implemented.
    """

    problem_info = problemMapping[problemName].get(problemType)
    
    # Check if the problem is marked as not implemented
    if problem_info.get("NotImplemented", False):
        raise NotImplementedError(f"The problem '{problemName}' with type '{problemType}' is not implemented.")
    
    # Dynamically import the module and class
    module = importlib.import_module(problem_info["module"])
    problem_class = getattr(module, problem_info["class"])
    
    # Retrieve the description dictionary
    description_info = problemMapping[problemName]["description"]
    
    return problem_class, description_info


def getWork(solutionStats, solver_type=None):
    newton = [me[1] for me in get_sorted(solutionStats, type=f"work_{solver_type}", sortby='time')] if solver_type == "newton" else [0]
    rhs = [me[1] for me in get_sorted(solutionStats, type=f"work_rhs", sortby='time')]
    nIter = [me[1] for me in get_sorted(solutionStats, type='niter', sortby='time')]

    return sum(newton) + sum(rhs), nIter


def setupConvergenceControllers(dt, QI, **kwargs):
    """
    Returns the dictionary containing the convergence controllers.

    Returns
    -------
    convergence_controllers : dict
        Contains convergence controllers for simulation.
    """

    use_adaptivity = kwargs.get("use_adaptivity", False)
    convergence_controllers = {}
    if use_adaptivity:
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        from pySDC.implementations.convergence_controller_classes.basic_restarting import BasicRestartingNonMPI

        adaptivity_params = {
            "e_tol": kwargs.get("e_tol_adaptivity", 1e-7),
            "dt_min": 1e-9,
            "dt_max": 1e-2,
            "avoid_restarts": False,
        }
        convergence_controllers.update({Adaptivity: adaptivity_params})

        restarting_params = {
            # "max_restarts": 5,
            "crash_after_max_restarts": False,
        }
        convergence_controllers.update({BasicRestartingNonMPI: restarting_params})

    # if QI == "MIN-SR-FLEX":
    #     update_qdelta_params = {
    #         "change_using_sweeps": kwargs.get("change_using_sweeps", False),
    #         "change_using_iters": kwargs.get("change_using_iters", True),
    #     }
    #     convergence_controllers.update({UpdateQDelta: update_qdelta_params})

    return convergence_controllers


def setupProblem(problemName, description, nNodes, problemType, **kwargs):
    r"""
    Here, the setup for the problem is set and passed to the current description dictionary.

    Parameters
    ----------
    problemName : str
        Name of the problem.
    description : dict
        Current description.
    nNodes : int
        Number of quadrature nodes.
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.

    Returns
    -------
    description : dict
        Updated description dictionary.
    """

    maxiter_adaptivity = 2 * nNodes - 1

    try:
        problem, descriptionInfo = getProblemDetails(problemName, problemType)

        e_tol = descriptionInfo["e_tol"]
        maxiter = descriptionInfo["maxiter"]
        newton_tol = descriptionInfo["newton_tol"]
        newton_maxiter = descriptionInfo["newton_maxiter"]

        description["level_params"]["e_tol"] = kwargs.get("e_tol", e_tol)

        description["step_params"] = {
            "maxiter": maxiter_adaptivity if kwargs.get("use_adaptivity", False) else kwargs.get("maxiter", maxiter)
        }

        description["problem_params"] = {"newton_tol": kwargs.get("newton_tol", newton_tol), "newton_maxiter": newton_maxiter}

        # Add problem-specific params "manually" (TODO: more efficient way to do that?)
        if problemName == "ANDREWS-SQUEEZER":
            index = descriptionInfo["index"]
            description["problem_params"].update({"index": kwargs.get("index", index)})

        elif problemName == "LINEAR-TEST":
            solver_type = descriptionInfo["solver_type"]
            description["problem_params"].update({"solver_type": kwargs.get("solver_type", solver_type)})

            if problemType in ["SPP-IMEX", "SPP-yp"]:
                description["problem_params"].pop("newton_maxiter", None)

        elif problemName == "MICHAELIS-MENTEN":
            if problemType in ["SPP-yp"]:
                description["problem_params"].pop("newton_maxiter", None)

        elif problemName == "PROTHERO-ROBINSON":
            if problemType in ["fullyImplicitDAE", "semiImplicitDAE"]:
                description["problem_params"].pop("newton_maxiter", None)

        eps = kwargs.get("eps")
        if eps != 0.0:
            description["problem_params"]["eps"] = eps

        description["problem_class"] = problem

    except NotImplementedError as e:
        print(e)

    return description


def setupSweeper(description, nNodes, QI, problemType, useMPI=False, **kwargs):
    r"""
    Updates ``description`` with parameters for the sweeper. Type of sweeper depends on type of
    the problem ``problemType``. Further, either a MPI or a nonMPI version is available.

    Parameters
    ----------
    description : dict
        Current description for simulation.
    nNodes : int
        Number of quadrature nodes.
    QI : str
        Type of :math:`Q_\Delta`-matrix.
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.
    useMPI : bool, optional
        Choose either MPI or nonMPI sweeper. Default is ``False``.
    comm : MPI.COMM_WORLD
        Communicator for parallel computations.

    Returns
    -------
    desciption : dict
        Updated description dictionary.
    """

    skip_residual_computation_default = ("IT_DOWN", "IT_UP", "IT_COARSE", "IT_FINE", "IT_CHECK")

    # Define categories
    is_sdc_method = QI in SDC_METHODS
    is_rk_method = QI in RK_METHODS
    is_collocation_method = QI in COLLOCATION_METHODS

    # Determine the correct sweeper
    if is_sdc_method:
        # Select the correct SDC sweeper mapping
        sweeper_mapping = SDC_SWEEPER_MAPPING_MPI if useMPI else SDC_SWEEPER_MAPPING

        if problemType not in sweeper_mapping:
            raise NotImplementedError(f"No SDC sweeper implemented for problem type: {problemType}")

        # Fetch module and class name
        module_name = sweeper_mapping[problemType]["module"]
        class_name = sweeper_mapping[problemType]["class"]

        # Dynamically import the module and class
        sweeper_module = importlib.import_module(module_name)
        sweeper = getattr(sweeper_module, class_name)

        # initialize sweeper parameters
        description["sweeper_params"].update({
            "quad_type": kwargs.get("quadType", "RADAU-RIGHT"),
            "num_nodes": nNodes,
            "QI": QI,
            "QE": "EE",
            "initial_guess": "spread",
            "skip_residual_computation": kwargs.get("skip_residual_computation", skip_residual_computation_default),
        })

        if problemType == "SPP-IMEX":
            description["sweeper_params"].update({"QE": "EE"})

        description["level_params"].update({"nsweeps": kwargs.get("nsweeps", 1)})

        # MPI-related checks
        if useMPI and "comm" in kwargs:
            comm = kwargs["comm"]
            description["sweeper_params"]["comm"] = comm
            assert nNodes == comm.Get_size(), f"Mismatch: {nNodes} nodes, but {comm.Get_size()} MPI processes."

    elif is_rk_method or is_collocation_method:
        if useMPI:
            raise NotImplementedError("Parallel Runge-Kutta and Collocation methods are not implemented with MPI.")
        
        problem_key = (
            "constrained" if "constrained" in problemType else
            "fullyImplicit" if "fullyImplicit" in problemType else
            "semiImplicit" if "semiImplicit" in problemType else None
        )

        if problem_key is None:
            raise NotImplementedError(f"No RK/Collocation module defined for problem type '{problemType}'.")

        # Choose the correct module based on problem type
        sweeper_mapping = RK_SWEEPER_MAPPING if is_rk_method else COLLOCATION_SWEEPER_MAPPING

        if problem_key not in sweeper_mapping or QI not in sweeper_mapping[problem_key]["class"]:
            raise NotImplementedError(f"Method '{QI}' not found for problem type '{problemType}'.")

        # Fetch module and class name
        module_name = sweeper_mapping[problem_key]["module"]
        class_name = sweeper_mapping[problem_key]["class"][QI]

        # Dynamically import the module and class
        sweeper_module = importlib.import_module(module_name)
        sweeper = getattr(sweeper_module, class_name)

        # RK sweeper settings
        description["level_params"].update({"restol": -1, "e_tol": -1, "nsweeps": 1})
        description["step_params"] = {"maxiter": 1}
        description["sweeper_params"] = {"skip_residual_computation": skip_residual_computation_default}

    # Assign sweeper class to description
    description["sweeper_class"] = sweeper

    #######################################################
    # if QI in SDC_METHODS:
        # if useMPI:
        #     if problemType == 'SPP':
        #         from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper
        #     elif problemType == "SPP-yp":
        #         from pySDC.projects.DAE.sweepers.fullyImplicitDAEMPI import FullyImplicitDAEMPI as sweeper
        #     elif problemType == 'embeddedDAE':
        #         from pySDC.playgrounds.DAE.genericImplicitDAEMPI import genericImplicitEmbeddedMPI as sweeper
        #     elif problemType == 'constrainedDAE':
        #         from pySDC.playgrounds.DAE.genericImplicitDAEMPI import genericImplicitConstrainedMPI as sweeper
        #     elif problemType == 'fullyImplicitDAE':
        #         from pySDC.projects.DAE.sweepers.fullyImplicitDAEMPI import FullyImplicitDAEMPI as sweeper
        #     elif problemType == 'semiImplicitDAE':
        #         from pySDC.projects.DAE.sweepers.semiImplicitDAEMPI import SemiImplicitDAEMPI as sweeper
        #     else:
        #         raise NotImplementedError(f"No MPI sweeper for {problemType} implemented!")

        #     comm = kwargs.get('comm', None)
        #     if comm is not None:
        #         description['sweeper_params'].update({'comm': comm})
        #         assert (
        #             nNodes == comm.Get_size()
        #         ), f"Number of nodes does not match with number of processes! Expected {nNodes}, got {comm.Get_size()}!"
        # else:
        #     if problemType == 'SPP':
        #         logResidualComp = kwargs.get('logResidualComp', False)
        #         if logResidualComp:
        #             from pySDC.playgrounds.DAE.genericImplicitDAE import genericImplicitOriginal as sweeper
        #         else:
        #             from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper
        #     elif problemType == "gmresSPP":
        #         from pySDC.implementations.sweeper_classes.gmres_sdc import GMRESSDC as sweeper
        #     elif problemType == "SPP-yp":
        #         from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE as sweeper
        #     elif problemType == 'embeddedDAE':
        #         from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded as sweeper
        #     elif problemType == 'constrainedDAE':
        #         from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained as sweeper
        #     elif problemType == 'fullyImplicitDAE':
        #         from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE as sweeper
        #     elif problemType == 'semiImplicitDAE':
        #         from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE as sweeper
        #     else:
        #         raise NotImplementedError(f"{problemType} is not implemented!")

    #     # initialize sweeper parameters
    #     description['sweeper_params'].update({
    #         'quad_type': kwargs.get('quadType', 'RADAU-RIGHT'),
    #         'num_nodes': nNodes,
    #         'QI': QI,
    #         'initial_guess': 'spread',
    #         'skip_residual_computation': kwargs.get('skip_residual_computation', skip_residual_computation_default),
    #     })

    # elif QI in RK_METHODS.keys() or QI in RK_METHODS_DAE.keys():  # Expect scheme to be a Runge-Kutta method
    #     if useMPI:
    #         raise NotImplementedError(f"No parallel Runge-Kutta schemes implemented!")
    #     if problemType == 'SPP':
    #         sweeper = RK_METHODS[QI]
    #     elif problemType in ['SPP-yp', 'fullyImplicitDAE', 'semiImplicitDAE']:
    #         sweeper = RK_METHODS_DAE[QI]
    #     elif problemType == 'constrainedDAE':
    #         sweeper = RK_METHODS_DAE_CONSTRAINED[QI]
    #     else:
    #         raise NotImplementedError(f"No Runge Kutta method for {problemType} is implemented!")

    #     description['level_params']['restol'] = -1
    #     description['level_params']['e_tol'] = -1
    #     description['level_params']['nsweeps'] = 1

    #     description['step_params'] = {'maxiter': 1}
    #     description['sweeper_params'] = {
    #         'skip_residual_computation': skip_residual_computation_default,
    #     }

    # description['sweeper_class'] = sweeper

    return description


def computeSolution(problemName, t0, dt, Tend, nNodes, QI, problemType, useMPI=False, hookClass=[], **kwargs):
    r"""
    Computes the solution for specific time step size. Simulation interval is specified for each problem class.
    Number of nodes ``nNodes`` and ``QI`` can be set and to get statistics, a list of classes logging data
    ``hookClass`` can be passed. The type of the problem (either a singular perturbation problem or a
    differential-algebraic equation) needs to be specified as well.

    Parameters
    ----------
    problemName : str
        Name of the problem.
    dt : float
        Time step size used for simulation.
    nNodes : int
        Number of collocation nodes. Note that the number of collocation nodes needs to be set whereas
        the quadrature type is not necessary but can be set using ``quadType`` by passing as keyword argument.
        Default is ``'RADAU-RIGHT'``.
    QI : str
        Indicates type of :math:`Q_\Delta`-matrix.
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.
    hookClass : list of pySDC.core.hooks, optional
        Different classes used to log data during simulation.

    Returns
    -------
    stats : dict
        Dictionary containing all statistics of simulation.

    Note
    ----
    The following can be set as keyword arguments (``**kwargs``):

    - ``restol``: Residual tolerance used for stopping criterion. Default is ``-1``.
    - ``e_tol``: Tolerance used for the increment as stopping criterion. For each defined problem there is a
      default value set resulting from studies.
    - ``maxiter``: Maximum number of iterations with default value for each problem set.
    - ``eps``: Perturbation parameter for singular perturbed problems. Only when :math:`\varepsilon > 0` then
      it is passed to the problem parameters, since the DAE cases does not require it.
    - ``quadType``: Type of quadrature. Default is ``'RADAU-RIGHT'``.
    - ``logResidualComp``: Flag to decide if residual components would like to log. This would change the sweeper since
      ``generic_implicit`` is not able to log the residual components.
    - ``skip_residual_computation``: Note that by default residual computation will be skipped. Otherwise, when using the
      residual this keyword argument should be set to avoid wrong residual results.
    """

    # setup description
    description = getDescription(dt, QI, **kwargs)

    description = setupProblem(problemName, description, nNodes, problemType, **kwargs)

    description = setupSweeper(description, nNodes, QI, problemType, useMPI=useMPI, **kwargs)

    computeOneStep = kwargs.get("computeOneStep", False)
    Tend = dt if computeOneStep else Tend

    # instantiate controller
    logger_level = kwargs.get("logger_level", 30)
    controller_params = {
        "logger_level": logger_level,
        "hook_class": hookClass,
        "mssdc_jac": False if kwargs.get("use_adaptivity", False) else True,
    }
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    _, solutionStats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return solutionStats


def checkLogDistribution(lst):
    # Check if the list has at least two elements
    if len(lst) < 2:
        return None

    # Calculate the ratio between the first two elements
    base = lst[1] / lst[0]

    # Check if the ratio is consistent for the entire list
    for i in range(1, len(lst)):
        ratio = lst[i] / lst[i - 1]
        if not np.isclose(ratio, base):
            return None

    # If all ratios are consistent, return the base of the logarithmic distribution
    return base


def computeNormalityDeviation(A):
    """
    Calculate the Frobenius norm of the commutator of A.
    
    The commutator of A is A*A - AA*.
    The Frobenius norm of the commutator measures how far A is from being normal.
    
    Parameters:
    A (numpy.ndarray): A square matrix
    
    Returns:
    float: The Frobenius norm of the commutator A*A - AA*
    """

    A_star = np.conjugate(A.T)  # Compute the conjugate transpose of A
    commutator = A @ A_star - A_star @ A
    deviation = np.linalg.norm(commutator, 'fro')  # Frobenius norm
    return deviation


def createProblemParamsDict(eps, **kwargs):
    problem_params = {}
    if eps != 0.0:
        problem_params['eps'] = eps
    problem_params.update(kwargs)
    return problem_params


def roundUpToNextX(number, x):
    return ((number + (x - 1)) // x) * x


def roundUpToNextBase10(number):
    if number <= 0:
        raise ValueError("Number must be positive")

    # Find the next power of 10 greater than the number using NumPy
    exponent = np.ceil(np.log10(number))
    nextPowerOf10 = 10**exponent

    # roundedValue = roundUpToNextX(number, nextPowerOf10)
    roundedValue = nextPowerOf10
    return roundedValue


def roundDownToPreviousBase10(number):
    if number <= 0:
        raise ValueError("Number must be positive")
    
    # Find the exponent for the previous power of 10 using NumPy
    exponent = np.floor(np.log10(number))  # Use `np.floor` to round down the exponent
    previousPowerOf10 = 10 ** exponent  # Compute the power of 10
    
    return previousPowerOf10
