import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


# < ---- BEGIN OLD STUFF ---- >
def getSetup():
    setup = {
        'LinearTestSPP': {
            'nSweeps': 24,
            'e_tol': 1e-11,
            'nSteps': np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000]),
            'Tend': 1.0,
            'solve_system_setup': {'lintol': 1e-12},
            'work_type': 'work_gmres',
        },
        'VanDerPol': {
            'nSweeps': 20,
            'e_tol': 5e-13,
            'nSteps': np.array([20, 50, 200, 500, 2000, 5000, 20000, 50000]),
            'Tend': 3.0,
            'solve_system_setup': {'newton_tol': 1e-12},
            'work_type': 'work_newton',
        }
    }
    return setup

# < ---- END OLD STUFF ---- >


# < ---- BEGIN NEW STUFF ---- >
QI_SERIAL = ['IE', 'LU', 'MIN-SR-S', 'MIN-SR-NS', 'MIN-SR-FLEX']

QI_PARALLEL = ['MIN-SR-S', 'MIN-SR-NS', 'MIN-SR-FLEX']


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
        'VAN-DER-POL': 1.5,
        'LINEAR-TEST': 1.0,
    }
    return Tprob[problemName]

def setupConvergenceControllers(**kwargs):
    """
    Returns the dictionary containing the convergence controllers.

    Returns
    -------
    convergence_controllers : dict
        Contains convergence controllers for simulation.
    """
    use_adaptivity = kwargs.get('use_adaptivity', False)

    convergence_controllers = {}
    if use_adaptivity:
        from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
        adaptivity_params = {
            'e_tol': kwargs.get('e_tol_adaptivity', 1e-7),
        }
        convergence_controllers.update({Adaptivity: adaptivity_params})
    return convergence_controllers

def getDescription(dt, **kwargs):
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

    level_params = {
        'dt': dt,
        'restol': kwargs.get('restol', -1),
    }

    convergence_controllers = setupConvergenceControllers(**kwargs)

    description = {
        'problem_class': None,
        'problem_params': {},
        'sweeper_class': None,
        'sweeper_params': {},
        'level_params': level_params,
        'step_params': {},
        'convergence_controllers': convergence_controllers,
    }

    return description


def setupProblem(problemName, description, problemType, **kwargs):
    r"""
    Here, the setup for the problem is set and passed to the current description dictionary.

    Parameters
    ----------
    problemName : str
        Name of the problem.
    description : dict
        Current description.
    problemType : str
        Type of problem. Can be ``'SPP'``, ``'embeddedDAE'`` or ``'constrainedDAE'``.

    Returns
    -------
    description : dict
        Updated description dictionary.
    """

    if problemName == 'VAN-DER-POL':
        if problemType == 'SPP':
            from pySDC.implementations.problem_classes.singularPerturbed import VanDerPol as problem
        elif problemType == 'embeddedDAE':
            from pySDC.projects.DAE.problems.vanDerPolDAE import VanDerPolEmbedded as problem
        elif problemType == 'constrainedDAE':
            from pySDC.projects.DAE.problems.vanDerPolDAE import VanDerPolConstrained as problem
        else:
            raise NotImplementedError(f"{problemType} is not implemented!")

        description['level_params']['e_tol'] = kwargs.get('e_tol', 5e-13)

        description['step_params'] = {'maxiter': kwargs.get('maxiter', 20)}

        description['problem_params'] = {'newton_tol': 1e-12, 'newton_maxiter': 100}

    elif problemName == 'LINEAR-TEST':
        if problemType == 'SPP':
            from pySDC.implementations.problem_classes.singularPerturbed import LinearTestSPP as problem
        elif problemType == 'embeddedDAE':
            from pySDC.projects.DAE.problems.LinearTestDAE import LinearTestDAEEmbedded as problem
        elif problemType == 'constrainedDAE':
            from pySDC.projects.DAE.problems.LinearTestDAE import LinearTestDAEConstrained as problem
        else:
            raise NotImplementedError(f"{problemType} is not implemented!")

        description['level_params']['e_tol'] = kwargs.get('e_tol', 1e-11)

        description['step_params'] = {'maxiter': kwargs.get('maxiter', 24)}

        description['problem_params'] = {'lintol': 1e-12}

    else:
        raise NotImplementedError(f"No setup for {problemName} implemented!")

    # TODO: How to reasonably iterate along the DAE and the ODE case when having eps=0 and having eps!=0?
    eps = kwargs.get('eps')
    if eps != 0.0:
        description['problem_params']['eps'] = eps

    description['problem_class'] = problem

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

    # initialize sweeper parameters
    skip_residual_computation_default = ('IT_DOWN', 'IT_UP', 'IT_COARSE', 'IT_FINE', 'IT_CHECK')
    description['sweeper_params'] = {
        'quad_type': kwargs.get('quadType', 'RADAU-RIGHT'),
        'num_nodes': nNodes,
        'QI': QI,
        'initial_guess': 'spread',
        'skip_residual_computation': kwargs.get('skip_residual_computation', skip_residual_computation_default),
    }

    if useMPI:
        if problemType == 'SPP':
            from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper
        else:
            raise NotImplementedError(f"No sweeper for {problemType} implemented!")

        comm = kwargs.get('comm', None)
        if comm is not None:
            description['sweeper_params'].update({'comm': comm})
            assert (
                nNodes == comm.Get_size()
            ), f"Number of nodes does not match with number of processes! Expected {nNodes}, got {comm.Get_size()}!"
    else:
        if problemType == 'SPP':
            logResidualComp = kwargs.get('logResidualComp', False)
            if logResidualComp:
                from pySDC.playgrounds.DAE.genericImplicitDAE import genericImplicitOriginal as sweeper
            else:
                from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper
        elif problemType == 'embeddedDAE':
            from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitEmbedded as sweeper
        elif problemType == 'constrainedDAE':
            from pySDC.projects.DAE.sweepers.genericImplicitDAE import genericImplicitConstrained as sweeper
        else:
            raise NotImplementedError(f"{problemType} is not implemented!")
    
    description['sweeper_class'] = sweeper

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
    description = getDescription(dt, **kwargs)

    description = setupProblem(problemName, description, problemType, **kwargs)

    description = setupSweeper(description, nNodes, QI, problemType, useMPI=useMPI, **kwargs)

    computeOneStep = kwargs.get('computeOneStep', False)
    Tend = dt if computeOneStep else Tend

    # instantiate controller
    controller_params = {'logger_level': 30, 'hook_class': hookClass, 'mssdc_jac': False if kwargs.get('use_adaptivity', False) else True}
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    _, solutionStats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    return solutionStats


def createProblemParamsDict(eps, **kwargs):
    problem_params = {}
    if eps != 0.0:
        problem_params['eps'] = eps
    problem_params.update(kwargs)
    return problem_params


def getColors(eps):
    colors = {
        1e-1: 'mistyrose',
        1e-2: 'lightsalmon',
        1e-3: 'lightcoral',
        1e-4: 'indianred',
        1e-5: 'firebrick',
        1e-6: 'brown',
        1e-7: 'maroon',
        1e-8: 'lightgray',
        1e-9: 'darkgray',
        1e-10: 'gray',
        1e-11: 'dimgray',
    }
    return colors[eps]


# < ---- END NEW STUFF ---- >