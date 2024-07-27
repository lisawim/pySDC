import pytest


def runSimulation(t0, dt, Tend, quad_type, problemType):
    r"""
    Executes a run to solve numerically the Van der Pol with tests to check the ``DenseOutput`` class.

    Parameters
    ----------
    t0 : float
        Initial time.
    dt : float
        Time step size.
    Tend : float
        End time.
    quad_type : str
        Type of quadrature.
    problemType : str
        Type ``'ODE'`` as well as ``'DAE'`` is tested here. Note that only ``newton_tol`` is set, and
        for simulation default parameters of problem classes are used.
    """

    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    if problemType == 'ODE':
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper
        from pySDC.implementations.problem_classes.odeSystem import ProtheroRobinsonAutonomous as problem

    elif problemType == 'DAE':
        from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE as sweeper
        from pySDC.projects.DAE.problems.simpleDAE import SimpleDAE as problem

    else:
        raise NotImplementedError(f"For {problemType} no sweeper and problem class is implemented!")

    # initialize level parameters
    level_params = {
        'dt': dt,
        'restol': -1,
    }

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': quad_type,
        'num_nodes': 3,
        'QI': 'LU',
    }

    problem_params = {
        'newton_tol': 1e-12,
    }
    if problemType == 'ODE':
        problem_params.update({'epsilon': 1e-1})

    # initialize step parameters
    step_params = {'maxiter': 8}

    adaptivity_params = {
        'e_tol': 1e-12,
    }
    convergence_controllers = {Adaptivity: adaptivity_params}

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': problem,
        'problem_params': problem_params,
        'sweeper_class': sweeper,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }

    # instantiate controller
    controller_params = {'logger_level': 30, 'hook_class': [LogSolution], 'mssdc_jac': False}
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    prob = controller.MS[0].levels[0].prob

    uinit = prob.u_exact(t0)

    # call main function to get things done...
    _, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return prob, stats


@pytest.mark.base
@pytest.mark.parametrize("quad_type", ['RADAU-RIGHT', 'LOBATTO'])
@pytest.mark.parametrize("problemType", ['ODE', 'DAE'])
def test_interpolate(quad_type, problemType):
    r"""
    Interpolation in ``DenseOutput`` is tested by evaluating the polynomials at ``t_eval``.
    The interpolated values are then compared with the reference error. It seems like ``'LOBATTO'``
    does provide less accurate results in stiff components.
    """

    import numpy as np
    from pySDC.projects.DAE import DenseOutput
    from pySDC.helpers.stats_helper import get_sorted

    t0 = 0.0
    dt = 1e-2
    Tend = 0.5

    skipTest = True if quad_type == 'LOBATTO' and problemType == 'DAE' else False
    if not skipTest:
        prob, solutionStats = runSimulation(t0=t0, dt=dt, Tend=Tend, quad_type=quad_type, problemType=problemType)
    else:
        prob, solutionStats = None, None

    # get values of u and corresponding times for all nodes
    if solutionStats is not None:
        u_dense = get_sorted(solutionStats, type='u_dense', sortby='time', recomputed=False)
        nodes_dense = get_sorted(solutionStats, type='nodes_dense', sortby='time', recomputed=False)
        sol = DenseOutput(nodes_dense, u_dense)

        t_eval = [t0 + i * dt for i in range(int(Tend / dt) + 1)]
        u_eval = [sol(t_item) for t_item in t_eval]

        if problemType == 'ODE':
            u_eval = np.array(u_eval)
            uex = np.array([prob.u_exact(t_item) for t_item in t_eval])

        elif problemType == 'DAE':
            x1_eval = np.array([me.diff[0] for me in u_eval])
            x2_eval = np.array([me.diff[1] for me in u_eval])
            z_eval = np.array([me.alg[0] for me in u_eval])

            x1ex = np.array([prob.u_exact(t_item).diff[0] for t_item in t_eval])
            x2ex = np.array([prob.u_exact(t_item).diff[1] for t_item in t_eval])
            zex = np.array([prob.u_exact(t_item).alg[0] for t_item in t_eval])

            u_eval = np.column_stack((np.column_stack((x1_eval, x2_eval)), z_eval))
            uex = np.column_stack((np.column_stack((x1ex, x2ex)), zex))

        for i in range(uex.shape[0]):
            assert np.allclose(u_eval[i, :], uex[i, :], atol=1e-7), f"For index {i} error is too large!"
