import pytest


def runSimulation(quad_type, problemType):
    r"""
    Executes a run to solve numerically the Van der Pol with tests to check the ``DenseOutput`` class.

    Parameters
    ----------
    quad_type : str
        Type of quadrature.
    problemType : str
        Type ``'ODE'`` as well as ``'DAE'`` is tested here. Note that only ``newton_tol`` is set, and
        for simulation default parameters of problem classes are used.
    """

    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    skipTest = True if quad_type == 'LOBATTO' and problemType == 'ODE' else False

    if not skipTest:
        if problemType == 'ODE':
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper
            from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol as problem

        elif problemType == 'DAE':
            from pySDC.projects.DAE.sweepers.SemiImplicitDAE import SemiImplicitDAE as sweeper
            from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE as problem

        else:
            raise NotImplementedError(f"For {problemType} no sweeper and problem class is implemented!")

        # initialize level parameters
        level_params = {
            'dt': 1e-2,
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

        # initialize step parameters
        step_params = {'maxiter': 6}

        adaptivity_params = {
            'e_tol': '1e-7',
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

        t0 = 0.0
        Tend = 1.0

        uinit = prob.u_exact(t0)

        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
        return stats
    else:
        return None


@pytest.mark.base
@pytest.mark.parametrize("quad_type", ['RADAU-RIGHT', 'LOBATTO'])
@pytest.mark.parametrize("problemType", ['ODE', 'DAE'])
def test_interpolate(quad_type, problemType):
    r"""Interpolation in ``DenseOutput`` is tested by evaluating the polynomials at ``t_eval``. The interpolated values are then compared with the reference error."""
    from pySDC.projects.DAE import DenseOutput
    from pySDC.helpers.stats_helper import get_sorted

    solutionStats = runSimulation(quad_type=quad_type, problemType=problemType)

    # get values of u and corresponding times for all nodes
    if solutionStats is not None:
        u_dense = get_sorted(solutionStats, type='u_dense', sortby='time', recomputed=False)
        nodes_dense = get_sorted(solutionStats, type='nodes_dense', sortby='time')
        sol = DenseOutput(nodes_dense, u_dense)

    
    