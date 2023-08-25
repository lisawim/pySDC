import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh


class ptype_dae(ptype):
    """
    Interface class for DAE problems. Ensures that all parameters are passed that are needed by DAE sweepers.

    Parameters
    ----------
    nvars : int
        Number of unknowns.
    newton_tol : float
        Inner tolerance for DAE sweepers to solve the nonlinear system.
    diff_nvars : int, optional
        Number of differential variables in the system (needed for semi-explicit treatment).
    """

    dtype_u = mesh
    dtype_f = mesh

    def __init__(self, nvars, newton_tol, diff_nvars=None):
        """Initialization routine"""
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'newton_tol', 'diff_nvars', localVars=locals(), readOnly=True)

    def eval_f(self, u, t, du=None):
        """
        Abstract interface to RHS computation of the ODE or DAE.

        Parameters
        ----------
        u : dtype_u
            Current values of solution u.
        t : float
            Current time.
        du : bool, optional
            Current values of derivative of differential variable u (in case of a DAE).

        Returns
        -------
        f : dtype_f
            The RHS values.
        """
        raise NotImplementedError('ERROR: problem has to implement eval_f(self, u, t, p=None, du=None)')