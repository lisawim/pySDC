import numpy as np

from qmat.lagrange import LagrangeApproximation


class DenseOutput:
    """
    Initialize the dense output class with given data.

    Parameters
    ----------
    nodes : list of tuples
        List where each tuple contains:
        (time_step_end, array of nodes within the step)
    uValues : list of tuples
        List where each tuple contains:
        (time_step_end, list of mesh values at each stage)
    """

    def __init__(self, nodes, uValues):
        """Initialization routine"""

        # Extract the initial time from the first entry in the first sublist of nodes
        self.t0 = nodes[0][1][0]
        self.t = np.array([self.t0] + [entry[0] for entry in nodes])

        # Extract the stage times and values
        self.nodes = [np.array(entry[1]) for entry in nodes]
        self.uValues = [[np.array(mesh) for mesh in entry[1]] for entry in uValues]

    def _find_time_interval(self, t):
        r"""
        Find the interval :math:`[t_n, t_{n+1}]` that contains :math:`t`.
        
        Parameters
        ----------
        t : float
            The time at which to find the solution.
        
        Returns
        -------
        index : int
            Index n such that ``t[n] <= t < t[n+1]``.
        """

        if t < self.t[0] or t > self.t[-1]:
            raise ValueError("t is out of the range of the provided time steps.")
        return np.searchsorted(self.t, t, side='right') - 1

    def _interpolate(self, t, index):
        """
        Interpolate the solution at time t for the interval corresponding to the index.

        Parameters:
        - t: float
            The time at which to interpolate the solution.
        - index: int
            The index of the time interval to use for interpolation.
        
        Returns:
        - y: array-like
            The interpolated solution at time t.
        """
        nodes = self.nodes[index]
        uValues = self.uValues[index]

        uValuesInterp = []

        # Assuming each mesh value has the same dimensionality, so we use the first value's shape to determine dimensions
        nDim = len(uValues[0])

        # Interpolate each dimension separately
        for dim in range(nDim):
            # Extract the values for the current dimension
            uValuesDim = [value[dim] for value in uValues]

            # Create an LagrangeApproximation object for this dimension
            sol = LagrangeApproximation(points=nodes, fValues=uValuesDim)

            # Interpolate the value at time t for this dimension
            uValuesInterp.append(sol.__call__(t))

        return np.array(uValuesInterp)

    def __call__(self, t):
        r"""
        Evaluate the dense output at time :math:`t`.

        Parameters
        ----------
        t : float
            Time at which to evaluate the solution.

        Returns
        -------
        y : array-like
            Solution at time :math:`t`.
        """

        index = self._find_time_interval(t)
        return self._interpolate(t, index)