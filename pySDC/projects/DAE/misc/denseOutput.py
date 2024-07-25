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

        # Extract the end times of each step
        self.times = np.array([entry[0] for entry in nodes])

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
            Index n such that ``times[n] <= t < times[n+1]``.
        """

        if t < self.times[0] or t > self.times[-1]:
            raise ValueError("t is out of the range of the provided time steps.")
        return np.searchsorted(self.times, t, side='right') - 1

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
        uValues = self.nodes[index]

        LagrangeInterpolation = LagrangeApproximation(points=nodes, fValues=uValues)
        return LagrangeInterpolation.__call__(t)

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