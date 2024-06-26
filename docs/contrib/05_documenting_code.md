# Documenting Code

When developing a new class or function, or improving current classes in `pySDC`, adding Python docstring to document the code is important, in particular :

- to help developer understanding how classes and functions work when reading the code
- to help user understanding how classes and functions work when reading the [documentation](https://parallel-in-time.org/pySDC/#api-documentation)

`pySDC` follows the [NumPy Style Python Docstring](https://numpydoc.readthedocs.io/en/latest/format.html), below is simplified example
for a class and a function :

> :bell: Don't document the `init` function, but rather the class itself. Also describe parameters (given to the `__init__`) and attributes (stored into the class) separately.

```python
class LagrangeApproximation(object):
    r"""
    Class approximating any function on a given set of points using barycentric
    Lagrange interpolation.

    Let note :math:`(t_j)_{0\leq j<n}` the set of points, then any scalar
    function :math:`f` can be approximated by the barycentric formula :

    .. math::
        p(x) =
        \frac{\displaystyle \sum_{j=0}^{n-1}\frac{w_j}{x-x_j}f_j}
        {\displaystyle \sum_{j=0}^{n-1}\frac{w_j}{x-x_j}},

    where :math:`f_j=f(t_j)` and

    .. math::
        w_j = \frac{1}{\prod_{k\neq j}(x_j-x_k)}

    are the barycentric weights.
    The theory and implementation is inspired from `this paper <http://dx.doi.org/10.1137/S0036144502417715>`_.

    Parameters
    ----------
    points : list, tuple or np.1darray
        The given interpolation points, no specific scaling, but must be
        ordered in increasing order.

    Attributes
    ----------
    points : np.1darray
        The interpolating points
    weights : np.1darray
        The associated barycentric weights
    """

    def __init__(self, points):
        pass  # Implementation ...

    @property
    def n(self):
        """Number of points"""
        pass  # Implementation ...

    def getInterpolationMatrix(self, times):
        r"""
        Compute the interpolation matrix for a given set of discrete "time"
        points.

        For instance, if we note :math:`\vec{f}` the vector containing the
        :math:`f_j=f(t_j)` values, and :math:`(\tau_m)_{0\leq m<M}`
        the "time" points where to interpolate.
        Then :math:`I[\vec{f}]`, the vector containing the interpolated
        :math:`f(\tau_m)` values, can be obtained using :

        .. math::
            I[\vec{f}] = P_{Inter} \vec{f},

        where :math:`P_{Inter}` is the interpolation matrix returned by this
        method.

        Parameters
        ----------
        times : list-like or np.1darray
            The discrete "time" points where to interpolate the function.

        Returns
        -------
        PInter : np.2darray(M, n)
            The interpolation matrix, with :math:`M` rows (size of the **times**
            parameter) and :math:`n` columns.
        """
        pass  # Implementation ...

```

A more detailed example is given [here ...](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)

:arrow_left: [Back to custom implementations](./04_custom_implementations.md) ---
:arrow_up: [Contributing Summary](./../../CONTRIBUTING.md) ---
:arrow_right: [Next to Adding a new project](./06_new_project.md)
