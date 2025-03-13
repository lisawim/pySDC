from pySDC.implementations.datatype_classes.mesh import MultiComponentMesh


class MeshDAE(MultiComponentMesh):
    r"""
    Datatype for DAE problems. The solution of the problem can be splitted in the differential part
    and in an algebraic part.

    This data type can be used for the solution of the problem itself as well as for its derivative.
    """

    components = ['diff', 'alg']

    # def __new__(cls, init, *args, **kwargs):
    #     obj = super().__new__(cls, init, *args, **kwargs)
    #     if isinstance(init, tuple):
    #         # Determine the shape for diff and alg
    #         if isinstance(init[0], tuple):
    #             shape = (init[0], init[1])

    #         elif isinstance(init[0], int):
    #             shape = (init[0], init[0])

    #         else:
    #             raise ValueError("Invalid format for init[0]. Must be tuple or integer.")

    #         obj.init_shapes = shape  # Store shapes for diff and alg

    #     return obj

    # @property
    # def diff(self):
    #     """Return the `diff` component with the correct shape."""
    #     # return self[0, :self.init_shapes[0]]
    #     return self[0, ...]

    # @property
    # def alg(self):
    #     """Return the `alg` component with the correct shape."""
    #     # return self[1, :self.init_shapes[1]]
    #     return self[1, ...]
