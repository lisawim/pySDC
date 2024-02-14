import numpy as np

from pySDC.implementations.datatype_classes.MultiComponentMesh import MultiComponentMesh
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import DataError


class DAEMesh(MultiComponentMesh):
    components = ['diff', 'alg']

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        """
        Overriding default ufunc, cf. https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs
        """
        args = []
        comm = None
        for i, input_ in enumerate(inputs):
            if isinstance(input_, DAEMesh):
                args.append(input_.view(np.ndarray))
                comm = input_.comm
            else:
                args.append(input_)

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs).view(DAEMesh)
        if type(self) == type(results):
            results._comm = comm
            results.__dict__ = self.__dict__
        return results

    # def __setitem__(self, key, value):
    #     """
    #     Overloading the set item operator
    #     """
