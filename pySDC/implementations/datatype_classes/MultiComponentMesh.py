import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import DataError


class MultiComponentMesh(mesh):
    components = []

    def __new__(cls, init, *args, **kwargs):
        if isinstance(init, tuple):
            obj = super().__new__(cls, ((len(cls.components), init[0]), *init[1:]), *args, **kwargs)
        else:
            obj = super().__new__(cls, init, *args, **kwargs)

        for comp, i in zip(cls.components, range(len(cls.components))):
            obj.__dict__[comp] = obj[i]
        return obj

    def __array_ufunc__(self, *args, **kwargs):
        results = super().__array_ufunc__(*args, **kwargs).view(type(self))

        if type(self) == type(results) and self.flags['OWNDATA']:
            for comp, i in zip(self.components, range(len(self.components))):
                results.__dict__[comp] = results[i]
        return results

    # def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
    #     """
    #     Overriding default ufunc, cf. https://numpy.org/doc/stable/user/basics.subclassing.html#array-ufunc-for-ufuncs
    #     """
    #     args = []
    #     comm = None
    #     for i, input_ in enumerate(inputs):
    #         if isinstance(input_, type(self)):
    #             args.append(input_.view(np.ndarray))
    #             comm = input_.comm
    #         else:
    #             args.append(input_)

    #     results = super().__array_ufunc__(ufunc, method, *args, **kwargs).view(type(self))
    #     if type(self) == type(results):
    #         results._comm = comm
    #         results.__dict__ = self.__dict__
    #     return results

    # def __setitem__(self, key, value):
    #     """
    #     Overloading the set item operator
    #     """
    #     if type(key) in [int, slice]:
    #         # asNumpy = np.append(self.diff[:], self.alg[:])
    #         # asNumpy.__setitem__(key, value)
    #         nval = len(value)
    #         print(dir(value))
    #         for comp, i in zip(self.components, range(len(self.components))):
    #             self.__dict__[comp] = value[i * nval : i * nval + nval]
                
    #         # self.diff[:] = value[:int(n / 2)]  # doesn’t work for ND, only 1D need to do flatten and unflatten or sth
    #         # self.alg[:] = value[int(n / 2):]
    #     else:
    #         super().__setitem__(key, value)