import numpy as np

try:
    # TODO : mpi4py cannot be imported before dolfin when using fenics mesh
    # see https://github.com/Parallel-in-Time/pySDC/pull/285#discussion_r1145850590
    # This should be dealt with at some point
    from mpi4py import MPI
except ImportError:
    MPI = None

from pySDC.implementations.datatype_classes.mesh import mesh, MultiComponentMesh


class MeshDAE(MultiComponentMesh):
    r"""
    Datatype for DAE problems. The solution of the problem can be splitted in the differential part
    and in an algebraic part.

    This data type can be used for the solution of the problem itself as well as for its derivative.
    """

    components = ['diff', 'alg']

    def __new__(cls, init, val=0.0, **kwargs):
        if (
            isinstance(init, tuple)
            and isinstance(init[0], (tuple, list))
            and len(init[0]) == 2
            and (init[1] is None or isinstance(init[1], MPI.Intracomm))
            and isinstance(init[2], np.dtype)
        ):
            n_diff, n_alg = init[0]
            comm = init[1]
            dtype = init[2]

            # Gesamt-Vektorlänge: Summe beider Teile
            total_len = n_diff + n_alg

            # Wir legen einen einfachen mesh-Container an
            base = mesh((total_len, comm, dtype), val, **kwargs)

            # View in unseren neuen Typ
            obj = base.view(cls)

            obj._n_diff = n_diff
            obj._n_alg  = n_alg
            return obj

        else:
            print("else")
            obj = super().__new__(cls, init, val, **kwargs)
            length = obj.shape[1] if obj.ndim > 1 else obj.shape[0]
            obj._n_diff = obj._n_alg = length
            return obj

    def __getattr__(self, name):
        if name == 'diff':
            return self[: self._n_diff].view(mesh)

        if name == 'alg':
            start = self._n_diff
            end   = self._n_diff + self._n_alg
            return self[start:end].view(mesh)

        return super().__getattribute__(name)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        for inp in inputs:
            if isinstance(inp, MeshDAE):
                args.append(inp.view(np.ndarray))
            else:
                args.append(inp)
        result = super().__array_ufunc__(ufunc, method, *args, out=out, **kwargs)
        return result.view(type(self))
