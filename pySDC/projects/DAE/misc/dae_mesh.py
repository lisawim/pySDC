import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.core.Errors import DataError


class DAEMesh(object):
    """
    TODO: write docu
    RHS data type for meshes with implicit and explicit components

    This data type can be used to have RHS with 2 components (here implicit and explicit)

    Attributes:
        impl (mesh.mesh): implicit part
        expl (mesh.mesh): explicit part
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: can either be a tuple (one int per dimension) or a number (if only one dimension is requested)
                  or another imex_mesh object
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """

        if isinstance(init, type(self)):
            self.diff = mesh(init.diff)
            self.alg = mesh(init.alg)
        elif (
            isinstance(init, tuple)
            and (init[1] is None or str(type(init[1])) == "MPI.Intracomm")
            and isinstance(init[2], np.dtype)
        ):
            self.diff = mesh((init[0][0], init[1], init[2]), val=val)
            self.alg = mesh((init[0][1], init[1], init[2]), val=val)

        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __abs__(self):
        """
        It chooses the maximum value between the differential part and the algebraic part. If the
        problem contains no algebraic part, then the maximum values is computed over the differential parts.
        """
        return max([abs(self.diff), abs(self.alg)]) if len(self.alg) > 0 else max([abs(self.diff)])

    def __add__(self, other):
        """
        Overloading the addition operator for DAE meshes.
        """

        if isinstance(other, type(self)):
            me = DAEMesh(self)
            me.diff[:] = self.diff + other.diff
            me.alg[:] = self.alg + other.alg
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for DAE meshes.
        """

        if isinstance(other, type(self)):
            me = DAEMesh(self)
            me.diff[:] = self.diff - other.diff
            me.alg[:] = self.alg - other.alg
            return me
        else:
            raise DataError("Type error: cannot subtract %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for DAE meshes.
        """

        if isinstance(other, float):
            me = DAEMesh(self)
            me.diff[:] = other * self.diff
            me.alg[:] = other * self.alg
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))

    def __str__(self):
        return f"({self.diff}, {self.alg})"

    def __getitem__(self, item):
        if type(item) in [int, slice]:
            asNumpy = np.append(self.diff[:], self.alg[:])
            return asNumpy.__getitem__(item)
        else:
            return super().__getitem__(item)

    # def __setitem__(self, key, value):
    #     if type(key) in [int, slice]:
    #         asNumpy = np.append(self.diff[:], self.alg[:])
    #         asNumpy.__setitem__(key, value)
    #         self.diff[:] = asNumpy[:len(self.diff)]  # doesn’t work for ND, only 1D need to do flatten and unflatten or sth
    #         self.alg[:] = asNumpy[len(self.diff):]
    #     else:
    #         super().__setitem__(key, value)

    def isend(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (non-blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            request handle
        """
        return comm.Issend(self[:], dest=dest, tag=tag)

    def irecv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        print('hi', self)
        return comm.Irecv(self[:], source=source, tag=tag)

    def bcast(self, root=None, comm=None):
        """
        Routine for broadcasting values

        Args:
            root (int): process with value to broadcast
            comm: communicator

        Returns:
            broadcasted values
        """
        comm.Bcast(self[:], root=root)
        return self
