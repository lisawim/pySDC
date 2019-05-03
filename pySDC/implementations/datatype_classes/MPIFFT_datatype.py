from mpi4py import MPI
import numpy as np

from pySDC.core.Errors import DataError
from mpi4py_fft import PFFT, DistArray


class fft_datatype(np.ndarray):
    """
    """

    def __new__(cls, init, val=0.0):
        if isinstance(init, fft_datatype):
            obj = np.ndarray.__new__(cls, init.shape, dtype=init.dtype, buffer=None)
            obj[:] = init[:]
            obj._comm = init._comm
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)) \
                and isinstance(init[2], np.dtype):
            obj = np.ndarray.__new__(cls, init[0], dtype=init[2], buffer=None)
            obj.fill(val)
            obj._comm = init[1]
        else:
            raise NotImplementedError(type(init))
        return obj

    @property
    def comm(self):
        return self._comm

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._comm = getattr(obj, '_comm', None)

    def __abs__(self):
        """
        Overloading the abs operator for fft mesh types

        Returns:
            float: absolute maximum of all mesh values
        """
        # take absolute values of the mesh values
        local_absval = float(np.amax(np.ndarray.__abs__(self)))

        if self.comm is not None:
            if self.comm.Get_size() > 1:
                global_absval = 0.0
                global_absval = max(self.comm.allreduce(sendobj=local_absval, op=MPI.MAX), global_absval)
            else:
                global_absval = local_absval
        else:
            global_absval = local_absval

        return float(global_absval)

    def send(self, dest=None, tag=None, comm=None):
        """
        Routine for sending data forward in time (blocking)

        Args:
            dest (int): target rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """

        comm.Send(self[:], dest=dest, tag=tag)
        return None

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
        return comm.Isend(self[:], dest=dest, tag=tag)

    def recv(self, source=None, tag=None, comm=None):
        """
        Routine for receiving in time

        Args:
            source (int): source rank
            tag (int): communication tag
            comm: communicator

        Returns:
            None
        """
        comm.Recv(self[:], source=source, tag=tag)
        return None

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


class rhs_imex_fft(object):
    """
    """

    def __init__(self, init, val=0.0):
        """
        Initialization routine

        Args:
            init: another pmesh_datatype or a tuple containing the communicator and the local dimensions
            val (float): an initial number (default: 0.0)
        Raises:
            DataError: if init is none of the types above
        """
        if isinstance(init, type(self)):
            self.impl = fft_datatype(init.impl)
            self.expl = fft_datatype(init.expl)
        elif isinstance(init, tuple) and (init[1] is None or isinstance(init[1], MPI.Intracomm)):
            self.impl = fft_datatype(init, val=val)
            self.expl = fft_datatype(init, val=val)
        # something is wrong, if none of the ones above hit
        else:
            raise DataError('something went wrong during %s initialization' % type(self))

    def __sub__(self, other):
        """
        Overloading the subtraction operator for rhs types

        Args:
            other: rhs object to be subtracted
        Raises:
            DataError: if other is not a rhs object
        Returns:
            differences between caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            me = rhs_imex_fft(self)
            me.impl = self.impl - other.impl
            me.expl = self.expl - other.expl
            return me
        else:
            raise DataError("Type error: cannot subtract %s from %s" % (type(other), type(self)))

    def __add__(self, other):
        """
         Overloading the addition operator for rhs types

        Args:
            other: rhs object to be added
        Raises:
            DataError: if other is not a rhs object
        Returns:
            sum of caller and other values (self-other)
        """

        if isinstance(other, type(self)):
            me = rhs_imex_fft(self)
            me.impl = self.impl + other.impl
            me.expl = self.expl + other.expl
            return me
        else:
            raise DataError("Type error: cannot add %s to %s" % (type(other), type(self)))

    def __rmul__(self, other):
        """
        Overloading the right multiply by factor operator for mesh types

        Args:
            other (float): factor
        Raises:
            DataError: is other is not a float
        Returns:
             copy of original values scaled by factor
        """

        if isinstance(other, float):
            me = rhs_imex_fft(self)
            me.impl = other * self.impl
            me.expl = other * self.expl
            return me
        else:
            raise DataError("Type error: cannot multiply %s to %s" % (type(other), type(self)))