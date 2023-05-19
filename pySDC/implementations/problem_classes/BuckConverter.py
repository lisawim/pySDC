import numpy as np

from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, imex_mesh


class buck_converter(ptype):
    r"""
    Example implementing the model of a buck converter, which is also called a step-down converter. The problem of simulating the
    converter consists of a nonhomogeneous linear system of ordinary differential equations (ODEs)

    .. math::
        \frac{\partial u}{\partial t} = Au+\vec{f}

    using an initial condition. A fully description of the buck converter can be found in the description of the PinTSimE project.

    Parameters
    ----------
    duty : float
        Cycle between zero and one indicates the time period how long the converter stays on one switching state
        until it switches to the other state.
    fsw : int
        Switching frequency, it is used to determine the number of time steps after the switching state is changed.
    Vs : float
        Voltage at the voltage source :math:`V_s`.
    Rs : float
        Resistance of the resistor :math:`R_s` at the voltage source.
    C1 : float
        Capacitance of the capacitor :math:`C_1`.
    Rp : float
        Resistance of the resistor in front of the inductor.
    L1 : float
        Inductance of the inductor :math:`L_1`.
    C2 : float
        Capacitance of the capacitor :math:`C_2`.
    Rl : float
        Resistance of the resistor :math:`R_{\pi}`

    Attributes
    ----------
        A: system matrix, representing the 3 ODEs
    """

    dtype_u = mesh
    dtype_f = imex_mesh

    def __init__(self, duty, fsw, Vs, Rs, C1, Rp, L1, C2, Rl):
        """Initialization routine"""

        # invoke super init, passing number of dofs
        nvars = 3
        super().__init__(init=(nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'duty', 'fsw', 'Vs', 'Rs', 'C1', 'Rp', 'L1', 'C2', 'Rl', localVars=locals(), readOnly=True
        )

        self.A = np.zeros((nvars, nvars))

    def eval_f(self, u, t):
        """
        Routine to evaluate the right-hand side of the problem.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution.
        t : float
            Current time of the numerical solution is computed.

        Returns
        -------
        f : dtype_f
            The right-hand side of the problem.
        """
        Tsw = 1 / self.fsw

        f = self.dtype_f(self.init, val=0.0)
        f.impl[:] = self.A.dot(u)

        if 0 <= ((t / Tsw) % 1) <= self.duty:
            f.expl[0] = self.Vs / (self.Rs * self.C1)
            f.expl[2] = 0

        else:
            f.expl[0] = self.Vs / (self.Rs * self.C1)
            f.expl[2] = -(self.Rp * self.Vs) / (self.L1 * self.Rs)

        return f

    def solve_system(self, rhs, factor, u0, t):
        r"""
        Simple linear solver for :math:`(I-factor\cdot A)\vec{u}=\vec{rhs}`.

        Parameters
        ----------
        rhs : dtype_f
            Right-hand side for the linear system.
        factor : float
            Abbrev. for the local stepsize (or any other factor required).
        u0 : dtype_u
            Initial guess for the iterative solver.
        t : float
            Current time (e.g. for time-dependent BCs).

        Returns
        -------
        me : dtype_u
            The solution as mesh.
        """
        Tsw = 1 / self.fsw
        self.A = np.zeros((3, 3))

        if 0 <= ((t / Tsw) % 1) <= self.duty:
            self.A[0, 0] = -1 / (self.C1 * self.Rs)
            self.A[0, 2] = -1 / self.C1

            self.A[1, 1] = -1 / (self.C2 * self.Rl)
            self.A[1, 2] = 1 / self.C2

            self.A[2, 0] = 1 / self.L1
            self.A[2, 1] = -1 / self.L1
            self.A[2, 2] = -self.Rp / self.L1

        else:
            self.A[0, 0] = -1 / (self.C1 * self.Rs)

            self.A[1, 1] = -1 / (self.C2 * self.Rl)
            self.A[1, 2] = 1 / self.C2

            self.A[2, 0] = self.Rp / (self.L1 * self.Rs)
            self.A[2, 1] = -1 / self.L1

        me = self.dtype_u(self.init)
        me[:] = np.linalg.solve(np.eye(self.nvars) - factor * self.A, rhs)
        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t.

        Parameters
        ----------
        t : float
            Time of the exact solution.

        Returns
        -------
        me : dtype_u
            The exact solution.
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)

        me[0] = 0.0  # v1
        me[1] = 0.0  # v2
        me[2] = 0.0  # p3

        return me
