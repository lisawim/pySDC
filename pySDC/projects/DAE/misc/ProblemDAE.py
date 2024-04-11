import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, newton_krylov
from scipy.optimize._nonlin import KrylovJacobian
from scipy.sparse.linalg import gmres

from pySDC.core.Problem import ptype, WorkCounter
from pySDC.projects.DAE.misc.dae_mesh import DAEMesh


class Jacobian:
    def __init__(self, u, func, rdiff=1e-9):
        self.n = len(u)
        self.m = len(func(u))
        self.rdiff = rdiff

        self.jacobian = np.zeros((self.n, self.m))

    def evalJacobian(self, u, func):
        e = np.zeros(self.n)
        e[0] = 1
        for k in range(self.n):
            self.jacobian[:, k] = 1 / self.rdiff * (func(u + self.rdiff * e) - func(u))
            e = np.roll(e, 1)
        # if sweeper_label == 'FI-SDC':
            # self.jacobian = (-1) * np.array(
                # [
                    # [1 - factor, 0, factor, - factor],
                    # [0, (1 + 1e4) * factor, 0, 0],
                    # [-factor, 0, 1, 0],
                    # [-factor, -factor, 0, -factor]
                # ]
            # )
        # elif sweeper_label == 'SI-SDC':
            # self.jacobian = (-1) * np.array(
                # [
                    # [1 - factor, 0, factor, -1],
                    # [0, (1 + 1e4) * factor, 0, 0],
                    # [-factor, 0, 1, 0],
                    # [-factor, -factor, 0, -1]
                # ]
            # )

    def invJacobian(self):
        return np.linalg.inv(self.jacobian)


class ptype_dae(ptype):
    r"""
    This class implements a generic DAE class and illustrates the interface class for DAE problems.
    It ensures that all parameters are passed that are needed by DAE sweepers.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the problem class.
    newton_tol : float
        Tolerance for the nonlinear solver.

    Attributes
    ----------
    work_counters : WorkCounter
        Counts the work, here the number of function calls during the nonlinear solve is logged and stored
        in work_counters['newton']. The number of each function class of the right-hand side is then stored
        in work_counters['rhs']
    """

    dtype_u = DAEMesh
    dtype_f = DAEMesh

    def __init__(self, nvars, newton_tol, method):
        """Initialization routine"""
        super().__init__((nvars, None, np.dtype('float64')))
        self._makeAttributeAndRegister('nvars', 'newton_tol', 'method', localVars=locals(), readOnly=True)

        self.work_counters['newton'] = WorkCounter()
        self.work_counters['linear'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()

        self.niter_newton = 0
        self.niter_linear = 0

        # logs Newton iterations for each node as list
        self.niter_newton_node = []
        self.niter_linear_node = []
        self.get_pr_norm = 0
        self.pr_norm = []
        self.pr_norm_iters = []
        self.count_pr_norm_plots = 1

    def solve_system(self, impl_sys, u0, t):
        r"""
        Solver for nonlinear implicit system (defined in sweeper).

        Parameters
        ----------
        impl_sys : callable
            Implicit system to be solved.
        u0 : dtype_u
            Initial guess for solver.
        t : float
            Current time :math:`t`.

        Returns
        -------
        me : dtype_u
            Numerical solution.
        """
        me = self.dtype_u(self.init)
        def implSysAsNumpy(unknowns, **kwargs):
            me.diff[:] = unknowns[: np.size(me.diff)].reshape(me.diff.shape)
            me.alg[:] =  unknowns[np.size(me.diff) :].reshape(me.alg.shape)
            sys = impl_sys(me, **kwargs)
            return np.append(sys.diff.flatten(), sys.alg.flatten())  # TODO: more efficient way?

        if self.method in ('hybr'):
            opt = root(
                implSysAsNumpy,
                np.append(u0.diff.flatten(), u0.alg.flatten()),
                method=self.method,
                tol=self.newton_tol,
            )
            sol = opt.x

            self.work_counters['newton'].niter += opt.nfev

        elif self.method == 'gmres':
            callback_pr_norm = lambda res: self.pr_norm_iters.append(res)

            u = np.append(u0.diff.flatten(), u0.alg.flatten())

            # computes right-hand side + define linear operator using setup method from class
            rhs = implSysAsNumpy(u)
            J = Jacobian(rhs, implSysAsNumpy)
            J.evalJacobian(u, implSysAsNumpy)

            n = 0
            newton_maxiter = 100
            while n < newton_maxiter:
                # check for termination
                res = np.linalg.norm(rhs, np.inf)
                if res < self.newton_tol:
                    break

                # compute direction for Newton, default for restart here: restart after 4(=size of DAE system) iterations
                dx, _ = gmres(
                    J.jacobian,
                    rhs,
                    rtol=1e-12,
                    maxiter=1000,
                    callback=self.callback_linear(),
                    callback_type='legacy',
                )

                # Update step with direction dx
                u = u - dx

                rhs = implSysAsNumpy(u)
                J.evalJacobian(u, implSysAsNumpy)
                
                n += 1
                self.work_counters['newton']()
                self.niter_newton += 1

            sol = u
            self.pr_norm.append(self.pr_norm_iters)
            self.pr_norm_iters = []
            self.work_counters['linear'].niter += self.niter_linear
            self.niter_linear_node.append(self.niter_linear)
            self.niter_linear = 0

        else:
            raise NotImplementedError
        
        self.niter_newton_node.append(self.niter_newton)
        # print(self.niter_newton_node)
        # print()
        self.niter_newton = 0

        # if abs(t - 0.01) < 1e-14:
        #     plt.figure()
        #     for m in range(len(self.pr_norm)):
        #         plt.semilogy(np.arange(1, len(self.pr_norm[m]) + 1), self.pr_norm[m], label=rf'Node $\tau_{m+1}$')
        #     plt.xlabel('Iterations', fontsize=16)
        #     plt.ylabel('pr_norm')
        #     plt.ylim(1e-16, 1e0)
        #     plt.legend(loc='best')
        #     self.pr_norm = []
        #     plt.savefig(f"data/LinearTestDAEMinion/Talk/Errors/pr_norm/{self.count_pr_norm_plots}_pr_normEachNode_FISDC_M=6_IE_dt=0.01.png", dpi=300, bbox_inches='tight')
        #     # plt.savefig(f"data/LinearTestDAEMinion/Talk/Errors/pr_norm/{self.count_pr_norm_plots}_pr_normEachNode_SISDC_M=6_IE_dt=0.01.png", dpi=300, bbox_inches='tight')
        #     self.count_pr_norm_plots += 1

        me.diff[:] = sol[: np.size(me.diff)].reshape(me.diff.shape)
        me.alg[:] = sol[np.size(me.diff) :].reshape(me.alg.shape)
        return me
    
    def du_exact(self, t):
        r"""
        Interface for derivative of exact solution. Note that in
        DAE case, this function might needed when for SDC a provisional
        solution is computed via a low-order method.

        Parameters
        ----------
        t : float
            Time of the derivative of exact solution 
        """
        raise NotImplementedError("ERROR: problem has to implement du_exact(self, t)!")
    
    def callback_linear(self):
        self.niter_linear += 1

    def getNitersNewtonNode(self):
        return self.niter_newton_node

    def setNitersNewtonNode(self):
        self.niter_newton_node = []

    def getNitersLinearNode(self):
        return self.niter_linear_node

    def setNitersLinearNode(self):
        self.niter_linear_node = []
