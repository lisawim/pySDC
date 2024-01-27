import numpy as np
from scipy.optimize import root, newton_krylov
from scipy.optimize._nonlin import KrylovJacobian
from scipy.sparse.linalg import gmres

from pySDC.core.Problem import ptype, WorkCounter
from pySDC.projects.DAE.misc.dae_mesh import DAEMesh


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
        # self.work_counters['linear'] = WorkCounter()
        self.work_counters['rhs'] = WorkCounter()
        self.niter = 0
        self.niter_linear_node = []

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

        if self.method in ('hybr', 'krylov'):#self.method == 'hybr':
            opt = root(
                implSysAsNumpy,
                np.append(u0.diff.flatten(), u0.alg.flatten()),
                method=self.method,
                tol=self.newton_tol,
            )
            sol = opt.x
            print(opt)
            if self.method == 'hybr':
                self.setNiter(opt.nfev)
            elif self.method == 'krylov':
                self.setNiter(opt.nit)

        elif self.method == 'gmres':
            # class to approximate Jacobian
            J = KrylovJacobian()
            u = np.append(u0.diff.flatten(), u0.alg.flatten())
            # opt = newton_krylov(
            #     implSysAsNumpy,
            #     np.append(u0.diff.flatten(), u0.alg.flatten()),
            #     method='gmres',
            #     f_tol=self.newton_tol,
            #     line_search=None,
            #     callback=self.callback(),
            # )
            # sol = opt

            # computes right-hand side + define linear operator using setup method from class
            Fu = implSysAsNumpy(u)
            J.setup(u.copy(), Fu, implSysAsNumpy)

            n = 0
            newton_maxiter = 100
            while n < newton_maxiter:
                # check for termination
                res = np.linalg.norm(Fu, np.inf)
                if res < self.newton_tol:
                    break

                # compute direction for Newton
                sol_lin, info = gmres(
                    J.op,
                    Fu,
                    tol=1e-10,
                    maxiter=100,
                    callback=self.callback(),
                    callback_type='legacy',
                )
                dx = -sol_lin

                if np.linalg.norm(dx) == 0:
                    raise ValueError("Jacobian inversion yielded zero vector. "
                                     "This indicates a bug in the Jacobian "
                                     "approximation.")

                u = u + dx

                Fu = implSysAsNumpy(u)
                J.update(u.copy(), Fu)
                
                n += 1

            sol = u

        else:
            raise NotImplementedError
        
        self.work_counters['newton'].niter += self.niter
        self.niter_linear_node.append(self.niter)
        
        me.diff[:] = sol[: np.size(me.diff)].reshape(me.diff.shape) # opt.x[: np.size(me.diff)].reshape(me.diff.shape)
        me.alg[:] = sol[np.size(me.diff) :].reshape(me.alg.shape) # opt.x[np.size(me.diff) :].reshape(me.alg.shape)
        # me[:] = opt.x
        # print(t, self.niter)#opt.nfev
        self.niter = 0
        return me
    
    def setNiter(self, niter):
        self.niter = niter

    def callback(self):
        self.niter += 1

    def getNitersLinearNode(self):
        return self.niter_linear_node

    def setNitersLinearNode(self):
        self.niter_linear_node = []
