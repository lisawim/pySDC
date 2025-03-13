import numpy as np
from scipy.linalg import solve

from pySDC.core.errors import ParameterError
from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta, ButcherTableau

from qmat import Q_GENERATORS


class Collocation(RungeKutta):
    def __init__(self, params):
        super().__init__(params)

        self.newton_tol = 1e-14
        self.newton_maxiter = 11

    def update_nodes(self):
        """
        Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes.
        """

        lvl = self.level
        prob = lvl.prob

        N = len(lvl.u[0].flatten())
        M = self.coll.num_nodes

        # TODO: For constrainedDAE quadrature only needs to be applied to differential parts!

        U = np.zeros(M * N)
        F = np.zeros(M * N)
        G = np.zeros(M * N)

        # Initial guess for the stage values
        for m in range(M):
            U[m * N:(m + 1) * N] = lvl.u[0].flatten()

        # Start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:
            for m in range(M):
                # Get function values
                f = prob.eval_f(U[m * N:(m + 1) * N].reshape(lvl.u[0].shape).view(type(lvl.u[0])), lvl.time + lvl.dt * self.coll.nodes[m + 1])

                # Convert it into a numpy array and insert it into F
                F[m * N:(m + 1) * N] = f.flatten()

            for m in range(M):
                G[m * N:(m + 1) * N] = U[m * N:(m + 1) * N] - lvl.u[0].flatten() - lvl.dt * sum(self.QI[m + 1, j + 1] * F[j * N:(j + 1) * N] for j in range(M))

            # Build the Jacobian matrix
            J = np.zeros((M * N, M * N))
            for i in range(M):
                for j in range(M):
                    if i == j:
                        J_ij = np.eye(N) - lvl.dt * self.QI[i + 1, j + 1] * self.numerical_jacobian(lvl.time + lvl.dt * self.coll.nodes[j + 1], U[j * N:(j + 1) * N])
                    else:
                        J_ij = -lvl.dt * self.QI[i + 1, j + 1] * self.numerical_jacobian(lvl.time + lvl.dt * self.coll.nodes[j + 1], U[j * N:(j + 1) * N])
                    J[i * N:(i + 1) * N, j * N:(j + 1) * N] = J_ij

            # Solve the linear system for updates
            delta = solve(J, G)

            # Update stage values
            U -= delta

            # Convergence check
            res = np.linalg.norm(delta)
            if res < self.newton_tol:
                break

            n += 1

        for m in range(M):
            # Get function values
            f = prob.eval_f(U[m * N:(m + 1) * N].reshape(lvl.u[0].shape).view(type(lvl.u[0])), lvl.time + lvl.dt * self.coll.nodes[m + 1])

            # Convert it into a numpy array and insert it into F
            F[m * N:(m + 1) * N] = f.flatten()

        # Compute the next solution
        u_next = lvl.u[0].flatten() + lvl.dt * sum(self.coll.weights[m] * F[m * N:(m + 1) * N] for m in range(M))

        lvl.u[-1][:] = u_next.reshape(lvl.u[0].shape).view(type(lvl.u[0]))[:]

        # indicate presence of new values at this level
        lvl.status.updated = True

        return None

    def numerical_jacobian(self, t, u, epsilon=1e-8):
        """
        Approximate the Jacobian of f with respect to y using finite differences.

        Parameters:
        - f: Function representing the ODE dy/dt = f(t, y).
        - t: Current time.
        - y: Current solution.
        - epsilon: Perturbation for finite differences.

        Returns:
        - J: Jacobian matrix.
        """

        lvl = self.level
        prob = lvl.prob

        N = len(u.flatten())
        J = np.zeros((N, N))

        # Get function value for u
        f0 = prob.eval_f(u.reshape(lvl.u[0].shape).view(type(lvl.u[0])), t).flatten()

        for i in range(N):
            u_perturbed = u.copy()

            u_perturbed += epsilon

            # Get function value for u
            f_perturbed = prob.eval_f(u_perturbed.reshape(lvl.u[0].shape).view(type(lvl.u[0])), t).flatten()

            J[:, i] = (f_perturbed - f0) / epsilon

        return J


class RadauIIA5(Collocation):
    """Method of Radau IIa family of order 5."""
    generator = Q_GENERATORS["Collocation"](
        nNodes=3, nodeType="LEGENDRE", quadType="RADAU-RIGHT", tLeft=0, tRight=1
    )

    nodes = generator.nodes.copy()
    weights = generator.weights.copy()
    matrix = generator.Q
    ButcherTableauClass = ButcherTableau


class RadauIIA7(Collocation):
    """Method of Radau IIa family of order 7."""
    generator = Q_GENERATORS["Collocation"](
        nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT", tLeft=0, tRight=1
    )

    nodes = generator.nodes.copy()
    weights = generator.weights.copy()
    matrix = generator.Q
    ButcherTableauClass = ButcherTableau
