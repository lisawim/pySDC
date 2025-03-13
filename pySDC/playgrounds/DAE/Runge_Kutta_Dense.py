import numpy as np
from scipy.linalg import solve

from pySDC.core.errors import ParameterError
from pySDC.implementations.sweeper_classes.Runge_Kutta import RungeKutta, ButcherTableau

from qmat import Q_GENERATORS


class RungeKuttaDense(RungeKutta):
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


class RungeKuttaDenseConstrained(RungeKuttaDense):
    def __init__(self, params):
        super().__init__(params)
    
    # def update_nodes(self):
    #     """
    #     Update the u- and f-values at the collocation nodes -> corresponds to a single sweep over all nodes.
    #     """

    #     lvl = self.level
    #     prob = lvl.prob

    #     M = self.coll.num_nodes

    #     N_diff = len(lvl.u[0].diff)
    #     N_alg = len(lvl.u[0].alg)

    #     N = N_diff + N_alg

    #     U = np.zeros((M, N_diff + N_alg))
    #     for m in range(M):
    #         U[m, :N_diff] = lvl.u[m + 1].diff.flatten()  # Initial guess for y components
    #         U[m, N_diff:] = lvl.u[m + 1].alg.flatten()  # Initial guess for z components
        
    #     # Start newton iteration
    #     n = 0
    #     res = 99
    #     while n < self.newton_maxiter:
    #         F = np.zeros((M, N_diff + N_alg))

    #         # Evaluate f at stage values
    #         for m in range(M):
    #             # Reshape U to have the mesh data type
    #             U_reshape = U[m, :].reshape(lvl.u[0].shape).view(type(lvl.u[0]))

    #             # Evaluate f
    #             F[m, :] = prob.eval_f(U_reshape, lvl.time + lvl.dt * self.coll.nodes[m + 1]).flatten()

    #         # Build residuals and Jacobian for Newton iteration
    #         residual = np.zeros((M, N_diff + N_alg))
    #         jacobian = np.zeros((M * (N_diff + N_alg), M * (N_diff + N_alg)))

    #         for m in range(M):
    #             residual[m, :N_diff] = U[m, :N_diff] - lvl.u[0].diff.flatten() - lvl.dt * sum(self.QI[m + 1, j + 1] * F[j, :N_diff] for j in range(M))
    #             residual[m, N_diff:] = F[m, N_diff:]  # Algebraic constraints g(y, z) = 0

    #             for j in range(M):
    #                 jacobian_block = self.compute_combined_jacobian(lvl.dt, N_diff, N_alg, U[j, :], lvl.dt * self.QI[m + 1, j + 1], lvl.time + lvl.dt * self.coll.nodes[m + 1])
    #                 jacobian[m * (N_diff + N_alg):(m + 1) * (N_diff + N_alg),
    #                          j * (N_diff + N_alg):(j + 1) * (N_diff + N_alg)] = jacobian_block
                    
    #         # Solve for updates
    #         delta = solve(jacobian, -residual.flatten()).reshape((M, N_diff + N_alg))

    #         # Update stages
    #         for i in range(M):
    #             U[i, :] += delta[i, :]

    #         # Check for convergence
    #         res = np.linalg.norm(delta)
    #         if res < self.newton_tol:
    #             break

    #         n += 1
        
    #     # For stiffly accurate methods collocation update is not necessary
    #     if np.isclose(self.coll.nodes[-1], lvl.time + lvl.dt, atol=1e-14):
    #         y_next = U[-1, :N_diff]
    #         z_next = U[-1, N_diff:]
        
    #     else:
    #         # Update y_{n+1} and z_{n+1}
    #         y_next = lvl.u[0].diff.flatten() + lvl.dt * sum(self.coll.weights[i] * F[i, :N_diff] for i in range(M))

    #         # Enforce constraint g(y_{n+1}, z_{n+1}) = 0
    #         z_next = lvl.u[0].alg.flatten()

    #         m_alg = 0
    #         res_alg = 99
    #         while m_alg < self.newton_maxiter:
    #             u_next = np.array([*y_next, *z_next]).reshape(lvl.u[0].shape).view(type(lvl.u[0]))

    #             g_residual = prob.eval_f(u_next, lvl.time + lvl.dt).flatten

    #             g_residual_alg = g_residual[N_diff:]

    #             res_alg = np.linalg.norm(g_residual_alg)
    #             if res_alg < self.newton_tol:
    #                 break

    #             g_jacobian = self.compute_combined_jacobian(lvl.dt, N_diff, N_alg, u_next, 0, lvl.time + lvl.dt)
                
    #             g_jacobian_alg = g_jacobian[N_diff:, N_diff:]

    #             delta = solve(g_jacobian_alg, -g_residual_alg)

    #             z_next += delta

    #             m_alg += 1

    #     # Initialize u having the correct datatype
    #     u = prob.dtype_u(prob.init)

    #     # Store the solution in it
    #     u.diff[:] = y_next
    #     u.alg[:] = z_next

    #     # Pass the solution to the global one
    #     lvl.u[-1].diff[:] = u.diff[:]
    #     lvl.u[-1].alg[:] = u.alg[:]

    #     return None

    def update_nodes(self):
        lvl = self.level
        prob = lvl.prob

        M = self.coll.num_nodes

        N_diff = len(lvl.u[0].diff)
        N_alg = len(lvl.u[0].alg)

        N = N_diff + N_alg

        # Initial guess for differential and algebraic variables
        U = np.zeros(M * N)
        for m in range(M):
            U[m * N:m * N + N_diff] = lvl.u[m + 1].diff.flatten()
            U[m * N + N_diff:(m + 1) * N] = lvl.u[m + 1].alg.flatten()

        # Initialize F and G in advance
        F = np.zeros(M * N)
        G = np.zeros(M * N)
        dG = np.zeros((M * N, M * N))

        # Start newton iteration
        n = 0
        res = 99
        while n < self.newton_maxiter:

            # Evaluate f at stage values
            for m in range(M):
                # Reshape U to have the mesh data type
                U_reshape = U[m * N:(m + 1) * N].reshape(lvl.u[0].shape).view(type(lvl.u[0]))

                # Store current index of stage
                k = m

                # Evaluate f
                F[m * N:(m + 1) * N] = self.evaluate_F(k, N_diff, N_alg, U_reshape)

            for m in range(M):
                # Store current index of stage
                k = m

                # Compute function G at current stage
                G[m * N:(m + 1) * N] = self.evaluate_G(k, N_diff, N_alg, U, F)

                # dG[m * N:(m + 1) * N, j * N:(j + 1) * N] = dG_block
                dG_block = self.compute_jacobian(k, N_diff, N_alg, U, F)
                dG[m * N:(m + 1) * N, :] = dG_block

            # Solve for updates
            try:
                delta = solve(dG, G.flatten())
            except np.linalg.LinAlgError:
                raise ValueError("Jacobian is singular. Verify the problem setup or initial guesses.")

            # Update stages
            for i in range(M):
                U[i * N:(i + 1) * N] -= delta[i * N:(i + 1) * N]

            # Check for convergence
            res = np.linalg.norm(delta)
            if res < self.newton_tol:
                break

            n += 1

        # For stiffly accurate methods collocation update is not necessary
        if np.isclose(self.coll.nodes[-1], lvl.time + lvl.dt, atol=1e-14):
            y_next = U[-1, :N_diff]
            z_next = U[-1, N_diff:]
        
        else:
            # Update y_{n+1} and z_{n+1}
            y_next = lvl.u[0].diff.flatten() + lvl.dt * sum(self.coll.weights[i] * F[i, :N_diff] for i in range(M))

            # Enforce constraint g(y_{n+1}, z_{n+1}) = 0
            z_next = lvl.u[0].alg.flatten()

            m_alg = 0
            res_alg = 99
            while m_alg < self.newton_maxiter:
                u_next = np.hstack([y_next, z_next])

                f = prob.eval_f(u_next.reshape(lvl.u[0].shape).view(type(lvl.u[0]))).flatten()

                g_residual_alg = f[N_diff:]

                res_alg = np.linalg.norm(g_residual_alg)
                if res_alg < self.newton_tol:
                    break

                g_jacobian = self.compute_jacobian(0, N_diff, N_alg, lvl.time + lvl.dt, u_next)

                g_jacobian_alg = g_jacobian[N_diff:, N_diff:]

                delta = solve(g_jacobian_alg, -g_residual_alg)

                z_next += delta

                m_alg += 1

        # Initialize u having the correct datatype
        u = prob.dtype_u(prob.init)

        # Store the solution in it
        u.diff[:] = y_next
        u.alg[:] = z_next

        # Pass the solution to the global one
        lvl.u[-1].diff[:] = u.diff[:]
        lvl.u[-1].alg[:] = u.alg[:]

        return None


    def compute_jacobian(self, k, N_diff, N_alg, U, F, epsilon=1e-11):
        lvl = self.level

        # Shortcuts for number of variables and number of collocation nodes/stages
        N = N_diff + N_alg
        M = self.coll.num_nodes

        # Initialize Jacobian and part of it
        J = np.zeros((N, M * N))
        J_part = np.zeros((N, N))

        # Initialize vector for perturbed function values
        F_perturbed = np.zeros(M * N)

        # Evaluate G at the current state
        G0 = self.evaluate_G(k, N_diff, N_alg, U, F)

        # Compute the Jacobian using finite differences
        for j in range(M):
            # Copy of original U
            U_perturbed_entire = U.copy()

            # Copy of piece in U
            U_perturbed = U_perturbed_entire[j * N:(j + 1) * N].copy()

            for i in range(N):
                # Perturb i-th component
                U_perturbed[i] += epsilon

                # Insert perturbed piece of U back to entire U
                U_perturbed_entire[j * N:(j + 1) * N] = U_perturbed

                # Evaluate f for perturbed U 
                for m in range(M):
                    # Store current index of stage
                    k_F = m

                    # Evaluate f
                    F_perturbed[m * N:(m + 1) * N] = self.evaluate_F(k_F, N_diff, N_alg, U_perturbed.reshape(lvl.u[0].shape).view(type(lvl.u[0])))

                # Get the perturbed function value of G
                G_perturbed = self.evaluate_G(k, N_diff, N_alg, U_perturbed_entire, F_perturbed)

                # Compute part of Jacobian
                J_part[:, i] = (G_perturbed - G0) / epsilon

            J[:, j * N:(j + 1) * N] = J_part

        return J

    def evaluate_F(self, k, N_diff, N_alg, U):
        """
        Evaluate the function values of U.

        Parameters:
        - k: Index of current stage.
        - N_diff: Dimension of y.
        - N_alg: Dimension of z.
        - U: Current value of U = (y, z) (shape: M * N).

        Returns:
        - F: Vector of function values.
        """
        lvl = self.level
        prob = lvl.prob

        N = N_diff + N_alg

        F = np.zeros(N)

        F[:] = prob.eval_f(U, lvl.time + lvl.dt * self.coll.nodes[k + 1]).flatten()

        return F


    def evaluate_G(self, k, N_diff, N_alg, U, F):
        """
        Evaluate the function G(U) for the residuals.

        Parameters:
        - i: Index of current stage.
        - N_diff: Dimension of y.
        - N_alg: Dimension of z.
        - U: Current value of U = (y, z) (shape: M * N).
        - Function values of U (shape: M * N).

        Returns:
        - G: Residual vector.
        """
        lvl = self.level

        N = N_diff + N_alg
        M = self.coll.num_nodes

        G = np.zeros(N)

        # Differential equations
        G[:N_diff] = U[k * N:k * N + N_diff] - lvl.u[0].diff.flatten()
        G[:N_diff] -= lvl.dt * sum(self.QI[k + 1, j + 1] * F[j * N:j * N + N_diff] for j in range(M))

        # Algebraic part
        G[N_diff:] = F[k * N + N_diff:(k + 1) * N]

        return G


    def compute_combined_jacobian(self, dt, N_diff, N_alg, U, factor, t, epsilon=1e-8):
        """
        Compute the Jacobian matrix of f with respect to [y, z] using finite differences.

        Parameters:
        - dt: Time step size.
        - N_diff: Number of differential variables.
        - N_alg: Number of algebraic variables.
        - U: Current U value (as numpy array).
        - factor: Weighting factor for the differential part.

        Returns:
        - Combined Jacobian matrix.
        """

        lvl = self.level
        prob = lvl.prob

        jacobian = np.zeros((N_diff + N_alg, N_diff + N_alg))

        # Partial derivatives w.r.t. y
        for i in range(N_diff):
            # Get copy of differential variable y
            y_perturbed = U[:N_diff].copy()

            # Apply a perturbation on it
            y_perturbed[i] += epsilon

            # Get a copy of entire U
            U_perturbed = U.copy()

            # Add the perturbation to it
            U_perturbed[:N_diff] = y_perturbed

            # Get the perturbed function value
            f_perturbed = prob.eval_f(U_perturbed.reshape(lvl.u[0].shape).view(type(lvl.u[0])), t).flatten()

            # Evaluate f at U
            f = prob.eval_f(U.reshape(lvl.u[0].shape).view(type(lvl.u[0])), t).flatten()

            # Compute Jacobian part
            df_dy = (f_perturbed - f) / epsilon

            jacobian[:N_diff, i] = (1 - dt * factor) * df_dy[:N_diff]

        # Partial derivatives w.r.t. z
        for i in range(N_alg):
            # Get copy of algebraic variable z
            z_perturbed = U[N_diff:].copy()

            # Add a perturbation on it
            z_perturbed[i] += epsilon

            # Get a copy of entire U
            U_perturbed = U.copy()

            # Add the perturbation to it
            U_perturbed[N_diff:] = z_perturbed

            # Get the perturbed function value
            f_perturbed = prob.eval_f(U_perturbed.reshape(lvl.u[0].shape).view(type(lvl.u[0])), t).flatten()

            # Evaluate f at U
            f = prob.eval_f(U.reshape(lvl.u[0].shape).view(type(lvl.u[0])), t).flatten()

            # Compute Jacobian part
            df_dz = (f_perturbed - f) / epsilon

            jacobian[:N_diff, N_diff + i] = -dt * factor * df_dz[:N_diff]

        # Algebraic constraints directly influence their own rows
        for i in range(N_alg):
            # Get copy of algebraic variable z
            z_perturbed = U[N_diff:].copy()

            # Add a perturbation on it
            z_perturbed[i] += epsilon

            # Get a copy of entire U
            U_perturbed = U.copy()

            # Add the perturbation to it
            U_perturbed[N_diff:] = z_perturbed
            # Get the perturbed function value
            f_perturbed = prob.eval_f(U_perturbed.reshape(lvl.u[0].shape).view(type(lvl.u[0])), t).flatten()

            # Evaluate f at U
            f = prob.eval_f(U.reshape(lvl.u[0].shape).view(type(lvl.u[0])), t).flatten()

            # Compute Jacobian part
            df_dz = (f_perturbed - f) / epsilon

            jacobian[N_diff:, N_diff + i] = df_dz[N_diff:]

        return jacobian


class RadauIIA5(RungeKuttaDense):
    """Method of Radau IIa family of order 5."""
    generator = Q_GENERATORS["Collocation"](
        nNodes=3, nodeType="LEGENDRE", quadType="RADAU-RIGHT", tLeft=0, tRight=1
    )

    nodes = generator.nodes.copy()
    weights = generator.weights.copy()
    matrix = generator.Q
    ButcherTableauClass = ButcherTableau


class RadauIIA7(RungeKuttaDense):
    """Method of Radau IIa family of order 7."""
    generator = Q_GENERATORS["Collocation"](
        nNodes=4, nodeType="LEGENDRE", quadType="RADAU-RIGHT", tLeft=0, tRight=1
    )

    nodes = generator.nodes.copy()
    weights = generator.weights.copy()
    matrix = generator.Q
    ButcherTableauClass = ButcherTableau


class RadauIIA5Constrained(RungeKuttaDenseConstrained, RadauIIA5):
    pass


class RadauIIA7Constrained(RungeKuttaDenseConstrained, RadauIIA7):
    pass