import numpy as np
import scipy.integrate as integrate
from numpy.linalg import inv
import matplotlib.pyplot as plt

from qmat import Q_GENERATORS, QDELTA_GENERATORS

# Exact solutions
def p(t, alpha, lambda_):
    return alpha * np.exp(lambda_ * t)

def y_exact(t, alpha, lambda_, mu):
    return np.exp(lambda_ * t) * (1.0 + mu * alpha * t)

def predict(M, u0):
    u_const, u_embed = np.zeros((2, M)), np.zeros((2, M))
    for m in range(M):
        u_const[:, m] = u0[:]
        u_embed[:, m] = u0[:]

    # Testing
    for m in range(M):
        assert np.allclose(u_const[:, m], u0, atol=1e-14)
        assert np.allclose(u_embed[:, m], u0, atol=1e-14)

    return u_const, u_embed

def eval_rhs(y, z, t, alpha, lambda_, mu):
    eq1 = lambda_ * y + mu * z
    eq2 = -z + (alpha * y) / (1 + mu * alpha * t)
    rhs = np.array([eq1, eq2])
    return rhs

def integrate_function(coll_nodes, dt, M, Qmat, u_const, u_embed, alpha, lambda_, mu):
    integral_const = np.zeros((2, M))
    integral_embed = np.zeros((2, M))

    for m in range(M):
        for j in range(M):
            f_const = eval_rhs(u_const[0, j], u_const[1, j], coll_nodes[j], alpha, lambda_, mu)
            f_embed = eval_rhs(u_embed[0, j], u_embed[1, j], coll_nodes[j], alpha, lambda_, mu)

            integral_const[0, m] += dt * Qmat[m, j] * f_const[0]
            integral_embed[:, m] += dt * Qmat[m, j] * f_embed

    return integral_const, integral_embed

def test_integrate(coll_nodes, dt, integral_const, integral_embed, M, Qmat, u_const, u_embed, alpha, lambda_, mu):
    f1_const = np.array([eval_rhs(u_const[0, m], u_const[1, m], coll_nodes[m], alpha, lambda_, mu)[0] for m in range(M)])
    assert np.allclose(integral_const[0, :], dt * Qmat.dot(f1_const), atol=1e-14)
    assert np.allclose(integral_const[1, :], np.zeros(M), atol=1e-14)

    f1_embed = np.array([eval_rhs(u_embed[0, m], u_embed[1, m], coll_nodes[m], alpha, lambda_, mu)[0] for m in range(M)])
    f2_embed = np.array([eval_rhs(u_embed[0, m], u_embed[1, m], coll_nodes[m], alpha, lambda_, mu)[1] for m in range(M)])
    assert np.allclose(integral_embed[0, :], dt * Qmat.dot(f1_embed), atol=1e-14)
    assert np.allclose(integral_embed[1, :], dt * Qmat.dot(f2_embed), atol=1e-14)

def solve_system_const(rhs, factor, u0, t, alpha, lambda_, mu, newton_tol=1e-12, newton_maxiter=15):
    u = u0.copy()

    # Start newton iteration
    n = 0
    res = 99
    while n < newton_maxiter:
        y, z = u[0], u[1]

        g = np.array([y - factor * (lambda_ * y + mu * z) - rhs[0], -z + (alpha * y) / (1 + mu * alpha * t)])

        # If g is close to 0, then we are done
        res = np.linalg.norm(g, np.inf)
        if res < newton_tol:
            break

        # Inverse of dg
        dg = np.array([[1 - factor * lambda_, -factor * mu], [alpha / (1 + mu * alpha * t), -1]])

        # newton update: u1 = u0 - g/dg
        u -= np.linalg.solve(dg, g)

        n += 1
    return u

def solve_system_embed(rhs, factor, u0, t, alpha, lambda_, mu, newton_tol=1e-12, newton_maxiter=15):
    u = u0.copy()

    # Start newton iteration
    n = 0
    res = 99
    while n < newton_maxiter:
        # Form the function g(u), such that the solution to the nonlinear problem is a root of g
        f = eval_rhs(u[0], u[1], t, alpha, lambda_, mu)

        g = np.array([u[0] - factor * f[0] - rhs[0], -factor * f[1] - rhs[1]])

        # If g is close to 0, then we are done
        res = np.linalg.norm(g, np.inf)
        if res < newton_tol:
            break

        # Inverse of dg
        dg = np.array([[1 - factor * lambda_, -factor * mu], [(-factor * alpha) / (1 + mu * alpha * t), factor]])

        # newton update: u1 = u0 - g/dg
        u -= np.linalg.solve(dg, g)

        n += 1
    return u


if __name__ == "__main__":
    alpha = 100.0
    lambda_ = 10.0
    mu = 1.0

    # Number of collocation nodes
    M = 3
    p_ord = 2 * M - 1
    quad_type = "RADAU-RIGHT"
    QI = "IE"

    QGenerator = Q_GENERATORS["coll"]
    coll = QGenerator(nNodes=M, nodeType="LEGENDRE", quadType=quad_type)
    Qmat = coll.Q
    nodes = coll.nodes
    weights = coll.weights

    QIGenerator = QDELTA_GENERATORS[QI]

    approx = QIGenerator(Q=Qmat, nNodes=M, nodeType="LEGENDRE", quadType=quad_type, nodes=nodes)

    QImat = approx.getQDelta()

    # Predict
    t0 = 0.0

    y0 = y_exact(t0, alpha, lambda_, mu)
    z0 = p(t0, alpha, lambda_)

    u0 = np.array([y0, z0])

    # Time settings
    dt_values = np.logspace(-3.0, -1.0, num=11)

    # Containers for errors
    errors_y_const = []
    errors_z_const = []

    errors_y_embed = []
    errors_z_embed = []

    errors_manifold_const = []
    errors_manifold_embed = []

    # Test for each step size
    maxiter = 5 #2 * M - 1
    for dt in dt_values:

        # Predict
        u_const, u_embed = predict(M, u0)

        t = t0

        # Collocation nodes
        coll_nodes = [t + dt * nodes[m] for m in range(M)]

        k = 0
        while k < maxiter:
            u_const_new, u_embed_new = np.zeros_like(u_const), np.zeros_like(u_embed)

            # Integrate
            integral_const, integral_embed = integrate_function(coll_nodes, dt, M, Qmat, u_const, u_embed, alpha, lambda_, mu)

            test_integrate(coll_nodes, dt, integral_const, integral_embed, M, Qmat, u_const, u_embed, alpha, lambda_, mu)

            for m in range(M):
                for j in range(M):
                    f_const = eval_rhs(u_const[0, j], u_const[1, j], coll_nodes[j], alpha, lambda_, mu)
                    f_embed = eval_rhs(u_embed[0, j], u_embed[1, j], coll_nodes[j], alpha, lambda_, mu)

                    integral_const[0, m] -= dt * QImat[m, j] * f_const[0]
                    integral_embed[:, m] -= dt * QImat[m, j] * f_embed

                integral_const[0, m] += y0
                integral_embed[0, m] += y0

            for m in range(M):
                b_const = integral_const[:, m].copy()
                b_embed = integral_embed[:, m].copy()

                for j in range(0, m):
                    f_const = eval_rhs(u_const_new[0, j], u_const_new[1, j], coll_nodes[j], alpha, lambda_, mu)
                    f_embed = eval_rhs(u_embed_new[0, j], u_embed_new[1, j], coll_nodes[j], alpha, lambda_, mu)

                    b_const[0] += dt * QImat[m, j] * f_const[0]
                    b_embed[:] += dt * QImat[m, j] * f_embed

                factor = dt * QImat[m, m]
                u_const_new[:, m] = solve_system_const(b_const, factor, u_const[:, m], coll_nodes[m], alpha, lambda_, mu)
                u_embed_new[:, m] = solve_system_embed(b_embed, factor, u_embed[:, m], coll_nodes[m], alpha, lambda_, mu)

            u_const[:] = u_const_new[:]
            u_embed[:] = u_embed_new[:]

            k += 1
        print(k)
        # Exact values at final time
        y_ex = y_exact(t + dt, alpha, lambda_, mu)
        z_ex = p(t + dt, alpha, lambda_)
        u_ex = np.array([y_ex, z_ex])

        manifold_ex = eval_rhs(y_ex, z_ex, t + dt, alpha, lambda_, mu)[1]

        # Store errors
        errors_y_const.append(abs(max([u_const[0, m] - y_ex for m in range(M)])))
        errors_z_const.append(abs(max([u_const[1, m] - z_ex for m in range(M)])))

        errors_y_embed.append(abs(max([u_embed[0, m] - y_ex for m in range(M)])))
        errors_z_embed.append(abs(max([u_embed[1, m] - z_ex for m in range(M)])))

        manifold_const = abs(max([eval_rhs(u_const[0, m], u_const[1, m], coll_nodes[m], alpha, lambda_, mu)[1] for m in range(M)]))
        manifold_embed = abs(max([eval_rhs(u_embed[0, m], u_embed[1, m], coll_nodes[m], alpha, lambda_, mu)[1] for m in range(M)]))

        errors_manifold_const.append(manifold_const)
        errors_manifold_embed.append(manifold_embed)

    # Plotting
    plt.figure()
    ord = k + 1 if k + 1 < 2 * M - 1 else 2 * M - 1
    ord_z = 1
    ref_ord = [1e2 * dt ** ord for dt in dt_values]
    ref_ord_1 = [1e4 * dt ** ord_z for dt in dt_values]
    plt.loglog(dt_values, errors_y_const, 'o', linestyle="solid", label='y error (constrained SDC)')
    plt.loglog(dt_values, errors_z_const, '*', linestyle="dashdot", label='z error (constrained SDC)')
    plt.loglog(dt_values, errors_y_embed, 's', linestyle="dashed", label='y error (embedded SDC)')
    plt.loglog(dt_values, errors_z_embed, 'H', linestyle="dotted", label='z error (embedded SDC)')
    plt.loglog(dt_values, ref_ord, linestyle="dashed", color="lightgrey", label=f'ref order {ord}')
    plt.loglog(dt_values, ref_ord_1, linestyle="dashed", color="darkgrey", label=f'ref order {ord_z}')
    plt.ylim((1e-15, 1e7))
    plt.xlabel('Time step size (h)')
    plt.ylabel('Error after one time step')
    plt.title('Error: Constrained SDC vs Embedded SDC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("order.png")
    plt.show()

    plt.figure()
    plt.loglog(dt_values, errors_manifold_const, 'o', linestyle="solid", label='manifold value (constrained SDC)')
    plt.loglog(dt_values, errors_manifold_embed, 'D', linestyle="solid", label='manifold value (embedded SDC)')
    plt.ylim((1e-15, 1e7))
    plt.xlabel('Time step size (h)')
    plt.ylabel('Manifold value after one time step')
    plt.title('Manifold: Constrained SDC vs Embedded SDC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("order.png")
    plt.show()
