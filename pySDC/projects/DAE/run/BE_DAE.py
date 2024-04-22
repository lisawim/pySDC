import numpy as np
from scipy.optimize import root

import matplotlib.pyplot as plt

def eval_f(u, du, t):
    file_name = './data/own.txt'
    f = np.zeros(2)
    y, z = u[0], u[1]
    dy = du[0]

    t_switch = np.inf

    h = 2 * y - 100
    if h >= 0 or t >= t_switch:
        f[0] = dy
        f[1] = y**2 - z**2 - 1
    else:
        f[0] = dy - z
        f[1] = y**2 - z**2 - 1
    file = open(file_name, 'a')
    # file.write(f'In eval_f at time {round(t, 5)}: {f}\n')
    file.close()
    return f

def u_exact(t):
    assert t >= 1, 'ERROR: u_exact only available for t>=1'
    t_switch_exact = np.arccosh(50)

    me = np.zeros(2)
    if t <= t_switch_exact:
        me[0] = np.cosh(t)
        me[1] = np.sinh(t)
    else:
        me[0] = np.cosh(t_switch_exact)
        me[1] = np.sinh(t_switch_exact)
    return me

def du_exact(t):
    assert t >= 1, 'ERROR: u_exact only available for t>=1'
    t_switch_exact = np.arccosh(50)

    me = np.zeros(2)
    if t <= t_switch_exact:
        me[0] = np.sinh(t)
        me[1] = np.cosh(t)
    else:
        me[0] = np.sinh(t_switch_exact)
        me[1] = np.cosh(t_switch_exact)
    return me

def main():
    t0 = 1.0
    Tend = 5.0
    dt = 0.1
    N = int((Tend - t0) / dt)
    print(N)

    u = np.zeros((2, N + 1))
    f = np.zeros((2, N + 1))

    u[:, 0] = u_exact(t0)
    f[:, 0] = du_exact(t0)

    t = t0

    for i in range(N):
        print(t, 'Predict u:', u[:, i])
        print(t, 'Predict f:', f[:, i])
        u_approx = u[:, i]

        def implSystem(unknowns):
            unknowns_mesh = np.zeros(2)
            unknowns_mesh[:] = unknowns

            local_u_approx = np.zeros(2)
            local_u_approx[:] = u_approx
            local_u_approx[:] += dt * 1 * unknowns_mesh[:]
            sys = eval_f(local_u_approx, unknowns_mesh, t + dt)
            return sys
        print(t, 'Initial guess:', f[:, i])
        print()
        opt = root(
            implSystem,
            f[:, i],
            method='hybr',
            tol=1e-12,
        )

        f[:, i + 1] = opt.x

        # collocation update
        u[:, i + 1] = u[:, i] + dt * 1 * f[:, i + 1]

        t += dt
    times = [t0 + i * dt for i in range(N + 1)]
    plt.figure()
    plt.plot(times, u[0, :], label='y')
    plt.plot(times, u[1, :], label='z')
    plt.legend(loc='best')
    plt.savefig('data/BE_DAE_sol.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(u)


if __name__ == "__main__":
    main()

            

