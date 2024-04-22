import numpy as np
np.set_printoptions(precision=30)
from scipy.optimize import root

import matplotlib.pyplot as plt
from BE_DAE import eval_f, u_exact, du_exact


def main():
    file_name = './data/own.txt'
    file = open(file_name, 'w')
    file.close()

    t0 = 3.0
    Tend = 3.5
    dt = 0.1
    N = int((Tend - t0) / dt)
    print(N)

    u = np.zeros((2, N + 2))
    f = np.zeros((8, N + 2))
    print(u.shape)
    u[:, 0] = u_exact(t0)
    f[: 2, 0] = du_exact(t0)

    t = t0
    i = 0
    # for i in range(N):
    while round(t, 10) <= Tend:
        print(i, round(t, 10))
        file = open(file_name, 'a')
        file.write(f"t={round(t, 5)}: Predict u: %s \n" % u[:, i])
        file.write(f"t={round(t, 5)}: Predict f: {f[: 2, i]} \n")
        file.close()

        u_init = u[:, i]
        # m = 0
        u_approx = u_init#u[:, i]
        file = open(file_name, 'a')
        file.write(f'u_approx at time {round(t + 0 * dt, 2)}: {u_approx}\n')
        file.close()

        def implSystem(unknowns):
            unknowns_mesh = np.zeros(2)
            unknowns_mesh[:] = unknowns

            local_u_approx = np.zeros(2)
            local_u_approx[:] = u_approx
            local_u_approx[:] += dt * 0.0 * unknowns_mesh[:]
            sys = eval_f(local_u_approx, unknowns_mesh, t + 0 * dt)
            file = open(file_name, 'a')
            # file.write(f'Sys at time {round(t + 0 * dt, 5)}: {sys}\n')
            file.close()
            return sys

        file = open(file_name, 'a')
        file.write(f'Initial guess at time {round(t, 5)}: {f[: 2, i]}\n')
        file.close()

        opt = root(
            implSystem,
            f[: 2, i],
            method='hybr',
            tol=1e-14,
        )
        file = open(file_name, 'a')
        file.write(f'Solution at time {round(t + 0 * dt, 5)}: {opt.x}\n')
        file.close()

        f[2 : 4, i] = opt.x

        # m = 1
        # u_approx2 = u_approx.copy()
        # u_approx2 += dt * 0.5 * f[2 : 4, i]
        u_approx2 = u_init.copy()
        u_approx2 += dt * 0.5 * f[2 : 4, i]
        file = open(file_name, 'a')
        file.write(f'u_approx at time {round(t + 0 * dt, 5)}: {u_approx2}\n')
        file.close()

        def implSystem2(unknowns2):
            unknowns_mesh2 = np.zeros(2)
            unknowns_mesh2[:] = unknowns2

            local_u_approx2 = np.zeros(2)
            local_u_approx2[:] = u_approx2
            local_u_approx2[:] += dt * 0.5 * unknowns_mesh2[:]
            sys = eval_f(local_u_approx2, unknowns_mesh2, t + 1 * dt)
            file = open(file_name, 'a')
            # file.write(f'Sys at time {round(t + 1 * dt, 5)}: {sys}\n')
            file.close()
            return sys

        file = open(file_name, 'a')
        file.write(f'Initial guess at time {round(t + 1 * dt, 5)}: {f[2 : 4, i]}\n')
        file.close()

        opt2 = root(
            implSystem2,
            f[2 : 4, i],
            method='hybr',
            tol=1e-14,
        )
        file = open(file_name, 'a')
        file.write(f'Solution at time {round(t + 1 * dt, 5)}: {opt2.x}\n')
        file.close()

        f[4 : 6, i] = opt2.x

        # m = 2
        u_approx3 = u_init.copy()
        u_approx3 += dt * 0.5 * f[2 : 4, i]
        u_approx3 += dt * 0.5 * f[4 : 6, i]
        # u_approx3 = u_approx2.copy()
        # u_approx3 += dt * 0 * f[4 : 6, i]

        file = open(file_name, 'a')
        file.write(f'u_approx at time {round(t + 1 * dt, 5)}: {u_approx3}\n')
        file.close()

        def implSystem3(unknowns3):
            unknowns_mesh3 = np.zeros(2)
            unknowns_mesh3[:] = unknowns3

            local_u_approx3 = np.zeros(2)
            local_u_approx3[:] = u_approx3
            local_u_approx3[:] += dt * 0.0 * unknowns_mesh3[:]
            
            sys = eval_f(local_u_approx3, unknowns_mesh3, t + 1 * dt)
            file = open(file_name, 'a')
            # file.write(f'local_u_approx={local_u_approx3}\n')
            # file.write(f'unknowns_mesh={unknowns_mesh3}\n')
            # file.write(f'Sys at time {round(t + 1 * dt, 5)}: {sys}\n')
            file.close()
            return sys

        file = open(file_name, 'a')
        file.write(f'Initial guess at time {round(t + 1 * dt, 5)}: {f[4 : 6, i]}\n')
        file.close()

        opt3 = root(
            implSystem3,
            f[4 : 6, i],
            method='hybr',
            tol=1e-14,
        )
        file = open(file_name, 'a')
        file.write(f'Solution at time {round(t + 1 * dt, 5)}: {opt3.x}\n')
        file.close()

        f[6 :, i] = opt3.x

        file = open(file_name, 'a')
        # file.write(f'Solution f (complete) at time {round(t + dt, 5)}: {f[:, i]}\n')
        file.close()
        # print(u[:, i] + dt * 0.5 * f[2 : 4, i] + dt * 0.5 * f[4 : 6, i] + dt * 0.0 * f[6 :, i])
        if i <= N:
            f[: 2, i + 1] = opt2.x

            # collocation update
            u[:, i + 1] = u[:, i] + dt * 0.5 * f[2 : 4, i] + dt * 0.5 * f[4 : 6, i] + dt * 0.0 * f[6 :, i]
            # print('if:', u)
            file = open(file_name, 'a')
            # file.write(f'Solution u at time {round(t + dt, 2)}: {u[:, i + 1]}\n')
            file.write(f'\n')
            file.close()
            # print(f'Solution u at time {round(t + dt, 2)}: {u[:, i + 1]}')
            print()
        t += dt
        i += 1
    times = [t0 + i * dt for i in range(N + 2)]
    file = open(file_name, 'a')
    # file.write(f'Solution u over time domain: \n')
    file.write(f'{u[0, 1:]}\n')
    file.write(f'{u[1, 1:]}\n')
    file.close()
    # print(u)
    # print(2 * u[0, -1] - 100)
    # plt.figure()
    # plt.plot(times, u[0, :], label='y')
    # plt.plot(times, u[1, :], label='z')
    # plt.legend(loc='best')
    # plt.savefig('data/TR_DAE_sol.png', dpi=300, bbox_inches='tight')
    # plt.close()


if __name__ == "__main__":
    main()
