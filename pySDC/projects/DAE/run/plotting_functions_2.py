import numpy as np
import matplotlib.pyplot as plt

def u_func(x, t, x_deriv=0, t_deriv=0):
    if x_deriv == 0 and t_deriv == 0:
        return 30 * (3 * x ** 2 - 8 * x ** 3 + 5 * x ** 4) * np.exp(t)
    elif x_deriv == 1 and t_deriv == 0:
        return 30 * (x ** 3 - 2 * x ** 4 + x ** 5) * np.exp(t)
    elif x_deriv == 2 and t_deriv == 0:
        return 30 * ((x ** 4) / 4 - (2 * x ** 5) / 5 + (x ** 6) / 6) * np.exp(t)
    elif x_deriv == 0 and t_deriv == 1:
        return 30 * (3 * x ** 2 - 8 * x ** 3 + 5 * x ** 4) * np.exp(t)

def v_func(x, t, x_deriv=0, t_deriv=0):
    if x_deriv == 0 and t_deriv == 0:
        return 30 * (3 * x ** 2 - 8 * x ** 3 + 5 * x ** 4) * np.exp(t)
    elif x_deriv == 1 and t_deriv == 0:
        return 30 * (x ** 3 - 2 * x ** 4 + x ** 5) * np.exp(t)
    elif x_deriv == 2 and t_deriv == 0:
        return 30 * ((x ** 4) / 4 - (2 * x ** 5) / 5 + (x ** 6) / 6) * np.exp(t)
    elif x_deriv == 0 and t_deriv == 1:
        return 30 * (3 * x ** 2 - 8 * x ** 3 + 5 * x ** 4) * np.exp(t)

def w_func(x, t, x_deriv=0):
    if x_deriv == 0:
        return (1 - 60 * ((x ** 4) / 4 - (2 * x ** 5) / 5 + (x ** 6) / 6)) * np.exp(t)
    elif x_deriv == 1:
        return -60 * (x ** 3 - 2 * x ** 4 + x ** 5) * np.exp(t)
    elif x_deriv == 2:
        return -60 * (3 * x ** 2 - 8 * x ** 3 + 5 * x ** 4) * np.exp(t)

def f_source(x, t):
    u = u_func(x, t, x_deriv=0, t_deriv=0)
    u_t = u_func(x, t, x_deriv=0, t_deriv=1)
    u_xx = u_func(x, t, x_deriv=2, t_deriv=0)
    w_x = w_func(x, t, x_deriv=1)

    return u_t - u_xx - (u * w_x)

def g_source(x, t):
    v = v_func(x, t, x_deriv=0, t_deriv=0)
    v_t = v_func(x, t, x_deriv=0, t_deriv=1)
    v_xx = v_func(x, t, x_deriv=2, t_deriv=0)
    w_x = w_func(x, t, x_deriv=1)

    return v_t - v_xx + (v * w_x)


if __name__ == "__main__":
    left_boundary = 0.0
    L = 1.0
    n_x = 256
    dx = L / (n_x + 1)
    xvalues = np.array([left_boundary + dx * (i + 1) for i in range(n_x)])

    t = 0.0

    print(w_func(0, t, x_deriv=0))

    plt.figure("f source")
    plt.title(f"{t=}")
    plt.plot(xvalues, f_source(xvalues, t))
    plt.xlabel("x")
    plt.ylabel("f(x, t)")
    plt.show()

    plt.figure("g source")
    plt.title(f"{t=}")
    plt.plot(xvalues, g_source(xvalues, t))
    plt.xlabel("x")
    plt.ylabel("g(x, t)")
    plt.show()

    plt.figure("u_x and v_x")
    plt.title(f"{t=}")
    plt.plot(xvalues, u_func(xvalues, t, x_deriv=1, t_deriv=0), label="u_x")
    plt.plot(xvalues, v_func(xvalues, t, x_deriv=1, t_deriv=0), linestyle="dotted", label="v_x")
    plt.legend(loc="upper right")
    plt.show()