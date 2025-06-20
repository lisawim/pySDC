import numpy as np
import matplotlib.pyplot as plt

def u_func(x, t, A, x_deriv=0, t_deriv=0):
    if x_deriv == 0 and t_deriv == 0:
        return A * np.sin(np.pi * x) * np.exp(t)
    elif x_deriv == 1 and t_deriv == 0:
        return A * np.pi * np.cos(np.pi * x) * np.exp(t)
    elif x_deriv == 2 and t_deriv == 0:
        return -A * np.pi ** 2 * np.sin(np.pi * x) * np.exp(t)
    elif x_deriv == 0 and t_deriv == 1:
        return A * np.sin(np.pi * x) * np.exp(t)

def v_func(x, t, B, x_deriv=0, t_deriv=0):
    if x_deriv == 0 and t_deriv == 0:
        return B * np.sin(np.pi * x) * np.exp(t)
    elif x_deriv == 1 and t_deriv == 0:
        return B * np.pi * np.cos(np.pi * x) * np.exp(t)
    elif x_deriv == 2 and t_deriv == 0:
        return -B * np.pi ** 2 * np.sin(np.pi * x) * np.exp(t)
    elif x_deriv == 0 and t_deriv == 1:
        return B * np.sin(np.pi * x) * np.exp(t)

def w_func(x, t, A, B, x_deriv=0):
    if x_deriv == 0:
        return (A + B) / (np.pi ** 2) * np.sin(np.pi * x) * np.exp(t)
    elif x_deriv == 1:
        return (A + B) / np.pi * np.cos(np.pi * x) * np.exp(t)
    elif x_deriv == 2:
        return -(A + B) * np.sin(np.pi * x) * np.exp(t)

def f_source(x, t, A, B):
    u = u_func(x, t, A, x_deriv=0, t_deriv=0)
    u_t = u_func(x, t, A, x_deriv=0, t_deriv=1)
    u_xx = u_func(x, t, A, x_deriv=2, t_deriv=0)
    w_x = w_func(x, t, A, B, x_deriv=1)

    return u_t - u_xx - (u * w_x)

def g_source(x, t, A, B):
    v = v_func(x, t, B, x_deriv=0, t_deriv=0)
    v_t = v_func(x, t, B, x_deriv=0, t_deriv=1)
    v_xx = v_func(x, t, B, x_deriv=2, t_deriv=0)
    w_x = w_func(x, t, A, B, x_deriv=1)

    return v_t - v_xx + (v * w_x)


if __name__ == "__main__":
    left_boundary = 0.0
    L = 1.0
    n_x = 256
    dx = L / (n_x + 1)
    xvalues = np.array([left_boundary + dx * (i + 1) for i in range(n_x)])

    A = -1.0
    B = A#-0.5

    t = 1.0

    print(w_func(0, t, A, B, x_deriv=0))

    plt.figure("f source")
    plt.title(f"{t=}")
    plt.plot(xvalues, f_source(xvalues, t, A, B))
    plt.xlabel("x")
    plt.ylabel("f(x, t)")
    plt.show()

    plt.figure("g source")
    plt.title(f"{t=}")
    plt.plot(xvalues, g_source(xvalues, t, A, B))
    plt.xlabel("x")
    plt.ylabel("g(x, t)")
    plt.show()

    plt.figure("u_x and v_x")
    plt.title(f"{t=}")
    plt.plot(xvalues, u_func(xvalues, t, A, x_deriv=1, t_deriv=0), label="u_x")
    plt.plot(xvalues, v_func(xvalues, t, B, x_deriv=1, t_deriv=0), linestyle="dotted", label="v_x")
    plt.legend(loc="upper right")
    plt.show()
