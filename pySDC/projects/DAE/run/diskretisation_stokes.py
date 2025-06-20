import numpy as np

from pySDC.helpers import problem_helper


nvars = [8, 8]
order = 2

dx, xvalues = problem_helper.get_1d_grid(nvars[0], "periodic")

# Discretisation for Laplacian
A, _ = problem_helper.get_finite_difference_matrix(
    derivative=2,
    order=order,
    stencil_type="center",
    dx=dx,
    size=nvars[0],
    dim=2,
    bc="periodic",
)

print(A.shape)
print(A)
print((dx ** 2) * A)

X, Y = np.meshgrid(xvalues, xvalues)
r2 = X**2 + Y**2
me = np.tanh((0.25 - np.sqrt(r2)) / (np.sqrt(2) * 0.04))
print(me.shape)
print(dx)
print(xvalues)