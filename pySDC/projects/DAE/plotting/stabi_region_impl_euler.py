import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals, inv

# Gitterparameter
x_min, x_max = -5, 5
y_min, y_max = -5, 5
resolution = 300
h = 1.0  # Zeitschrittgröße

# Gitter in der komplexen Ebene
x = np.linspace(x_min, x_max, resolution)
y = np.linspace(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y  # Komplexes Lambda

# Array für Spektralradien
rho = np.zeros_like(X)

# Schleife über Gitterpunkte
for i in range(resolution):
    for j in range(resolution):
        lam = Z[i, j]
        
        # Definiere A(lambda)
        A = np.array([[lam, 1],
                      [0,   lam]])
        
        # Stabilitätsmatrix für impliziten Euler
        I = np.eye(2)
        R = inv(I - h * A)
        
        # Spektralradius
        eigs = eigvals(R)
        rho[i, j] = np.max(np.abs(eigs))

# Plot Stabilitätsgebiet (wo Spektralradius ≤ 1)
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, rho, levels=[0, 1], colors=["#add8e6"])  # hellblau für stabil

# Stabilitätsrand (rho = 1) als schwarze Linie
plt.contour(X, Y, rho, levels=[1], colors="black")

# Achsen und Gitter
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.xlabel(r'$\mathrm{Re}(\lambda)$')
plt.ylabel(r'$\mathrm{Im}(\lambda)$')
plt.title('Stabilitätsgebiet für $A(\lambda)$ mit implizitem Euler')
plt.grid(True, linestyle=':')
plt.gca().set_aspect('equal')
plt.show()