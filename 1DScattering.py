import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

m = 1
hbar = 1
E = 2

a = 4 # Gaussian stdev
# Gaussian potential
def V(x):
    V0 = 8
    return -V0 * np.exp(-x**2 / (2 * a**2))

x_m = 4 * a # Boundry 

# Order
k = 5

# Number of basis functions
N = 50
assert(N > k)

# number of points on x_grid
xdim = N - k + 1

xl = 0
xr = x_m

x_grid = [xl + (xr - xl) * n / (xdim-1) for n in range(0, xdim)]
print(x_grid)

# beta = -logder 
# B1' = -B2'
# {c1B1 + c2B2, B3, ..., Bxdim +k -1}
# Knot set
t = []
for _ in range(k):
    t.append(x_grid[0])
t += x_grid
for _ in range(k):
    t.append(x_grid[xdim-1])

print(t)
print(len(t))

# Coefficient array
c = [0 for _ in range(N+1)]

# discrete x-axis
x_inner = np.linspace(0, x_m, 2**10 + 1)
x_outer = np.linspace(x_m, 2*x_m, 2**10 + 1)
x_full = np.linspace(-2, 2*x_m, 2**10 + 1)

# x array, order, id, knot set, coeff arr
def B(i, x):
        c[i] = 1
        spl = BSpline(t, c, k)
        c[i] = 0
        return spl(x)

def dB(i, x):
        c[i] = 1
        spl = BSpline(t, c, k).derivative()
        c[i] = 0
        return spl(x)

# y_0 = 0.4
# basis = [ y_0 * (B(0, x_inner) + B(1, x_inner)) ] + [ B(i, x_inner) for i in range(2, N+1)]
# dbasis = [ y_0 * (dB(0, x_inner) + dB(1, x_inner)) ] + [ dB(i, x_inner) for i in range(2, N+1)]

basis = [ B(i, x_inner) for i in range(0, N)]
dbasis = [ dB(i, x_inner) for i in range(0, N)]

plt.figure(figsize=(9, 9))
for i in range(N):
   plt.plot(x_inner, basis[i], alpha=0.1)
   
# Initialize hamiltonian and S matrices
T = np.zeros((N, N))
A = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        # calculate Tij
        integrand1 = -dbasis[i] * dbasis[j] + 2 * basis[i] * (E - V(x_inner)) * basis[j]
        T[i, j] = integrate.simpson(integrand1, x=x_inner)
        # calculate Aij
        A[i, j] = B(i, x_m) * B(j, x_m)

eigvals, eigvecs = la.eig(T, A)
eigvals = eigvals.real
eigvecs = eigvecs.real
print(eigvals)

idx = 0
for i, val in enumerate(eigvals):
    if -float('inf') < val < float('inf'):
        idx = i

b = eigvals[idx]
cs = eigvecs[:, idx]

print("Log Derivative (-b): ", -b)

# calculate wavefunction from bspline
def psi(x, c):
    spl = BSpline(t, c, k)
    return spl(x)

inner_psi = psi(x_inner, cs)

k = np.sqrt(2 * m * E) / hbar
even_shift = np.arctan(b / k) - (k * x_m)
A = inner_psi[-1] / np.cos(k * x_m + even_shift)
outer_psi = A * np.cos([k * x + even_shift for x in x_outer])


plt.plot(x_inner, inner_psi, label='ψ(x)', linewidth=1.5)
plt.plot(x_outer, outer_psi, label='ψ(x)', linewidth=1.5)
plt.plot(x_full, V(x_full), label="Gaussian Potential", color="black", linewidth=2)

plt.gca().set_aspect(10, adjustable='datalim')  # Ratio of y-unit to x-unit
plt.axhline(0, color='black', linewidth=1)  
plt.title("1D Scattering Wavefunction")
plt.xlabel("Position x")
plt.ylabel("Wavefunction")
# plt.legend()
plt.show()





