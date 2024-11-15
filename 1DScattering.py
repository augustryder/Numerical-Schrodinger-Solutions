import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

m = 1
hbar = 1

a = 1 # Gaussian stdev
# Gaussian potential
def V(x):
    V0 = 1
    return -V0 * np.exp(-x**2 / (2 * a**2))

x_m = 4 * a # Boundry 

# Number of basis functions
N = 30
# Order
k = 5
# Knot set
t = []
for _ in range(k):
    t.append(0)
for i in range(N - k + 1):
    t.append(i)
for _ in range(k):
    t.append(N - k)
    
# Coefficient array
c = [0 for _ in range(N)]

# discrete x-axis
x_inner = np.linspace(0, x_m, 2**10 + 1)
x_outer = np.linspace(x_m, 2*x_m, 2**10 + 1)
x_full = np.linspace(-2, 2*x_m, 2**10 + 1)

plt.figure(figsize=(9, 9))

# x array, order, id, knot set, coeff arr
def B(x, k, i, t, c):
   c[i] = 1
   spl = BSpline(t, c, k)
   c[i] = 0
   return spl(x)

def dB(x, k, i, t, c):
   c[i] = 1
   spl = BSpline(t, c, k).derivative()
   c[i] = 0
   return spl(x)

# Initialize hamiltonian and S matrices
T = np.zeros((N, N))
A = np.zeros((N, N))
E = 1

for i in range(N):
    for j in range(N):
        # calculate Tij
        integrand1 = -dB(x_inner, k, i, t, c) * dB(x_inner, k, j, t, c) + 2 * B(x_inner, k, i, t, c) * (E - V(x_inner)) * B(x_inner, k, j, t, c)
        T[i, j] = integrate.simpson(integrand1, x=x_inner)
            
        # calculate Aij
        A[i, j] = B(x_m, k, i, t, c) * B(x_m, k, j, t, c)


eigvals, eigvecs = la.eig(T, A)
eigvals = eigvals.real
eigvecs = eigvecs.real

b = eigvals[1]
cs = eigvecs[:, 1]

print("Log Derivative (-b): ", -b)

# calculate wavefucntion from bspline
def psi(x, c):
    spl = BSpline(t, c, k)
    return spl(x)

inner_psi = psi(x_inner, cs)

k = np.sqrt(2 * m * E) / hbar
even_shift = np.arctan(b / k) - (k * x_m)
A = inner_psi[-1] / np.cos(k * x_m + even_shift)
outer_psi = A * np.cos([k * x + even_shift for x in x_outer])

plt.figure(figsize=(9, 9))
plt.plot(x_inner, inner_psi, label='ψ(x)', linewidth=1.5)
plt.plot(x_outer, outer_psi, label='ψ(x)', linewidth=1.5)
plt.plot(x_full, V(x_full), label="Gaussian Potential", color="black", linewidth=2)

plt.axhline(0, color='black', linewidth=1)  
plt.title("1D Scattering Wavefunction")
plt.xlabel("Position x")
plt.ylabel("Wavefunction")
# plt.legend()
plt.show()





