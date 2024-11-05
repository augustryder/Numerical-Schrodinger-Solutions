import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import matplotlib.pyplot as plt


a = 1 # Gaussian stdev
# Gaussian potential
def V(x):
    V0 = 1
    return -V0 * np.exp(-x**2 / (2 * a**2))

x_m = 4 * a # Boundry 

# Sine basis functions
def B(n, x):
    return np.sin(n * np.pi * x / 4)

def dB(n, x):
    return n * np.pi / 4 * np.cos(n * np.pi * x / 4)

# Number of basis functions
N = 50

# discrete x-axis
x = np.linspace(0, x_m, 2**10 + 1)

# Initialize hamiltonian and S matrices
T = np.zeros((N, N))
A = np.zeros((N, N))
E = 2

for i in range(N):
    for j in range(N):
        # calculate Tij
        integrand1 = -dB(i + 1, x) * dB(j + 1, x) + 2 * B(i + 1, x) * (E - V(x)) * B(j + 1, x)
        T[i, j] = integrate.simpson(integrand1, x=x)
            
        # calculate Aij
        A[i, j] = B(i + 1, x_m) * B(j + 1, x_m)

eigvals, eigvecs = la.eig(T, A)
eigvals = eigvals.real
eigvecs = eigvecs.real

b = eigvals[0]
c = eigvecs[:, 0]

print("Negative Log Derivative (b): ", b)

# calculate and plot wavefucntions
def psi(x, c):
    return sum(c[k] * B(k + 1, x) for k in range(len(c)))

inner_psi = psi(x, c)

x_outer = np.linspace(x_m, 2*x_m, 2**10)
m = 1
hbar = 1
k = np.sqrt(2 * m * E) / hbar
even_shift = np.atan(-k / b) - k * x_m
outer_psi = 1 * np.sin([k*x0 + even_shift for x0 in x_outer])

plt.figure(figsize=(9, 9))
plt.plot(x, inner_psi, label='ψ(x)', linewidth=1.5)
plt.plot(x_outer, outer_psi, label='ψ(x)', linewidth=1.5)

fullx = np.linspace(-1, 2*x_m, 2**10)
# plot potential
plt.plot(fullx, V(fullx), label="Gaussian Potential", color="black", linewidth=2)

plt.axhline(0, color='black', linewidth=1)  
plt.title("Wavefunctions and Energy Levels")
plt.xlabel("Position x")
plt.ylabel("Wavefunction")
# plt.legend()
plt.show()





