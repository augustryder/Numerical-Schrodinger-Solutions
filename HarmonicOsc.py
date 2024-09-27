import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Infinite square well solution bases
a = 1.0
def B(n, x):
    return np.sqrt(2 / a) * np.sin(n * np.pi * x / a)

def d2B(n, x):
    return -1 * (n * np.pi / a)**2 * B(n, x)

# Harmonic oscillator constants
m = 1.0
omega = 1.0
hbar = 1.0
N = 50

# Harmonic osc potential
def V(x):
    return 0.5 * m * omega**2 * (x - a / 2)**2

x = np.linspace(0, a, 2**10 + 1)

H = np.zeros((N, N))
S = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        # calculate Tij
        integrand1 = B(i + 1, x) * d2B(j + 1, x)
        H[i, j] += ( -1 * hbar**2 / (2 * m)) * integrate.simpson(integrand1)
        
        # calculate Vij
        integrand2 = B(i + 1, x) * V(x) * B(j + 1, x)
        H[i, j] += integrate.simpson(integrand2)

        # calculate Sij
        integrand3 = B(i + 1, x) * B(j + 1, x)
        S[i, j] = integrate.simpson(integrand3)

eigvals, eigvecs = la.eig(H, S)
eigvals = eigvals.real
eigvecs = eigvecs.real
solutions = [tuple] * len(eigvals)
for i in range(len(eigvals)):
    solutions[i] = (eigvals[i], eigvecs[i])
solutions = sorted(solutions)
print(eigvals)

# Plot the potential and first few wavefunctions
plt.figure(figsize=(9, 9))
plt.plot(x, V(x), label="Harmonic Oscillator Potential", color="black", linewidth=2)

# Plot the first few eigenfunctions
for i in range(6,7):
    (energy, eigv) = solutions[i]
    psi = np.zeros_like(x)
    for j in range(N):
        psi += eigv[j] * B(j + 1, x)
    psi = psi / np.sqrt(integrate.simpson(psi**2))  
    plt.plot(x, psi, label=f'ψ_{i+1}(x)', linewidth=1.5)

plt.axhline(0, color='black', linewidth=1)  
plt.title("Wavefunctions and Energy Levels")
plt.xlabel("Position x")
plt.ylabel("Wavefunction + Energy")
plt.legend()
plt.show()








