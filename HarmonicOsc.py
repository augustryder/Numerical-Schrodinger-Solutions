import numpy as np
import scipy.linalg as la
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Infinite square well solution bases
a = 10.0
def B(n, x):
    return np.sqrt(2 / a) * np.sin(n * np.pi * x / a)

def d2B(n, x):
    return -1 * (n * np.pi / a)**2 * B(n, x)

# System constants
m = 1.0
omega = 1.0
hbar = 1.0
# Number of basis functions
N = 50

# Harmonic osc potential
def V(x):
    return 0.5 * m * omega**2 * (x - a / 2)**2

# discrete x-axis
x = np.linspace(0, a, 2**12 + 1)

# Initialize hamiltonian and S matrices
H = np.zeros((N, N))
S = np.zeros((N, N))

def delta(m, n):
    if m == n: return 1
    else: return 0

for i in range(N):
    for j in range(N):
        # calculate Tij
        # integrand1 = B(i + 1, x) * d2B(j + 1, x)
        # H[i, j] += ( -1 * hbar**2 / (2 * m)) * integrate.simpson(integrand1, x=x)
        
        H[i, j] += delta(i + 1, j + 1) * 0.5 * ((j + 1) * np.pi / a)**2
        
        # calculate Vij
        # integrand2 = B(i + 1, x) * V(x) * B(j + 1, x)
        # H[i, j] += integrate.simpson(integrand2, x=x)
        if i == j:
            H[i, j] += (a**2 / 24) * (1 - 6 / ((j + 1) * np.pi)**2)
        else:
            H[i, j] += (2 * a**2 * (i + 1) * (j + 1) * (1 + (-1)**(i + j + 2))) / (((i + 1)**2 - (j + 1)**2)**2 * (np.pi)**2)
            

        # calculate Sij
        integrand3 = B(i + 1, x) * B(j + 1, x)
        S[i, j] = integrate.simpson(integrand3, x=x)

eigvals, eigvecs = la.eig(H, S)
eigvals = eigvals.real
eigvecs = eigvecs.real

# sortings solutions by energy
solutions = [tuple] * len(eigvals)
for i in range(len(eigvals)):
    solutions[i] = (eigvals[i], eigvecs[:, i])
solutions = sorted(solutions)
for (e, v) in solutions: print(e)

# plot potential
plt.figure(figsize=(9, 9))
plt.plot(x, V(x), label="Harmonic Oscillator Potential", color="black", linewidth=2)

# calculate and plot wavefucntions
for i in range(10):
    (energy, eigv) = solutions[i]
    psi = np.zeros_like(x)
    for j in range(N):
        psi += eigv[j] * B(j + 1, x)
    psi = -psi / np.sqrt(integrate.simpson(psi**2, x=x))  
    plt.plot(x, psi + energy, label=f'ψ_{i+1}(x)', linewidth=1.5)

plt.axhline(0, color='black', linewidth=1)  
plt.title("Wavefunctions and Energy Levels")
plt.xlabel("Position x")
plt.ylabel("Wavefunction + Energy (ℏω)")
y_min, y_max = plt.ylim()
y_ticks = np.arange(np.floor(y_min), np.ceil(y_max), 0.5)
plt.yticks(y_ticks)
# plt.legend()
plt.show()








