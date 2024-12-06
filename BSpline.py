import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

# def B(x, k, i, t):
#    if k == 0:
#       return 1.0 if t[i] <= x < t[i+1] else 0.0
#    if t[i+k] == t[i]:
#       c1 = 0.0
#    else:
#       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
#    if t[i+k+1] == t[i+1]:
#       c2 = 0.0
#    else:
#       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
#    return c1 + c2

# Number of basis functions
n = 30 - 1

# Order
k = 5

N = n + k + 1

# Knot set
t = []
for _ in range(k):
    t.append(0)
for i in range(n - k + 1):
    t.append(i)
for _ in range(k):
    t.append(n - k)
   
print(t)

# Coefficient array
c = [0 for _ in range(n+1)]
x = np.linspace(0, n - k, 10000)

# id, x array 
def B(i, x):
   c[i] = 1
   spl = BSpline(t, c, k)
   c[i] = 0
   return spl(x)

y_0 = 0.4
basis = [ y_0 * (B(0, x) + B(1, x)) ] + [ B(i, x) for i in range(2, n+1)]

plt.figure(figsize=(9, 9))
for i in range(n):
    plt.plot(x, basis[i])

plt.title("BSpliney")
plt.xlabel("x")
plt.ylabel("y")
plt.show()








