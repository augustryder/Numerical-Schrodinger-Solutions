import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

def B(x, k, i, t):
   if k == 0:
      return 1.0 if t[i] <= x < t[i+1] else 0.0
   if t[i+k] == t[i]:
      c1 = 0.0
   else:
      c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
   if t[i+k+1] == t[i+1]:
      c2 = 0.0
   else:
      c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
   return c1 + c2

n = 10
k = 3
t = [ i for i in range(n + k + 1) ]
c = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
spl = BSpline(t, c, k)
x = np.linspace(0, 10, 3000)

plt.figure(figsize=(9, 9))
assert (n >= k+1) and (len(c) >= n)
for i in range(n):
    b = np.zeros_like(x)
    for j in range(len(x)):
        b[j] = c[i] * B(x[j], k, i, t)
    plt.plot(x, b)

plt.title("BSpliney")
plt.xlabel("x")
plt.ylabel("y")
plt.show()








