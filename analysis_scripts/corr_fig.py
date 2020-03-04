import numpy as np
import matplotlib.pyplot as plt

A = np.zeros(10000)
C = np.zeros(10000)
B = np.arange(10000)/100

for i in range(10000):
    A[i] = 1 + 0.25 * np.cos(B[i]/4) * np.exp(-B[i]/8)
    C[i] = 1 + 0.25 * np.cos(B[i]/8)

plt.plot(B,A)
plt.plot(B,C)
plt.title('Correlation Length')
plt.ylabel('g(r)')
plt.xlabel('r')
plt.legend(['Amorphous','Crystalline'])