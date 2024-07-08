# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:35:51 2023

@author: filip
"""

import matplotlib.pyplot as plt
import numpy as np


def fun(x0, p0, N, K):
    x, p = [x0], [p0]
    for n in range(N):
        p.append(p[n] + K * np.sin(x[n]))
        x.append((x[n] + p[n+1]) % (2*np.pi))
    return [x, p]


x1, p1 = 3, 1.9
x2, p2 = 3, 1.8999

K = 1.2
N = 50
n = np.arange(N+1)

x_1, p_1 = fun(x1, p1, N, K)
x_2, p_2 = fun(x2, p2, N, K)


plt.figure(figsize = (5,3), dpi=300)


plt.subplot(2,1,1)
plt.plot(n, x_1, label = "x1(n)")
plt.xlabel("n")
plt.ylabel("x1")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(n, x_2, label = "x2(n)")
plt.xlabel("n")
plt.ylabel("x2")

plt.title("x(n)")
plt.show()



plt.figure(figsize = (5,3), dpi=300)

plt.subplot(2,1,1)
plt.plot(n, p_1, label = "p1(n)")
plt.xlabel("n")
plt.ylabel("p1")

plt.subplot(2,1,2)
plt.plot(n, p_2, label = "p2(n)")
plt.xlabel("n")
plt.ylabel("p2")

plt.title("p(n)")
plt.show()


