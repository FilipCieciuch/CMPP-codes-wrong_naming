# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:14:09 2023

@author: filip
"""


import matplotlib.pyplot as plt
import numpy as np



def fun(x0, p0, N, K):
    x, p = [x0], [p0]
    xnew, pnew = x, p
    for n in range(N):
        pnew.append((p[n] + K * np.sin(x[n])) % (2*np.pi))
        xnew.append((x[n] + p[n+1]) % (2*np.pi))
        x = xnew
        p = pnew
    return [x, p]


x1, p1 = 3, 1.9
x2, p2 = 3, 1.8999

K = 5.5 #1.9, 2.1, 5.5
N = 1000
n = np.arange(N+1)

x_1, p_1 = fun(x1, p1, N, K)
x_2, p_2 = fun(x2, p2, N, K)


plt.figure(figsize = (5,5), dpi=150)



for i in range(100):
    r, g, b = np.random.random(3)
    x0, p0 = np.random.random(2)*2*np.pi
    x, p = fun(x0, p0, N, K)    
    plt.plot(x, p, ',', color = (r,g,b))
    
plt.xlabel("x")
plt.ylabel("p")
plt.title("p(x)")

plt.show()
#plt.savefig('Pictures/K=1.9.png')

