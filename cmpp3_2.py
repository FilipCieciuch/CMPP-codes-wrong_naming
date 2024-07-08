# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:38:28 2023

@author: filip
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m



def get_matrix(N):
    h = np.random.randn(N,N)
    return (h+h.T)/2

def wignerGOE(s):
    return m.pi / 2 * s * m.exp( -m.pi / 4 * s ** 2)

def wignerGUE(s):
    return 32 / (m.pi**2) * s**2 * m.exp( -4 / m.pi * s ** 2)

N, n = 6, 20000
#N, n = 20, 10000
#N, n = 200, 500
eigen = []
difflist = []

for ni in range(n):
    M = get_matrix(N)
    eigen.append(np.linalg.eig(M)[0])
    eigen = np.sort(eigen)
    diff = np.diff(eigen)
    diff /= np.average(np.array(diff))
    difflist.append(diff)

eigen = [value for l in eigen for value in l]
difflist = [value for l in difflist for value in l]
#eigen = eigen / np.average(np.array(eigen))


plt.figure(figsize = (5,3), dpi=200)
n, bins, _ = plt.hist(diff, 50, density=True, alpha=0.75)
y = [wignerGOE(x) for x in bins]
#plt.plot(bins, y)


plt.show()





'''

plt.figure(figsize = (5,3), dpi=200)
n, bins, _ = plt.hist(eigen, 50, density=True, alpha=0.75)
n = n/np.average(np.array(n))

#bins = np.linspace(-5, 5, 100)
y = [wigner(x, N) for x in bins]
plt.plot(bins, y, 'r-', linewidth=2)
plt.show()

'''
