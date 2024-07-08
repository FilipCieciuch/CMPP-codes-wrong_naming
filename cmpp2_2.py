# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 02:01:07 2023

@author: filip
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
#from scipy.fft import fft, ifft
import scipy.fft

def norm(psiG):
    return psiG / np.linalg.norm(psiG)


M = 1000
K = 2.1
n = np.arange(M)
x = 2 * m.pi * n / M
p = 2 * m.pi * n / M


x0, p0 = 2, 2
#x0, p0 = 2.5, 0.1
hbar = 2 * m.pi / M
j = (-1)**(1/2)



d_list = []
for d in range(-4,4):
    d_list.append(np.exp( - ((x - x0 + 2 * m.pi * d * np.ones(len(x)))**2) / (2 * hbar)))

psiG = np.zeros(M, dtype = complex)
psiG = np.exp(j * p0 * x / hbar) * np.sum(d_list, axis=0)

psiG = norm(psiG)

wn = scipy.fft.fft(psiG)
wn = norm(wn)

'''
plt.figure(figsize = (5,5), dpi=150)
plt.plot(x, np.abs(psiG)**2)
plt.plot(x, np.abs(wn)**2)
plt.show()
'''


Vn = np.exp( - j / hbar * K * np.cos(x))
Pm = np.exp( - j / (2*hbar) * p**2)


for i in range(20):
    psiG = scipy.fft.ifft(Pm*scipy.fft.fft(Vn * psiG))
    #wn = scipy.fft.fft(psiG)
    psiG = norm(psiG)
    #wn = norm(wn)
    plt.figure(figsize = (5,5), dpi=150)
    plt.plot(x, np.abs(psiG)**2)
    #plt.plot(x, np.abs(wn)**2)
    #plt.ylim(-0.015, 0.05)
    #plt.savefig("wave_packet" + str(t).zfill(3) + ".png")
    #plt.close()
    plt.show()



