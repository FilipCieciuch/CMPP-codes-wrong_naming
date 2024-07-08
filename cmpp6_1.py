# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:59:13 2-123

@author: filip
"""

import numpy as np
import matplotlib.pyplot as plt


#%% [1] Definitions

def get_weights(x):
    w = np.zeros((30, 30))
    n = 0
    for x_mu in x:
        x_mu = x_mu.flatten()
        w += np.kron(x_mu, x_mu).reshape((30,30)) - np.ones((len(x_mu), len(x_mu)))
        n += 1

    return w/n

def get_energy(x, w):
    # return -1/2 * np.tensordot(x, w, axes = 1) * np.tensordot(w, x, axes = 1)
    return -1/2 * np.matmul(np.transpose(x), np.matmul(w, x))
    # return -1/2 * np.dot(x, np.dot(w, x))

def update_network(x, w):
    # return np.sign(np.matmul(w, x))
    print("w = ", np.shape(w))
    output = np.matmul(w, np.transpose(x))
    print(np.shape(output))
    return np.sign(output)

def plot(x, title):
    x = x.reshape((6,5))
    plt.figure(dpi=300)
    im = plt.imshow(x, vmin=-2, vmax=2)
    plt.colorbar(im)
    # plt.xticks(np.arange(0, 5, 1), [])
    # plt.yticks(np.arange(0, 1, 6), [])
    # plt.yticks([])
    # plt.grid(True, axis='x', lw=1, c='black')
    # plt.tick_params(axis='x', length=0)
    plt.title(title)

    #plt.savefig("Chromosomes/" + str(t).zfill(3) + '.png')
    plt.show()
    plt.close()
    
def plot_energy(energy):
    plt.figure(dpi=300)
    x = np.linspace(1, len(energy))
    plt.plot(x, energy)
    plt.title("Energy of time") 
    plt.show()
    plt.close()

#%% [2] Defining sample matrices

zero = np.matrix([
-1, 1, 1, 1, -1,
1, -1, -1, -1, 1,
1, -1, -1, -1, 1,
1, -1, -1, -1, 1,
1, -1, -1, -1, 1,
-1, 1, 1, 1, -1
])

one= np.matrix([
-1, 1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1,
-1, -1, 1, -1, -1
])


two = np.matrix([
1, 1, 1, -1, -1,
-1, -1, -1, 1, -1,
-1, -1, -1, 1, -1,
-1, 1, 1, -1, -1,
1, -1, -1, -1, -1,
1, 1, 1, 1, 1,
])

noisy0 = np.matrix([
-1, 1, 1, 1, -1,
1, -1, -1, -1, -1,
1, -1, -1, -1, 1,
-1, -1, -1, -1, -1,
-1, -1, -1, -1, -1,
-1, -1, 1, -1, -1,
])


#%% [3] 

x0 = [zero, one, two]
x = noisy0
energy = []

for i in range(10):
    w = get_weights(x0)
    x = update_network(x, w)
    print(np.shape(x))
    print(np.shape(w))
    energy.append(get_energy(x, w))

plot(energy)




#%% [4] Plotting

plot(noisy0, "x")

    
    





