# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:48:33 2023

@author: filip
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import kron


def H(lambda_):
    sz = np.array( [[1,0.],[0.,-1]] )
    sx = np.array( [[0.,1],[1,0.]] )
    one = np.eye(2)
    sz_1 = kron( sz, kron(one, one))
    sz_2 = kron( one, kron(sz, one))
    sz_3 = kron( one, kron(one, sz))
    sx_1 = kron( sx, kron(one, one))
    sx_2 = kron( one, kron(sx, one))
    sx_3 = kron( one, kron(one, sx))
    
    h1 = 0.6
    h2 = 0
    h3 = 0
    J12 = -1.1
    J13 = -2.1
    J23 = -3.8
    
    H0 = -sx_1 - sx_2 - sx_3
    H1 = -J12 * np.dot(sz_1, sz_2) - J13 * np.dot(sz_1, sz_3) - J23 * np.dot(sz_2, sz_3) - h1 * sz_1 - h2 * sz_2 - h3 * sz_3
    return (1 - lambda_) * H0 + lambda_ * H1

def delta_E(H, lambda_):
    w, v = np.linalg.eigh(H(lambda_))
    return w[1] - w[0]

def ground_state(H, lambda_):
    w, v = np.linalg.eigh(H(lambda_))
    ground_state = v[:, 0]
    return ground_state

def S(lambda_, ground_state):
    one = np.eye(2)
    sz = np.array( [[1,0.],[0.,-1]] )
    sz_1 = kron( sz, kron(one, one))
    sz_2 = kron( one, kron(sz, one))
    sz_3 = kron( one, kron(one, sz))
    S1 = np.dot(np.transpose(ground_state), np.dot(sz_1, ground_state))
    S2 = np.dot(np.transpose(ground_state), np.dot(sz_2, ground_state))
    S3 = np.dot(np.transpose(ground_state), np.dot(sz_3, ground_state))
    return S1, S2, S3

def plotting_E(H):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    lambda_list = np.linspace(0, 1, 100)
    delta_E_list = [delta_E(H, lambda_) for lambda_ in lambda_list]
    ax.plot(lambda_list, delta_E_list)
    plt.show()

def plotting_S(H):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    lambda_list = np.linspace(0, 1, 100)
    S1_list, S2_list, S3_list = [], [], []
    for lambda_ in lambda_list:
        S1, S2, S3 = S(lambda_, ground_state(H, lambda_))
        S1_list.append(S1)
        S2_list.append(S2)
        S3_list.append(S3)
        
    ax.plot(lambda_list, S1_list, label = "S1")
    ax.plot(lambda_list, S2_list, label = "S2")
    ax.plot(lambda_list, S3_list, label = "S3")
    plt.legend()
    plt.show()
    
def T_AQC():
    N = 1000
    T_AQC = 0
    lambda_list = np.linspace(0, 1, N)
    d_lambda = 1/N
    for lambda_ in lambda_list:
        T_AQC += d_lambda/(delta_E(H, lambda_))**2
    
    return T_AQC

def Graph():
    # Graph = {'Q1' : ['Q2', 'Q6', 'Q5'], 'Q2' : ['Q3', 'Q7', 'Q6'], 'Q3' : ['Q4', 'Q8', 'Q7']}
    Graph = {'Q1' : [2, 6, 5], 'Q2' : [3, 7, 6], 'Q3' : [4, 8, 7]}
    return Graph

def sz_list():
    one = np.eye(2)
    sz = np.array( [[1, 0.],[0., -1]] )
    sz_list = []
    
    for i in range(8):
        if i==0:
            szi = sz
            for j in range(7):
                szi = kron(szi, one)
            sz_list.append(szi)
        else:
            szi = one
            for j in range(7):
                if j == i:
                    szi = kron(szi, sz)
                else:
                    szi = kron(szi, one)
            sz_list.append(szi)
    return sz_list

def H8(lambda_):
    no_of_atoms = 8
    J = np.ones((no_of_atoms, no_of_atoms))
    h = np.ones(no_of_atoms)
    h.fill(0.9)
    sz_list_ = sz_list()
    graph_ = Graph()
    H1 = 0
    for atom_id, atom in enumerate(graph_):
        # print(atom)
        for connection in graph_[atom]:
            # print(connection)
            H1 -= J[atom_id][connection-1] * np.dot(sz_list_[atom_id], sz_list_[connection-1])
    for i in range(no_of_atoms):
        H1 -= h[i] * sz_list_[i]
    
    H0 = -sum(sz_list_)
    
    return (1 - lambda_) * H0 + lambda_ * H1

        
plotting_E(H8)
# plotting_S(H8)




#%%
# print("T_AQC = ", T_AQC())





