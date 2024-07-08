# -*- coding: utf-8 -*-
"""
Created on Fri May  5 01:31:58 2023

@author: filip
"""

import numpy as np

def calculate_rho_and_f(Ny, Nx, u, f):
    W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]

    f=[]
    rho = []
    for k in range(Nx):
        for i in range(Ny):
            f_i=[]
            rho_i = np.sum(f[k][i])
            u2 = np.sum((u[0][i])**2)
            for j in range(9):
                f_i.append( W[j] * rho_i * (1 + 3 * np.dot(e[j], u[0][i])  + 9/2 * np.dot(e[j], u[0][i])**2 - 3/2 * u2 ))
            f.append(np.array(f_i))
            rho.append(np.array(rho_i))

    f = np.array(f)
    rho = np.array(rho)
    return [rho, rho*f]

def calculate_in_rho_and_f(Ny, Nx, u0):
    W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]

    f=[]
    rho = []
    for i in range(Ny):
        f_i=[]
        rho_i = []
        u2 = np.sum((u0[0][i])**2)
        for k in range(9):
            f_i.append( W[k] * (1 + 3 * np.dot(e[k], u0[0][i])  + 9/2 * np.dot(e[k], u0[0][i])**2 - 3/2 * u2 ))
        rho_ij = ( 2*(f_i[3] + f_i[6] + f_i[7]) + (f_i[0] + f_i[2] + f_i[4]) ) /  (1 - u2**(1/2))
        rho_i.append( rho_ij )
        for j in range(9):
            f_i.append( W[j] * rho_ij * (1 + 3 * np.dot(e[j], u0[0][i])  + 9/2 * np.dot(e[j], u0[0][i])**2 - 3/2 * u2 ))
        f.append(np.array(f_i))
        rho.append(np.array(rho_i))
    f = np.array(f)
    rho = np.array(rho)
    return [rho, rho*f]


def collision():

    f_col = f * obstacle
    for i, column in enumerate(obstacle):
        for j, elem in enumerate(column):
            if elem:
                f3 = f[i][j][3]
                f4 = f[i][j][4]
                f7 = f[i][j][7]
                f8 = f[i][j][8]
                f_col[i][j][3] = f[i][j][1]
                f_col[i][j][1] = f3
                f_col[i][j][4] = f[i][j][2]
                f_col[i][j][2] = f4
                f_col[i][j][7] = f[i][j][5]
                f_col[i][j][5] = f7
                f_col[i][j][8] = f[i][j][6]
                f_col[i][j][6] = f8
    return f_col
#%%

wedge = wedge(Nx, Ny)


fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
ax.imshow(np.transpose(wedge), cmap = 'hot')
plt.show()
#    plt.savefig('')
plt.close()

