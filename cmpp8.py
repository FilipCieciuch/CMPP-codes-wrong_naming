# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:29:57 2023

@author: filip
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

#%% [1]

def plot(u, t, Re, name):
    shape = (np.shape(u)[0], np.shape(u)[1])
    u_show = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            u_show[i][j] = np.sum(u[i][j]**2)
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    m = ax.imshow(np.transpose(u_show), cmap = 'hot')
    plt.title("Re = "+ str(Re) + "; t = " + str(t))
#    ax.legend(m)
#    plt.show()
    plt.savefig(name + '/t' + str(t).zfill(5) + ".png")
    plt.close()

def PBC(f, f_eq):
    f[1][0][:] = f_eq[1][0][:]
    f[5][0][:] = f_eq[5][0][:]
    f[8][0][:] = f_eq[8][0][:]
    f[3][Nx-1][:] = f[3][Nx-2][:]
    f[6][Nx-1][:] = f[6][Nx-2][:]
    f[7][Nx-1][:] = f[7][Nx-2][:]
    return f

def calculate_u0(Nx, Ny, u_in, epsilon=0.0001):
    u0 = []
    for i in range(Nx):
        u0_i = []
        for j in range(Ny):
            u0_i.append(u_in * (1 + epsilon * np.sin( j * 2 * np.pi  / (Ny-1)) ) * np.array((1,0)))
        u0.append(np.array(u0_i))
    return np.array(u0)

def calculate_f(rho, u):
    W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    f = []
    
    u2 = np.sum(u**2, axis=2)
    u = np.transpose(u, axes = (0,2,1))
    
    for k in range(9):
        f.append(  W[k] * rho * (1 + 3 * np.dot(e[k], u)  + 9/2 * np.dot(e[k], u)**2 - 3/2 * u2 ))
    
    f = np.array(f)
    return f

def calculate_u(rho, f):
    e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    f = np.transpose(f, axes = (1,2,0))
    return np.dot(f, e)/np.tensordot(rho, np.ones(2), axes=0)

def calculate_rho(f):
    return np.sum(f, axis=0)

def inlet_f(f, u0):
    W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    f = np.transpose(f, axes = (1,2,0))
    for i, f_i in enumerate(f[0]):
        u2 = np.sum((u0[0][i])**2)
        rho_ij = ( 2*(f_i[3] + f_i[6] + f_i[7]) + (f_i[0] + f_i[2] + f_i[4]) ) /  (1 - u2**(1/2))
        for j, f_ij in enumerate(f_i):
            f[0][i][j] =  W[j] * rho_ij * (1 + 3 * np.dot(e[j], u0[0][i])  + 9/2 * np.dot(e[j], u0[0][i])**2 - 3/2 * u2 )
    f = np.transpose(f, axes = (2,0,1))  
    return f

def collision_step(f, f_eq, tau):
    return f - (f - f_eq)/np.array(tau)

def wedge(Nx, Ny):
    wedge = np.fromfunction(lambda x, y: np.fabs(x - Nx/4) + np.fabs(y) < Ny/2 , (Nx, Ny))
    return wedge

def cylinder(Nx, Ny):
    cylinder = np.fromfunction(lambda x, y: (x - Nx/4)**2 + (y - Ny/2)**2 < Ny , (Nx, Ny))
    return cylinder

def reverse(i):
    i_in = [0, 1, 2, 3, 4, 5, 6, 7, 8] 
    i_out = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    for index in range(9):
        if i == i_in[index]:
            return i_out[index]

def collision(f, f_col, obstacle):
#    wedge = np.fromfunction(lambda x, y: np.fabs(x - Nx/4) + np.fabs(y) < Ny/2 , (Nx, Ny))
#    wedge = np.fromfunction(lambda x, y: (x - Nx/4)**2 + (y - Ny/2)**2 < Ny , (Nx, Ny))
    for i in range(9):
        f_col[i, obstacle] = f[reverse(i), obstacle]
    return f_col

def stream(f):
    e = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
#    f = np.transpose(f, axes = (2,0,1))         # changing np.shape(f) from (520, 180, 9) to (9, 520, 180)
    for i, ei in enumerate(e):
        f[i] = np.roll(f[i], ei, axis=(0,1))
        
#    f = np.transpose(f, axes = (1,2,0))         # changing np.shape(f) back from (9, 520, 180) to (520, 180, 9)
    return f
    

#%% [2]


Nx = 520
Ny = 180
u_in = 0.04                 #velocity in lattice units
Re = 220                   #Reynolds number
v_LB = u_in * (Ny/2)/Re     #viscosity
tau = 3*v_LB + 1/2          #relaxation time
epsilon = 0.0001


#Algorithm

#0
u0 = calculate_u0(Nx, Ny, u_in)
rho0 = np.ones((Nx, Ny))
f = calculate_f(rho0, u0)


time=20000

#%%

name = "LBMc220"

obstacle = cylinder(Nx, Ny)
plot(u0, 0, Re, name)
for t in range(time):

    #1
    f_eq = inlet_f(f, u0)
    
    #2
    f = PBC(f, f_eq)
    
    #3
    rho = calculate_rho(f)
    u = calculate_u(rho, f)
    f_eq = calculate_f(rho, u)
    
    #4
    f_col = collision_step(f, f_eq, tau)
    
    #5
    f_col = collision(f, f_col, obstacle)
    
    #6
    f = stream(f_col)
    
    #Plotting
    u = calculate_u(rho, f)
    
#    print(t)
    
    if not t%100:
        plot(u, t, Re, name)
        print(t/time*100, '%')

#%% [] Create GIF


# Create the frames
frames = []
imgs = glob.glob(name+"/t?????.png")
a=0

for i in imgs:
    a+=1    
    im = Image.open(i)
    frames.append(im.copy())

print(a)



frames[0].save(name+'.gif', format='GIF', disposal=2,
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=1)



