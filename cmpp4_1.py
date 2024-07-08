# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:42:42 2023

@author: filip

Task: Gray-Scott 1D
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
from progressbar import progressbar

plt.ioff()


def Laplace(x):
    return (np.roll(x,-1) + np.roll(x, 1) - 2*x) / (0.02**2)

x = np.linspace(0, 2, 100)


N=100
u = np.ones(N) 
v = np.zeros(N)
xs= np.arange(N)
for i in range(int(N/4), int(3*N/4)):
    u[i] = np.random.random()*0.2+0.4
    v[i] = np.random.random()*0.2+0.2



Du = 2*10**(-5)
Dv = 1*10**(-5)
F = 0.025
k = 0.055

'''
plt.figure(figsize = (5,3), dpi=200)
plt.plot(x, u)
plt.show()
'''

def TimeDerivative_u(u, v):
    return Du*Laplace(u) - u * v**2 + F * (1 - u)

def TimeDerivative_v(u, v):
    return Dv*Laplace(v) + u * v**2 - (F + k) * v


def Evolution_u(u, v):
    return u + TimeDerivative_u(u, v)

def Evolution_v(u, v):
    return v + TimeDerivative_v(u, v)

u_matrix = [u]
v_matrix = [v]

for t in progressbar(range(500)):
    v = Evolution_v(u, v)
    u = Evolution_u(u, v)
    u_matrix.append(u)
    v_matrix.append(v)
#    plt.plot(x, u)

    if t%10==0:
        print(t)    
        plt.figure(figsize = (5,3), dpi=200)
        plt.plot(x, v)
#        plt.title("t = ", t)
#        plt.show()
        plt.savefig("Gray-Scott_1D/" + str(t).zfill(3) + ".png")
        plt.close()


# Create the frames
frames = []
imgs = glob.glob("Gray-Scott_1D/???.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('ut(t).gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=1)



'''


fig=plt.figure(figsize=(5,3), dpi=200) 
ax=fig.add_subplot(111)
cax = ax.imshow(u_matrix, interpolation='nearest', aspect='auto')
cax.set_clim(vmin=0, vmax=1)
cbar = fig.colorbar(cax, ticks=[0,0.3, 0.5,1], orientation='vertical')
#plt.clf()
#plt.show()

'''








