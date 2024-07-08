# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:31:32 2023

@author: filip
"""

#%% [0] Importing

import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

plt.ioff()

#%% [1] Plotting function definition

def plot(x, title, t, s):
    plt.figure(dpi=300)
    plt.imshow(x, extent=[0, s, 0, s], cmap='Greys')
    plt.xticks(np.arange(0, len(x), 1), [])
    plt.yticks(np.arange(0, 1, len(x)), [])
    plt.yticks([])
    plt.grid(True, axis='x', lw=1, c='black')
    plt.tick_params(axis='x', length=0)
    plt.title(title)

    plt.savefig("Chromosomes/" + str(t).zfill(3) + '.png')
    plt.close()

# Calculating Neighbourhood State N from state x
def get_N(x, s):
    positions = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 0), (0, 1),
                 (1, -1), (1, 0), (1, 1)]
    N = np.zeros((s,s))

    for i in range(9):
        N += 2**i * np.roll(x, positions[i], (1,0))

    return N

def Step(N, C, s):
    newx = [ C [ int(N[i][j]) ] for i in range(s) for j in range(s) ]
    newx = np.reshape(newx, (s,s))
    return newx

def Evolution(x, C, s, steps = 100):
    for t in range(steps):
        N = get_N(x, s)
        x = Step(N, C, s)
    return x

def Evolution_plotting(x, C, s, steps = 100):
    for t in range(steps):
        N = get_N(x, s)
        x = Step(N, C, s)
        plot(x, "t = " + str(t), t, s)
    return x


def calculate_points(x):
    
    horiz = np.roll(x, (1, 0), (1, 0)) - x == 0
    vert = np.roll(x, (0, 1), (1, 0)) - x == 0
    neighbors = horiz + vert

    diag_right = np.roll(x, (1, 1), (1, 0)) - x == 0
    diag_left = np.roll(x, (1, -1), (1, 0)) - x == 0
    diag = diag_right + diag_left

    N = neighbors * (-3) + (np.logical_not(neighbors)) * ( (diag) * 8 - np.logical_not(diag) * 5)

    return np.sum(N)


def Evaluation(C, n=10, s=12):
    points = 0

    for t in range(n):
        x = np.random.randint(0, 2, size=(s, s))
        x = Evolution(x, C, s)
        points += calculate_points(x)
    
    return points/n


def NewPopulation(Clist):
    
    dtype = [("matrix", np.ndarray), ("points", np.float64)]
    Clist = np.array(Clist, dtype = dtype)
    Clist = np.sort(Clist, order="points")
    
    newpopulation = []
    
    for i, item in enumerate(Clist):
        if not i%2 and i<10:
#            print(i)
            new_member1 = []
            new_member2 = []
            for j, C in enumerate(Clist[i][0]):
                if j%2:
                    new_member1.append(Clist[i][0][j])
                    new_member2.append(Clist[i+1][0][j])
                else:
                    new_member1.append(Clist[i+1][0][j])
                    new_member2.append(Clist[i][0][j])
                
            newpopulation.append(new_member1)
            newpopulation.append(new_member2)
        
    return newpopulation

#%% [3] 

Clist = []

for i in range(10):
    C = np.random.randint(0, 2, size=512)
    points = Evaluation(C)
    Clist.append((C, points))
    
Clist = NewPopulation(Clist)

generations = 100

for i in range(generations):
    Clist_new = []
    for C in Clist:
        points = Evaluation(C)
        Clist_new.append((C, points))
    
    Clist = NewPopulation(Clist_new)

Clist_new = []

for C in Clist:
    points = Evaluation(C)
    Clist_new.append((C, points))


#%% 

Clist = Clist_new

dtype = [("matrix", np.ndarray), ("points", np.float64)]
Clist = np.array(Clist, dtype = dtype)
Clist = np.sort(Clist, order="points")

#print(Clist)

s = 20
x = np.random.randint(0, 2, size=(s, s))

x = Evolution_plotting(x, Clist[0][0], s)


frames = []
imgs = glob.glob("Chromosomes/*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save('Chromosomes/GIF.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=1)








