# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:16:39 2023

@author: filip
"""

import numpy as np
import matplotlib.pyplot as plt
import copy


#%% [0] Definitions

def update_grid(grid, result):
    C = result
    C_grid = grid
    W = np.roll(result, 1, axis=1)
    W_grid = np.roll(grid, 1, axis=1)
    E = np.roll(result, -1, axis=1)
    E_grid = np.roll(grid, -1, axis=1)
    N = np.roll(result, 1, axis=0)
    N_grid = np.roll(grid, 1, axis=0)
    S = np.roll(result, -1, axis=0)
    S_grid = np.roll(grid, -1, axis=0)
    NW = np.roll(np.roll(result, 1, axis=1), 1, axis=0)
    NW_grid = np.roll(np.roll(grid, 1, axis=1), 1, axis=0)
    NE = np.roll(np.roll(result, -1, axis=1), 1, axis=0)
    NE_grid = np.roll(np.roll(grid, -1, axis=1), 1, axis=0)
    SW = np.roll(np.roll(result, 1, axis=1), -1, axis=0)
    SW_grid = np.roll(np.roll(grid, 1, axis=1), -1, axis=0)
    SE = np.roll(np.roll(result, -1, axis=1), -1, axis=0)
    SE_grid = np.roll(np.roll(grid, -1, axis=1), -1, axis=0)

    M_max = np.zeros(C.shape)
    all_neighbors = [C, W, E, N, S, NW, NE, SW, SE]
    all_strategies = [C_grid, W_grid, E_grid, N_grid, S_grid, NW_grid, NE_grid, SW_grid, SE_grid]
    updated_grid = grid
    for idx, neighbor in enumerate(all_neighbors):
        M_max_old = copy.deepcopy(M_max)
        M_max = np.maximum(neighbor, M_max)
        M_if = M_max > M_max_old
        updated_grid = updated_grid * (np.ones(grid.shape) - M_if.astype(int)) + all_strategies[idx] * M_if.astype(int)

    # address_of_max = np.argmax(np.ndarray((C, W, E, N, S, NW, NE, SW, SE)), axis=0)
    return updated_grid

def calculate_points(grid, b):
    positions = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 0), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    points = np.zeros((201, 201))

    for position in positions:
        #print(position)
        points += grid * np.logical_not(np.roll(grid, position, (0,1))) * b + np.logical_not(grid) * ( np.logical_not(np.roll(grid, position, (0,1))))

    return points

def plot_evolution(grid0, grid1, t):
    cc = np.logical_and(np.logical_not(grid0), np.logical_not(grid1)).astype(float) * 0
    dc = np.logical_and(grid0, np.logical_not(grid1)).astype(float) * 0.3
    dd = np.logical_and(grid0, grid1).astype(float) * 0.66
    cd = np.logical_and(np.logical_not(grid0), grid1).astype(float) * 1
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)
    cax = ax.imshow(cc + dc + dd + cd, interpolation='nearest')
    #plt.savefig('state' + str(t), dpi=300)
    plt.show()
    plt.close()
    

def simulation(b):
    grid = np.random.randint(0,2,(201, 201))
    for t in range(100):
        points = calculate_points(grid, b)
        grid_old = grid
        grid = update_grid(grid, points)
        #plot_evolution(grid_old, grid, t)
    f = np.sum(grid)
    return f
        
    
#%% [1]

# 1 = True = Defector
# 0 = False = Cooperate

zeros = np.zeros((201, 201), dtype=bool)
grid = np.zeros((201, 201), dtype=bool)
grid[100][100] = 1

b = 1.9


#if_zeros = np.bool(grid == zeros)

'''
for t in range(100):
    points = calculate_points(grid, b)
    grid_old = grid
    grid = update_grid(grid, points)
    plot_evolution(grid_old, grid, t)
    
simulation(b)
'''
num = 25
bs = np.linspace(1.75, 2.2, num)
f = []
i=0
for b in bs:
    f.append(simulation(b))
    i+=1
    print(i/25*100, '%')

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
cax = plt.scatter(bs, f)
plt.show()


#%% [2]

'''
ind = np.zeros((201, 201))
for i, line in enumerate(points):
    for j, element in enumerate(line):
        matrix = grid[ i-1 : i+1 ][ j-1 : j+1 ]
        ind[i][j] = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)

'''





