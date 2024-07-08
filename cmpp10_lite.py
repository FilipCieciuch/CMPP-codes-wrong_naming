# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:07:39 2023

@author: filip
"""

import numpy as np
import matplotlib.pyplot as plt
import copy


#%% [1]

def create_grid(Nx, Ny, I):
    grid_height = np.zeros((Ny, Nx))
    for y in range(Ny):
        for x in range(Nx):
            R = np.random.rand()
            grid_height[y][x] = I*y + R*10**(0)
            
    return grid_height

def plot(grid, drop):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    im = ax.imshow(grid)
    # im = ax.imshow(grid, vmin = -250, vmax = 200)
    plt.colorbar(im)
    ax.invert_yaxis()
    plt.title("# of drops = " + str(drop))
    # plt.show()
    plt.savefig("Rain2/drop_" + str(drop).zfill(6), bbox_inches = 'tight')
    plt.close()

def move_droplet(indices, grid_height, beta, Nx, Ny, drop, n):
    directions = [[1,0], [-1,0], [0,1], [0,-1]]
    probs = []
    sum_probs = 0
    height = grid_height[indices[1]][indices[0]]
    possible_directions = []
    
    for i in range(4):
        new_indices = indices + np.array(directions[i])
        new_indices[0] = new_indices[0] % Nx
        new_height = grid_height[new_indices[1]][new_indices[0]]
        x = height - new_height 
        if x >= 0:
            prob = np.exp(beta*x)
            probs.append( prob )
            sum_probs += prob
        else:
            probs.append( 0 )
        possible_directions.append(new_indices)
        
    if sum_probs > 0:
        probs /= sum_probs
    else:
        return (-1, -1)
        # probs = [1/4, 1/4, 1/4, 1/4]
    chosen_ind_n = np.random.choice((0, 1, 2, 3), 1, p = probs)[0]
    final_indices = indices + directions[chosen_ind_n]
    final_indices[0] = final_indices[0] % Nx
    return final_indices


def droplet_track(grid_height, Nx, Ny, beta,drop):
    wet_sites = []
    indices = np.array([np.random.randint(0, Nx-1), np.random.randint(0, Ny-2)])
    n = 0
    while indices[1] > 0:
        indices = move_droplet(indices, grid_height, beta, Nx, Ny, drop, n)
        wet_sites.append(tuple(indices))
        n+=1
    
    x_hashable = map(tuple, wet_sites)
    wet_sites = set(x_hashable)
    return wet_sites


def update_height(grid_height, wet_sites, dh):
    for site in wet_sites:
        grid_height[site[1], site[0]] -= dh
    
    return grid_height

def rain(grid_height, drops, dh, R):
    for drop in range(drops):
        if not(drop % 500):
            print(drop/drops*100, "%", "\t drops: ", drop)
            print("----------------------------------")
            plot(grid_height, drop)
        wet_sites = droplet_track(grid_height, Nx, Ny, beta, drop)
        
        grid_height = update_height(grid_height, wet_sites, dh)
        avalanche_happened = True
        
        while avalanche_happened:
            avalanche_happened, avalanche_sites_list = avalanche_sites(grid_height, R, wet_sites, drop)
            perform_avalanche(avalanche_sites_list, grid_height)
            

def avalanche_sites(grid_height, R, wet_sites, drop):
    avalanche_happened = False
    avalanche_sites = []
    for site in wet_sites:
        x = site[0]
        y = site[1]
        if y == 0:
            locations = [(y + 1, x),
                         (y, (x + 1) % Nx),
                         (y, (x - 1) % Nx)]
        elif y == Ny-1:
            locations = [(y - 1, x),
                         (y, (x + 1) % Nx),
                         (y, (x - 1) % Nx)]            
        else:
            locations = [(y + 1, x), (y - 1, x),
                         (y, (x + 1) % Nx),
                         (y, (x - 1) % Nx)]
    
        max_difference = 0
        for neighbor in locations:
            shifted_indices = [neighbor[1], neighbor[0]]        # [x,y]
            difference = abs(grid_height[y, x] - grid_height[shifted_indices[1], shifted_indices[0]])
            if difference > max_difference:
                max_difference = difference

        if max_difference > R:
            avalanche_sites.append((x, y, max_difference))
            avalanche_happened = True

        if max_difference > 1e5:
            print("max = ", max_difference)
            print("position x = ", x, " y = ", y)
            print("exit")
            exit

    return avalanche_happened, avalanche_sites


def perform_avalanche(avalanche_sites, grid_height):
    for site in avalanche_sites:
        grid_height[site[1], site[0]] -= 0.25 * -site[2]

#%% [2]

Nx = 300     #300
Ny = 200    #200
I = 1.0
dh = 10
beta = 0.05

# r = 200
r = 10000

# drops = 1
# drops = 10001
drops = 100001


grid = create_grid(Nx, Ny, I)

rain(grid, drops, dh, r)

#plot(grid)



            