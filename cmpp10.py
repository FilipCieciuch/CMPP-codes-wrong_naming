# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:07:39 2023

@author: filip
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy


#%% [1]

def create_grid(Nx, Ny, I):
    grid_height = np.zeros((Ny, Nx))
    for y in range(Ny):
        for x in range(Nx):
            R = np.random.rand()
            grid_height[y][x] = I*y + R*10**(0)
            
    return grid_height

def plot(grid, title):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    im = ax.imshow(grid)
    # im = ax.imshow(grid, vmin = -250, vmax = 200)
    plt.colorbar(im)
    ax.invert_yaxis()
    plt.title(title)
    plt.show()
    # plt.savefig("initial slope")
    plt.close()

def plot3D(grid_height, rivers_map_grid, title):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    xx, yy = np.meshgrid(np.linspace(0,1,300), np.linspace(0,1,200))
    surf = ax.plot_surface(xx, yy, grid_height, cmap = cm.viridis)
    surf2 = ax.contour(xx, yy, rivers_map_grid, offset = 2000)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(title)
    plt.show()
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
        if not np.any(np.isnan(probs)):
            chosen_ind_n = np.random.choice((0, 1, 2, 3), 1, p = probs)[0]
            final_indices = indices + directions[chosen_ind_n]
            final_indices[0] = final_indices[0] % Nx
            drop_in_hole = False
            return final_indices, drop_in_hole

    drop_in_hole = True
    return indices, drop_in_hole


def droplet_track(grid_height, Nx, Ny, beta,drop):
    wet_sites = []
    indices = np.array([np.random.randint(0, Nx-1), np.random.randint(0, Ny-2)])
    n = 0
    drop_in_hole = False
    while indices[1] > 0 and not drop_in_hole:
        indices, drop_in_hole = move_droplet(indices, grid_height, beta, Nx, Ny, drop, n)
        wet_sites.append(tuple(indices))
        n+=1
    
    x_hashable = map(tuple, wet_sites)
    wet_sites = set(x_hashable)
#    wet_sites = list(dict.fromkeys(wet_sites))
    return wet_sites


def update_height(grid_height, wet_sites, dh):
    for site in wet_sites:
        grid_height[site[1], site[0]] -= dh
    
    return grid_height

def rain(grid_height, drops, dh, R, R_crit):
    # drop_threshold = 400000
    for drop in range(drops):
        if not(drop % 2000):
            print("{0:1.0f} % \t drops: {1:7.0f}".format(drop/drops*100, drop))
            print("----------------------------------")
            plot(grid_height, "# of drops = " + str(drop))
        # if drop > drop_threshold:
        #     print("drop no ", drop)
        #     print("calculating wet sites")
        wet_sites = droplet_track(grid_height, Nx, Ny, beta, drop)
       
        # if drop > drop_threshold:
        #     print("updating grid height")
        
        grid_height = update_height(grid_height, wet_sites, dh)
        avalanche_happened = True
        
        # if drop > drop_threshold:
        #     print("avalanche")
        
        # n = 0
        while avalanche_happened:
            # print("avalanche 1 calculating")
            avalanche_happened, avalanche_sites_list, possible_avalanche_sites = avalanche_sites(
                    grid_height, R, wet_sites, drop)
            avalanche2_happened = avalanche_happened
            while avalanche2_happened:
                avalanche2_happened, avalanche_sites_list, possible_avalanche_sites = avalanche_sites(
                    grid_height, R, possible_avalanche_sites, drop)
                perform_avalanche(avalanche_sites_list, grid_height)
                # n+=1
                # if avalanche2_happened:
                #     print("avalanche 2 happened")
                # print("avalanche 2 calculating")
            perform_avalanche(avalanche_sites_list, grid_height)
    
        # if n > 0 and not(drop % 20):
        #     print(n, " second avalanches happened")
    
    print("calculating rivers")
    rivers(grid_height, Ny, Nx, R_crit)
    print("rivers done")

def avalanche_sites(grid_height, R, possible_avalanche_sites, drop):
    avalanche_happened = False
    avalanche_sites = []
    # drop_threshold = 500
    possible_avalanche_sites2 = copy.copy(possible_avalanche_sites)
    for site in possible_avalanche_sites:
#        if drop > drop_threshold:
#            print("site ", site)
        x = site[0]
        y = site[1]
        # n = 0
        if y == 0:
            locations = [(x, y + 1),
                         ((x + 1) % Nx, y),
                         ((x - 1) % Nx, y)]
        elif y == Ny-1:
            locations = [(x, y - 1),
                         ((x + 1) % Nx, y),
                         ((x - 1) % Nx, y)]            
        elif y < Ny-1 and y > 0:
            locations = [(x, y + 1), (x, y - 1),
                         ((x + 1) % Nx, y),
                         ((x - 1) % Nx, y)]
        else:
            # print(site)
            break
        possible_avalanche_sites2.update(set(map(tuple, locations)))
    
        max_difference = 0
        # if drop>250:
        #     print("neighbor start")
        for neighbor in locations:
            shifted_indices = [neighbor[0], neighbor[1]]        # [x,y]
            difference = grid_height[y, x] - grid_height[shifted_indices[1], shifted_indices[0]]
            if abs(difference) > max_difference:
                max_difference = difference
        # if drop>250:
        #     print("neighbor stop \t max difference")

        if abs(max_difference) > R:
            avalanche_sites.append((x, y, max_difference))
#            grid_height[y, x] -= 0.25 * max_difference
#            print("max = ", max_difference)
#            n+=1
#                print("avalanche ", n)
            avalanche_happened = True
#                print(n)

        if max_difference > 1e5:
            print("max = ", max_difference)
            print("position x = ", x, " y = ", y)
            print("exit")
            exit

    return avalanche_happened, avalanche_sites, possible_avalanche_sites2


def perform_avalanche(avalanche_sites, grid_height):
    # n=0
    for site in avalanche_sites:
#        if drop > drop_threshold:
#            print("avalanche site ", site)
        grid_height[site[1], site[0]] -= 0.25 * site[2]
        # n+=1
#        print("avalanche done")
    # if n != 0:
    #     print(n, " avalanches performed")
    

def rivers(grid_height, Ny, Nx, R_crit):
    rivers_grid = np.ones((Ny, Nx))
    i=0
    for y in range(Ny-2):
        for x in range(Nx):
            i+=1
            # print("steepest descent", i)
            # print("x, y = ", x, y)
            droplet_track = droplet_track_steepest_descent(grid_height, (x, y + 1), i) 
            # print("steepest descent done")            
            rivers_grid_update(rivers_grid, droplet_track)
            if not(i%6000):
                print(i/(200*300)*100, "%")
            # print("iteration ", i)
            # print("location", x, y)
        rivers_map_grid = rivers_map(rivers_grid, R_crit)
    plot(rivers_map_grid, 'rivers')
    plot3D(grid_height, rivers_map_grid, '3D plot, # of drops = ' + str(drops))

def move_droplet_steepest_descent(indices, grid_height, Nx, Ny, drop):
    directions = [[1,0], [-1,0], [0,1], [0,-1]]
    height = grid_height[indices[1]][indices[0]]
    max_height_difference = 0
    # chosen_indices = indices + np.array(directions[3])
    
    for i in range(4):
        new_indices = indices + np.array(directions[i])
        new_indices[0] = new_indices[0] % Nx
        new_height = grid_height[new_indices[1]][new_indices[0]]
        height_difference = height - new_height 
        # print("height difference = ",height_difference)
        if height_difference > max_height_difference:
            max_height_difference, chosen_indices = height_difference, new_indices
    if max_height_difference > 0:
        # in_hole = False
        return chosen_indices
    else:
        # in_hole = True
        return (-1,-1)

def droplet_track_steepest_descent(grid_height, start_indices, drop):
    indices = start_indices
    indices_list = [indices]
    i=0
    while indices[1] > 1:
        # print("y = ", indices[1])
        indices = move_droplet_steepest_descent(indices, grid_height, Nx, Ny, drop)
        indices_list.append(indices)
        i+=1
        # if i>10:
            # print("iterations", i)
            # print("x, y = ", indices[0], indices[1])
    return indices_list

def rivers_grid_update(rivers_grid, droplet_track):
    for indices in droplet_track:
        rivers_grid[indices[1], indices[0]] += 1

def rivers_map(rivers_grid, R_crit):
    rivers_map = np.zeros((Ny, Nx))
    for x in range(Nx):
        for y in range(Ny-1):
            if rivers_grid[y,x] > R_crit:
                rivers_map[y, x] = 1.
    return rivers_map

#%% [2]

Nx = 300     #300
Ny = 200    #200
I = 1.0
dh = 10
beta = 0.05
R_crit = 200

# r = 200
r = 10000

# drops = 1
drops = 100001
#drops = 100000
# drops = 5000


grid = create_grid(Nx, Ny, I)

rain(grid, drops, dh, r, R_crit)

#plot(grid)

#%% [3] Debugging
'''
grid_height = grid
R=r

for drop in range(drops):
    if not(drop % 100):
        print("{0:1.0f} % \t drops: {1:7.0f}".format(drop/drops*100, drop))
        print("----------------------------------")
        plot(grid_height, drop)
    # if drop > drop_threshold:
    #     print("drop no ", drop)
    #     print("calculating wet sites")
    wet_sites = droplet_track(grid_height, Nx, Ny, beta, drop)
   
    # if drop > drop_threshold:
    #     print("updating grid height")
    
    grid_height = update_height(grid_height, wet_sites, dh)
    avalanche_happened = True
    
    # if drop > drop_threshold:
    #     print("avalanche")
    
    # n = 0
    while avalanche_happened:
        # print("avalanche 1 calculating")
        avalanche_happened, avalanche_sites_list, possible_avalanche_sites = avalanche_sites(grid_height, R, wet_sites, drop)
        avalanche2_happened = avalanche_happened
        while avalanche2_happened:
            avalanche2_happened, avalanche_sites_list, possible_avalanche_sites = avalanche_sites(grid_height, R, possible_avalanche_sites, drop)
            perform_avalanche(avalanche_sites_list, grid_height)
            # n+=1
            # if avalanche2_happened:
            #     print("avalanche 2 happened")
            # print("avalanche 2 calculating")
        perform_avalanche(avalanche_sites_list, grid_height)

#%%
print("calculating rivers")
rivers_grid = np.ones((Ny, Nx))
i=0
for y in range(Ny-1):
    for x in range(Nx):
        i+=1
        droplet_track_list = droplet_track_steepest_descent(grid_height, (x, y), i)             
        rivers_grid_update(rivers_grid, droplet_track_list)
        if not(i%6000):
            print(i/(200*300)*100, "%")
        # print("iteration ", i)
        # print("location", x, y)

rivers_map_grid = rivers_map(rivers_grid, R_crit)
plot(rivers_map_grid, 'rivers')

#%%

fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')
xx, yy = np.meshgrid(np.linspace(0,1,300), np.linspace(0,1,200))
surf = ax.plot_surface(xx, yy, grid_height, cmap = cm.viridis)
surf2 = ax.contour(xx, yy, rivers_map_grid, offset = 500)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
plt.close()





'''












            