# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:32:33 2023

@author: filip
"""


import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image

#%%

def evolution(grid_fish, grid_sharks):
    directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    
    sharks_list = []
    for x, line in enumerate(grid_sharks):
        for y, elem in enumerate(line):
            reproduction_time, energy_level = elem
            if energy_level > 0:
                sharks_list.append([x, y, reproduction_time, energy_level])
    number_of_sharks = len(sharks_list)
    
    for x, y, reproduction_time, energy_level in sharks_list:
        possible_directions = []
        fish_places = []
        for i in range(4):
            new_position_possible = [(x + directions[i][0]) % Nx, (y + directions[i][1]) % Nx]
            if grid_fish[new_position_possible[0]][new_position_possible[1]] > 0:
                fish_places.append(new_position_possible)
            elif grid_sharks[new_position_possible[0]][new_position_possible[1]][1] == 0:
                possible_directions.append(i)

        if len(fish_places) > 0:
            chosen_fish_coordinates = fish_places[np.random.choice(len(fish_places))]
            grid_fish[chosen_fish_coordinates[0]][chosen_fish_coordinates[1]] = 0
            if reproduction_time == 0:
                grid_sharks[x][y] = [B, E]
                grid_sharks[chosen_fish_coordinates[0]][chosen_fish_coordinates[1]] = [B, E]
            else:
                grid_sharks[x][y] = [0, 0]
                grid_sharks[chosen_fish_coordinates[0]][chosen_fish_coordinates[1]] = [reproduction_time - 1, E]
            
        elif len(possible_directions) > 0:
            new_direction = np.random.choice(possible_directions)
            new_position = [(x + directions[new_direction][0]) % Nx, (y + directions[new_direction][1]) % Nx]
            if reproduction_time == 0:
                grid_sharks[x][y] = [B, E]
                grid_sharks[new_position[0]][new_position[1]] = [B, energy_level-1]
            else:
                grid_sharks[x][y] = [0, 0]
                grid_sharks[new_position[0]][new_position[1]] = [reproduction_time - 1, energy_level-1]
        
    fish_list = []
    for x, line in enumerate(grid_fish):
        for y, is_fish in enumerate(line):
            if is_fish:
                reproduction_time = is_fish
                fish_list.append([x, y, reproduction_time])    

    number_of_fish = len(fish_list)                
    
    for x, y, reproduction_time in fish_list:
        possible_directions = []
        for i in range(4):
            new_position_possible = [(x + directions[i][0]) % Nx, (y + directions[i][1]) % Nx]
            if grid_fish[new_position_possible[0]][new_position_possible[1]] == 0 and grid_sharks[new_position_possible[0]][new_position_possible[1]][1] == 0:
                possible_directions.append(i)
        # print(len(possible_directions))
        if len(possible_directions) > 0:
            # print(True)
            new_direction = np.random.choice(possible_directions)
            new_position = [(x + directions[new_direction][0]) % Nx, (y + directions[new_direction][1]) % Nx]
            if reproduction_time-1 == 0:
                grid_fish[x][y] = A
                grid_fish[new_position[0]][new_position[1]] = A
            else:
                grid_fish[x][y] = 0
                grid_fish[new_position[0]][new_position[1]] = reproduction_time - 1

    return grid_fish, grid_sharks, number_of_fish, number_of_sharks

def plot(grid_fish, grid_sharks, t, name):
    figure = plt.figure(dpi=200)
    ax = figure.add_subplot(111)
    grid_sharks_plot = np.transpose(grid_sharks, axes = (2, 1, 0))
    grid_fish= np.transpose(grid_fish, axes = (1, 0))
    fish = (grid_fish > 0) * int(1)
    sharks = (grid_sharks_plot[1] > 0) * int(1)
    im = ax.imshow(fish-sharks, cmap = 'Spectral', vmin = -2, vmax = 2)
    plt.colorbar(im)
    plt.title("t = " + str(t))
    name += "/"
    plt.savefig(name + str(t).zfill(4))
    # plt.show()
    plt.close()

def plot_in_time(n_fish_list, n_sharks_list, part):
    figure = plt.figure(dpi=200)
    figure.add_subplot(111)
    size = 2
    length = int(len(n_fish_list)*part)
    x = np.linspace(1, length, length)
    plt.scatter(x, n_fish_list[:length], s = size)
    plt.scatter(x, n_sharks_list[0:length], s = size)
    plt.show()

def plot_phase_space(n_fish_list, n_sharks_list):
    figure = plt.figure(dpi=200)
    figure.add_subplot(111)
    plt.scatter(n_fish_list, n_sharks_list, s=2)
    plt.xlabel("N of fish")
    plt.ylabel("N of sharks")
    plt.show()

def gif(name, images):
    # Create the frames
    frames = []
    imgs = glob.glob(name+"/????.png")
    a=0

    print("Importing images...")

    for i, image in enumerate(imgs):
        # if i<855:
            a+=1
            im = Image.open(image)
            frames.append(im.copy())
            if not(i % (images//10)):
                print(i/images*100, "%", "\t frame ", i)
    print(a, "images imported")

    print("Creating GIF...")

    frames[0].save(name+'.gif', format='GIF', disposal=2,
                   append_images=frames[1:],
                   save_all=True,
                   duration=100, loop=1)

    print("GIF created")


#%%

Nx = 200
Nf = 15000
Ns = 2500
T = 5000
# Nx = 5
# Nf = 10
# Ns = 1
# T = 10
A = 3
B = 20
E = 4
name = "WATOR3_f_15000_s_2500_e_4"

grid_fish = np.zeros((Nx, Nx))
grid_sharks = np.zeros((Nx, Nx, 2))

for i in range(Nf):
    fish_made = False
    while not fish_made:
        x = np.random.randint(0, Nx)
        y = np.random.randint(0, Nx)
        is_fish = (grid_fish[x][y] > 0) * 1
        if not is_fish:
            reproduction_time = np.random.randint(1, A+1)
            grid_fish[x][y] = reproduction_time
            fish_made = True

for i in range(Ns):
    # print("shark no ", i)
    shark_made = False
    while not shark_made:
        x = np.random.randint(0, Nx)
        y = np.random.randint(0, Nx)
        # print(x, y)
        is_fish = (grid_fish[x][y] > 0) * 1
        is_shark = (grid_sharks[x][y][1] > 0) * 1
        if not is_fish and not is_shark:
            reproduction_time = np.random.randint(1, B)
            energy_level = np.random.randint(1, E)
            grid_sharks[x][y] = [reproduction_time, energy_level]
            shark_made = True

n_fish_list = []
n_sharks_list = []

for t in range(T):
    if not (t%1):
        plot(grid_fish, grid_sharks, t, name)
    if not (t%50):
        print(t/T*100, "%\t t = ", t)
    grid_fish, grid_sharks, number_of_fish, number_of_sharks = evolution(grid_fish, grid_sharks)
    n_fish_list.append(number_of_fish)
    n_sharks_list.append(number_of_sharks)
    
# plot_phase_space(n_fish_list, n_sharks_list)

#%%
plot_in_time(n_fish_list, n_sharks_list, 1)
# plot_phase_space(n_fish_list, n_sharks_list)

#%%
gif(name, T)



