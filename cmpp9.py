# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:23:41 2023

@author: filip
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

np.random.seed(int(datetime.now().timestamp()))

'''
def make_grid(Nx, Ny):
    grid_pheromones_forage = np.zeros((Nx, Ny))
    grid_pheromones_home = np.zeros((Nx, Ny))
    grid_food = np.zeros((Nx, Ny))
    for i in range(10, 20):
        for j in range(5, 15):
            grid_food[i][j] = 200
        for j in range(60, 65):
            grid_food[i][j] = 50
    for i in range(60, 70):
        for j in range(5, 15):
            grid_food[i][j] = 100
    nest_size = 5
    for i in range(int(3*Nx/4 - nest_size), int(3*Nx/4 + nest_size)):
        for j in range(int(3*Ny/4 - nest_size), int(3*Ny/4 + nest_size)):
            grid_food[i][j] = -100
            
    grid_ants = np.zeros((Nx, Ny))
    return grid_food, grid_pheromones_forage, grid_pheromones_home, grid_ants
'''

class Ant:
    directions = [[-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0]]
    n = 81
    Nx, Ny = n, n
#    home_coordinates = np.array((int(3*Nx/4), int(3*Ny/4)))

    def __init__(self, home_coordinates):
        self.coordinates = home_coordinates
        index = np.random.randint(0, 8)
#        print("index = ", index)
        self.velocity_index = index
        self.velocity = np.array(self.directions[index])
        self.pheromone_forage_level = 1
        self.have_food = False
    
    def is_food(self, grid_food):
        if grid_food[self.coordinates[0]][self.coordinates[1]] > 0:
#            print(grid_food[self.coordinates[0]][self.coordinates[1]])
            if not self.have_food:
                grid_food[self.coordinates[0]][self.coordinates[1]] -= 25
#           print('food')
            self.pheromone_home_level=1
            self.have_food = True
            return True
        else:
            return False
        
    def is_obstacle(self, grid_obstacles):
        if grid_obstacles[self.coordinates[0]][self.coordinates[1]]:
            return True
        else:
            return False
    
    def is_home(self, grid_food, AntsPit):
        if grid_food[self.coordinates[0]][self.coordinates[1]] < 0:
            self.pheromone_forage_level=1
            if self.have_food:
                AntsPit.brought_food += 1
            self.have_food = False
            return True
        else:
            return False
    
    def probability(self, I, h, alpha):
#        return (h+I)**alpha
        return I

    def search_path(self, grid_pheromones_forage, grid_pheromones_home, grid_food):
        if self.have_food:
            grid_pheromones = grid_pheromones_forage
        else:
            grid_pheromones = grid_pheromones_home
            
        probs = []
        indices = []
        for i in range(3):
            index = (self.velocity_index + i - 1)%8
            indices.append(index)
            coordinates = self.coordinates + np.array(self.directions[index])
            if grid_food[coordinates[0]][coordinates[1]] > 0:
                return index 
            I = float(grid_pheromones[coordinates[0]][coordinates[1]])
            h, alpha = 2., 4.
            probs.append(self.probability(I, h, alpha))
        probs = np.array(probs)
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs /= prob_sum
        else:
            probs = np.array((1/3, 1/3, 1/3))
        
        ind = np.random.choice(indices, 1, p = probs)[0]
        return ind
        
    
    def leave_pheromone(self, grid_pheromones_forage, grid_pheromones_home):
        if self.have_food:
            if self.pheromone_home_level > 0:
                grid_pheromones_home[self.coordinates[0]][self.coordinates[1]] += self.pheromone_home_level
                self.pheromone_home_level *= 0.95
        else:
            if self.pheromone_forage_level > 0:
                grid_pheromones_forage[self.coordinates[0]][self.coordinates[1]] += self.pheromone_forage_level
                self.pheromone_forage_level *= 0.95

    def move_ant(self, AntsPit):
        grid_pheromones_forage = AntsPit.grid_pheromones_forage
        grid_pheromones_home = AntsPit.grid_pheromones_home
        grid_food = AntsPit.grid_food
        grid_obstacles = AntsPit.grid_obstacles
        self.leave_pheromone(grid_pheromones_forage, grid_pheromones_home)
        ind = self.search_path(grid_pheromones_forage, grid_pheromones_home, grid_food)
        self.is_home(grid_food, AntsPit)
        if np.any(np.floor_divide(self.coordinates + np.array(self.directions[ind]), 79)) or self.is_food(grid_food) or self.is_obstacle(grid_obstacles):
            self.coordinates = self.coordinates + np.array(self.directions[ (ind + 4) % 8 ])
            self.velocity_index = (ind + 4) % 8
        else:
            self.coordinates = self.coordinates + np.array(self.directions[ind])
            self.velocity_index = ind
        
    def get_coordinates(self):
        return self.coordinates
    

class AntsPit:
    def __init__(self, Nx, Ny):        
        self.grid_pheromones_forage = np.zeros((Nx, Ny))
        self.grid_pheromones_home = np.zeros((Nx, Ny))
        self.grid_food = np.zeros((Nx, Ny))
        for i in range(10, 20):
            for j in range(5, 15):
                self.grid_food[i][j] = 200
            for j in range(60, 65):
                self.grid_food[i][j] = 50
        for i in range(60, 70):
            for j in range(5, 15):
                self.grid_food[i][j] = 100
        nest_size = 5
        for i in range(int(3*Nx/4 - nest_size), int(3*Nx/4 + nest_size)):
            for j in range(int(3*Ny/4 - nest_size), int(3*Ny/4 + nest_size)):
                self.grid_food[i][j] = -100
                
        self.grid_ants = np.zeros((Nx, Ny))
        self.obstacles(Nx, Ny)
        self.brought_food = 0

    def grid_antsf(self, ants, Nx, Ny):
        self.grid_ants = np.zeros((Nx, Ny))
        for ant in ants:
            coordinates = ant.get_coordinates()
            self.grid_ants[coordinates[0]][coordinates[1]] = -100
    
    def plot(self, t):
        figure = plt.figure(dpi=200)
        ax = figure.add_subplot(111)
        a = 10
        im = ax.imshow(self.grid_food + self.grid_pheromones_forage*a + self.grid_pheromones_home*a + self.grid_ants - 200*self.grid_obstacles,
                       vmin = -200, vmax=100, cmap = 'Spectral')
    #    im = ax.imshow(grid_food)
        plt.colorbar(im)
        plt.title("t = " + str(t) + "\n food = " + str(self.brought_food))
#        plt.show()
        name = "ANTS/"
        plt.savefig(name + str(t).zfill(4))
        plt.close()
    
    def pheromone_dry(self):
        self.grid_pheromones_home *= .99
        self.grid_pheromones_forage *= .99

    def obstacles(self, Nx, Ny):
        grid_obstacles = np.zeros((Nx, Ny))
        for y in range(35, 40):
            for x in range(45, 65):
                grid_obstacles[y][x] = 1
        
        for y in range(50, 70):
            for x in range(40, 45):
                grid_obstacles[y][x] = 1
        
        self.grid_obstacles = grid_obstacles

n = 81
Nx, Ny = n, n
home_coordinates = np.array([int(3*Nx/4), int(3*Ny/4)])
#grid_food, grid_pheromones_forage, grid_pheromones_home, grid_ants0 = make_grid(Nx, Ny)

AntsPit = AntsPit(Nx, Ny)

ants = []

#%% 

time = 1500
for t in range(853, time):
    if t<100:
#        print(home_coordinates)
        ant = Ant(np.array([int(3*Nx/4), int(3*Ny/4)]))
#        print(ant.get_coordinates())5
        ants.append(ant)
#        print(t)
    for i, ant in enumerate(ants):
        ants[i].move_ant(AntsPit)
    AntsPit.pheromone_dry()
    AntsPit.grid_antsf(ants, Nx, Ny)
    AntsPit.plot(t)
#    plot_food(grid_food, t)
    if not(t % 30):
        print(t/time*100, "%", "\t frame ", t)

print("Images calculated")
#%%
import glob
from PIL import Image

name = "ANTS"
# Create the frames
frames = []
imgs = glob.glob(name+"/????.png")
a=0

print("Importing images...")

for i, image in enumerate(imgs):
    if i<855:
        a+=1
        im = Image.open(image)
        frames.append(im.copy())
        if not(i % (time//10)):
            print(i/time*100, "%", "\t frame ", i)
print(a, "images imported")

print("Creating GIF...")

frames[0].save(name+'.gif', format='GIF', disposal=2,
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=1)

print("GIF created")
