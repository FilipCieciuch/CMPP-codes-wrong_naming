# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:33:04 2023

@author: filip
"""

import glob
from PIL import Image

name = "Rain2"
# Create the frames
frames = []
imgs = glob.glob(name+"/drop_??????.png")
a=0

print("Importing images...")

for i, image in enumerate(imgs):
#    if i<450:
        a+=1  
        im = Image.open(image)
        frames.append(im.copy())
print(a, "images imported")

print("Creating GIF...")

frames[0].save(name+'.gif', format='GIF', disposal=2,
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=1)

print("GIF created")
