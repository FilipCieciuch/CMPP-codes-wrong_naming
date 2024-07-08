# -*- coding: utf-8 -*-
"""
Created on Mon May  8 22:51:38 2023

@author: filip
"""

import glob
from PIL import Image

from PIL import GifImagePlugin as GifPl
GifPl.LOADING_STRATEGY = GifPl.LoadingStrategy.RGB_AFTER_DIFFERENT_PALETTE_ONLY



#Convert To "P" Gif 255 color Pallette mode:
def convertToP(im):
    if im.getcolors() is not None:
        # There are 256 colors or less in this image
        p = Image.new("P", im.size)
        transparent_pixels = []
        for x in range(im.width):
            for y in range(im.height):
                pixel = im.getpixel((x, y))
                if pixel[3] == 0:
                    transparent_pixels.append((x, y))
                else:
                    color = p.palette.getcolor(pixel[:3])
                    p.putpixel((x, y), color)
        if transparent_pixels and len(p.palette.colors) < 256:
            color = (0, 0, 0)
            while color in p.palette.colors:
                print("happening")
                if color[0] < 255:
                    color = (color[0] + 1, color[1], color[2])
                else:
                    color = (color[0], color[1] + 1, color[2])
            transparency = p.palette.getcolor(color)
            p.info["transparency"] = transparency
            for x, y in transparent_pixels:
                p.putpixel((x, y), transparency)
        return p
    return im.convert("P")



name = "LBM1"
# Create the frames
frames = []
imgs = glob.glob(name+"/t?????.png")
a=0

method = Image.FASTOCTREE
colors = 250
for i in imgs:
    a+=1
    try:
        im = Image.open(i)
        pImage = im.quantize(colors=colors, method=method, dither=0)
        frames.append(pImage.copy().convert('RGBA'))
    except:
       print(f'ERROR: Unable to open {i}')

for frame in frames:
    frame.convert('RGB')

print(a)

ExportFrames = frames
# Save into a GIF file that loops forever
'''
frames[0].save(name+'.gif', format='GIF', disposal=2,
               append_images=frames[1:],
               save_all=True,
               duration=100, loop=1, optimize=False, lossless=True)
'''

'''
ExportFrames[0].save("GIF after export.gif", disposal=2, save_all=True,
                                     append_images=ExportFrames[1:], loop=0,
                                     duration=durationFrame, optimize=False, lossless=True)
'''



#Export and Save Gif Function:
def SaveAnimationFunction(ExportFrames,newfilepathname, formatname, extension, disposalID,FPS,savepath):
    durationFrame = 1000 / FPS
    if extension == ".gif":
        for i, frame in enumerate(ExportFrames):
#            print(ExportFrames.index(frame))
            ExportFrames[i]=convertToP(frame)
    ExportFrames[0].save(newfilepathname + formatname + extension, disposal=disposalID, save_all=True,
                             append_images=ExportFrames[1:], loop=0,
                             duration=durationFrame, optimize=False, lossless=True)



#Modifications can happen here.

#Set Up Export Settings:
newfilepathname = name
formatname ="animation"
extension = ".gif"
disposalID = 2
FPS = 30
savepath=('GIF before import.gif',)

#Export/ Save:
SaveAnimationFunction(ExportFrames,newfilepathname, formatname, extension, disposalID,FPS,savepath)



