# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 09:38:34 2020

@author: Pranjal Vithlani
"""

import cv2
import os
from tqdm import tqdm
import glob
import numpy as np


image_folder = './data/movies_data/dark_knight_rises_movie/'
arr = np.load('ans_list.npy')
images_list = list()
for i in arr:
    images_list.append(image_folder+"dark_knight_rises_movie_frame%06d.jpg" % i)
    
video_name = 'dkr_created.avi'#save as .avi
#is changeable but maintain same h&w over all  frames
width=1280 
height=544
#this fourcc best compatible for avi
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video=cv2.VideoWriter(video_name,fourcc, 24, (width,height))



for i in tqdm(images_list):
     x=cv2.imread(i)
     video.write(x)

cv2.destroyAllWindows()
video.release()