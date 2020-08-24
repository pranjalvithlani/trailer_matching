# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:00:28 2020

@author: Pranjal Vithlani
"""


import cv2
import os
import argparse

# dark_knight_rises_movie.mp4  236710.0
#dark_knight_rises.mp4  3122.0

# harry_potter_movie.mp4   187842.0

# python extract_frames.py --data ./data/movies/dark_knight_rises_movie.mp4 --resize --width 1280 --height 544 --nframes_per_clip 236710



parser = argparse.ArgumentParser(description='extract frames')
parser.add_argument('--data', metavar='DIR', default = './data/movies/',
                    help='path to video clip')
parser.add_argument('--save_data', metavar='DIR', default = './data/movies_data/',
                    help='path to saving frames')
parser.add_argument('--nframes_per_clip', default=8, type=int, metavar='N',
                    help='number of frames to extract per video clip'+
                    '236710 for dark knight movie'+
                    '187842 for harry potter movie')
parser.add_argument('--resize', action="store_true",
                    help = 'resize the frame and save?')
parser.add_argument('--width', default = 0, type = int,
                    help = 'width resize')
parser.add_argument('--height', default = 0, type = int,
                    help = 'height resize')
argparams = parser.parse_args()

nframes_per_clip = argparams.nframes_per_clip # 8 equally distributed frames from video
loc = argparams.data   #'./data/movies/dark_knight_rises_movie.mp4'
out = argparams.save_data
resize = False
resize = argparams.resize
width = argparams.width
height = argparams.height

if not os.path.exists(out):
    os.mkdir(out)
    
err  = []

class_id = loc.split('/')[-1]
clip_path = loc
if not os.path.exists(out + class_id[:-4] + '/'):
    os.mkdir(out + class_id[:-4] + '/')
    


print(clip_path)
print(os.listdir('./data/movies/'))
vidcap = cv2.VideoCapture(clip_path)
count = 0
vidcap.set(cv2.CAP_PROP_POS_FRAMES,125)
success,image = vidcap.read()
total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)-1
print(total_frames)
fps = vidcap.get(cv2.CAP_PROP_FPS)
print(fps)
multiplier = total_frames // nframes_per_clip
#print(multiplier)

if resize:
    
    while success:
        frameId = int(round(vidcap.get(1)))
        if frameId % multiplier == 0:
            
            output = cv2.resize(image, (width,height))
            cv2.imwrite(out + class_id[:-4]+'/'+class_id[:-4]+"_frame%06d.jpg" % count, output)  # save frame as JPEG file
            count+=1
            
        #print(count)
        if count == nframes_per_clip:
            break
        success,image = vidcap.read()
        
else:
    
    while success:
        frameId = int(round(vidcap.get(1)))
        if frameId % multiplier == 0:
            
            cv2.imwrite(out + class_id[:-4]+'/'+class_id[:-4]+"_frame%06d.jpg" % count, image)  # save frame as JPEG file            
            count+=1
            
        
        if count == nframes_per_clip:
            break
        success,image = vidcap.read()
    
    

vidcap.release()
if count == nframes_per_clip:
    print("done: "+class_id)
else:
    print("----------not done: "+class_id)
    err.append(class_id)

print(err)
    