#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:29:08 2017
### Pedestrian detection 2.0
@author: terencephilippon
"""

import numpy as np
import cv2
import imutils
import os

print('starting...')
def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    cap=cv2.VideoCapture('/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4')
    while True:
        _,frame=cap.read()
        found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        draw_detections(frame,found)
        cv2.imshow('feed',frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
print('end.')
    
    
    





####### Archives

####################################################
#if os.name == 'posix':      
#    homepath = os.environ['HOME']
#else: homepath = os.environ['HOMEPATH']
#
#if os.name == 'nt':    
#    # WINDOWS PATH (to video)     
#    path = homepath+"\\Videos\\campus1.mp4"
#else:
#    # MAC OS PATH (to video)
#    path = homepath+'/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4'
####################################################

# * * *
### Read from the video camera (Mac OS)
#camera = cv2.VideoCapture('/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4')
### Read from video or camera (Windows)
#camera = cv2.VideoCapture(path)
# * * *

# Choosing start frame.
#camera.set(1, 365);