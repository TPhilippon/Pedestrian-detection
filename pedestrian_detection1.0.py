# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:43:56 2016

@author: terencephilippon
"""

# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import time
import datetime
import sys, os, select

print("starting...")

# ==================================================================
########## initialize the HOG descriptor/person detector ###########
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ===================================================================
################### Setting data and path ###########################
if os.name == 'posix':
    homepath = os.environ['HOME']
else: homepath = os.environ['HOMEPATH']

if os.name == 'nt':
    path = homepath+"\\Videos\\campus4c1.mp4"
else:
    path = homepath+'/Users/terencephilippon/Python/VIDEO/Video Data/campus4c1.mp4'

# ===================================================================
### Read from the video camera (Mac OS)
#camera = cv2.VideoCapture('/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4')

### Read from video or camera (Windows)
camera = cv2.VideoCapture(path)

# Choosing start frame.
camera.set(1, 200);

# initialize the first frame in the video stream.
firstFrame = None
# ==================================================================
##################### Loop on the frames of video ##################
while True:
    # grab the current frame and initialize the occupied/unoccupied
    (grabbed, frame) = camera.read()
    # if the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        break
    
    # Resize if needed, then detect people on frame...
    frame = imutils.resize(frame, width=500)
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8,8),
    		padding=(24, 24), scale=1.15)
    
    # ==============================================================
    ### draw the original bounding boxes 
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    # Create array containing rectangles coordinates.
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    
    # ==============================================================
    ### If needed : apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people (if too much false detections appears).
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes (use rects or pick if non-maxima are applied).
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        
    # ==============================================================
    ### draw the text with number of people détected on the frame.
    number = len(pick)
    cv2.putText(frame, "People detected : "+str(number),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # Display image of record + boxes + text counting people.
#    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", frame)
    
    # ================================================================
    # Wait for key to be pressed if needed to stop the process.
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        camera.release()
        cv2.destroyAllWindows()
        print("interruption.")
        sys.exit()
    # Record key press and break if hit "esc".

print("end.")