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

# =============================================================================
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
    path = homepath+'/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4'

# ===================================================================
# Set ROI (Region of interest)


### Read from the video camera (Mac OS)
camera = cv2.VideoCapture('/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4')
### Read from video or camera (Windows)
#camera = cv2.VideoCapture(path)

# Choosing start frame.
camera.set(1, 365);

# initialize the first frame in the video stream.
firstFrame = None
line_counter = 0
prevXp = 250
#==============================================================================
# ##################### Loop on the frames of video ##################
#==============================================================================
while True:
    # grab the current frame and initialize the occupied/unoccupied
    (grabbed, frame) = camera.read()
    # if the frame could not be grabbed, then we have reached the end of the video
    if not grabbed:
        break
    
    # Resize if needed, then detect people on frame...
    frame = imutils.resize(frame, width=500)
    fullframe = frame
    
    #### ROI ####
    # Draw region of interest (ROI) and draw rectangle
    frame = frame[100:350,50:450]
#    cv2.rectangle(fullframe,(100,150),(400,350),(255,0,0),2)
    cv2.rectangle(fullframe,(50,100),(450,350),(255,0,0),2)
    # Draw line
    cv2.line(fullframe,(250,400),(250,100),(255,0,0),1)
    # HOG
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8,8),
    		padding=(16, 16), scale=1.10)
    
#     draw the original bounding boxes 
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    # Create array containing rectangles coordinates.
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    
#==============================================================================
#     ### If needed : apply non-maxima suppression to the bounding boxes using a
#     fairly large overlap threshold to try to maintain overlapping
#     boxes that are still people (if too much false detections appears).
#==============================================================================
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes (use rects or pick if non-maxima are applied).
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
#        xA, xB, yA, yB = int(xA), int(xB), int(yA), int(yB)
        cv2.circle(frame, ((xA+xB)/2, ((yA+yB)/2)-10), 0, (0,0,255), 3)
        Xp = 50+(xA+xB)/2
#        print(Xpoint)
        delta = Xp-prevXp
        if Xp >= 250 and prevXp < 250 and delta <= 5:
            line_counter += 1
        prevXp = Xp
    # ==============================================================
    ### draw the text with number of people détected on the frame.
    number = len(pick)
    cv2.circle(frame, (390, 240), number*3, (0,0,255), -1)
    cv2.putText(fullframe, "People detected : "+str(number),
                (10, fullframe.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(fullframe, "Sens --> : "+str(line_counter),
                (20, fullframe.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    # ==============================================================
    # Display image of record + boxes + text counting people.
#    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", fullframe)
    prevpick = pick
    # ================================================================
    # Wait for key to be pressed if needed to stop the process.
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        camera.release()
        cv2.destroyAllWindows()
        for i in range (1,5):
            print(i)
        print("interruption.")
        sys.exit()
    # Record key press and break if hit "esc".

print("end.")




# ARCHIVES
    # Moment and centroïd
#    if not pick:
#        pass
#    else:
#        for obj in pick:
#            nppick = np.asarray(pick)
#            nppick = obj
#            M = cv2.moments(nppick)
#            cx = int(M['m10']/M['m00'])
#            cy = int(M['m01']/M['m00'])
#            cv2.line(fullframe,cx,cy,(0,0,255),2)
#            cv2.circle(fullframe, cx, cy, (0,0,255),2)