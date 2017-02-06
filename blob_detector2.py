#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 11:31:47 2017
### Blob detector 2
@author: terencephilippon
"""

import cv2
from cv2 import *
import numpy as np
import os
import imutils

####################################################
if os.name == 'posix':      
    homepath = os.environ['HOME']
else: homepath = os.environ['HOMEPATH']

if os.name == 'nt':    
    # WINDOWS PATH (to video)     
    path = homepath+"\\Videos\\campus1.mp4"
else:
    # MAC OS PATH (to video)
    path = homepath+'/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4'
####################################################

# Kernels definition.
kernel0 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16,16))

#==============================================================================
#  setting video to read.
#==============================================================================
cap = cv2.VideoCapture('/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4')
# Choosing start frame.
frame_start = 250
cap.set(1, frame_start)
fgbg = cv2.BackgroundSubtractorMOG()

#==============================================================================
# Setup SimpleBlobDetector parameters.
#==============================================================================
params = cv2.SimpleBlobDetector_Params()

# Filter by Area
params.filterByArea = True
params.minArea = 5000
params.maxArea = 120000

# Filter by Circularity
params.filterByCircularity = False

# Filter by Convexity
params.filterByConvexity = False

# Filter by Inertia
params.filterByInertia = False

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector(params)

#==============================================================================
# Looping over the video.
#==============================================================================
while True:
    (grabbed, frame) = cap.read()
    
    frame = imutils.resize(frame, width=500)     # If resize is needed
    fgmask = fgbg.apply(frame)
    
    # ===========================================================
    # Morpho math
#    erosion = cv2.erode(fgmask, kernel0, iterations = 1)
#    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel1)
    dilation = cv2.dilate(fgmask, kernel2, iterations = 3)
    erosion = cv2.erode(dilation, kernel2, iterations = 1)
#    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel3)
    img = erosion
    img = (255-img)
    # ===========================================================
    # Tresholding
#    thresh = 0
#    maxValue = 255
#    th, dst = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
    # ===========================================================
    # Finding contours
#    contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#    cv2.drawContours(frame,contours,-1,(0,255,0),2)

    # Detect blobs.
    keypoints = detector.detect(img)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    # draw the text with number of people détected on the frame.
#    number = len(contours)
#    cv2.putText(frame, "Counting : "+str(number),
#        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    # ===========================================================
    # Plotting and saving
    cv2.imshow('frame', im_with_keypoints)
#    cv2.imwrite(homepath+"\\Videos\\MOG_backgroundsub\\"+"frame_"+str(nbframe)+".png", frame)
    # ===========================================================
#    nbframe += 1
    # Breaking and ending if escape is pressed.
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        print('interrupting...')
        break
    

print('end')

################# Archives ####################

#    # Convert image in grayscale for the blob detector.
#    frame = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
