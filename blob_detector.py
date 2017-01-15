# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 14:53:00 2017

@author: TP
"""

from __future__ import print_function
import numpy as np
import cv2
import os
import imutils
from imutils import paths

####################################################
if os.name == 'posix':      
    homepath = os.environ['HOME']
else: homepath = os.environ['HOMEPATH']

if os.name == 'nt':    
    # WINDOWS PATH (to files)     
    path = homepath+"\\Videos\\campus4-c1mp4.mp4"
else:
    # MAC OS PATH (to files)
    path = homepath+'/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4'
####################################################
# BLOB PARAMETERS =================================>
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 0;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 100
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.5
 
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)
# <==================================================
#####################################################




    # Detect blobs.
#    keypoints = detector.detect(closing)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#    im_with_keypoints = cv2.drawKeypoints(closing, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
#    cv2.matchTemplate(closing, origin, CV_TM_SQDIFF[, origin]) 