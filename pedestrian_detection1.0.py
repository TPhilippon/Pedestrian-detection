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


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --------------------
# Read from the camera
camera = cv2.VideoCapture('/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4')
#time.sleep(0.25)
camera.set(1, 500);

# initialize the first frame in the video stream
firstFrame = None
# --------------------
while True:
    # grab the current frame and initialize the occupied/unoccupied
    (grabbed, frame) = camera.read()
#    orig = frame.copy()
    
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if not grabbed:
        break
    
    # Resize, detect people...
#    frame = imutils.resize(frame, width=500)
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
    		padding=(8, 8), scale=1.05)
    
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#    pick = non_max_suppression(rects, probs=None, overlapThresh=0.25)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in rects:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)

    # draw the text and timestamp on the frame
    number = len(rects)
    cv2.putText(frame, "People detected : "+str(number),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
#    cv2.putText(frame, "ok", "org", FONT_HERSHEY_PLAIN, 1, )
#    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", frame)
#    cv2.waitKey(0)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        camera.release()
        cv2.destroyAllWindows()
        sys.exit()
    # Record key press and break if it is 'q'
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()