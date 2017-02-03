# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:59:00 2017

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

# setting video to read.
cap = cv2.VideoCapture('/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4')
# Choosing start frame.
frame_start = 40
cap.set(1, frame_start)
fgbg = cv2.BackgroundSubtractorMOG()
#(grabbed, frame) = cap.read()
#origin = frame_start
nbframe = frame_start
# ===============================================================
# Looping over the video.
while True:
    (grabbed, frame) = cap.read()

    frame = imutils.resize(frame, width=500)     # If resize is needed
    fgmask = fgbg.apply(frame)
    
    # ===========================================================
    # Morpho math
    erosion = cv2.erode(fgmask, kernel1, iterations = 1)
#    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel3)
    dilation = cv2.dilate(erosion, kernel3, iterations = 2)
#    erosion = cv2.erode(dilation, kernel3, iterations = 6)
#    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel3)
    img = dilation
    # ===========================================================
    # Tresholding
    thresh = 0
    maxValue = 255
    th, dst = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY);
    # ===========================================================
    # Finding contours
    contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame,contours,-1,(0,255,0),2)
    
    # draw the text with number of people détected on the frame.
    number = len(contours)
    cv2.putText(frame, "Counting : "+str(number),
        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    # ===========================================================
    # Plotting and saving
    cv2.imshow('frame', frame)
#    cv2.imwrite(homepath+"\\Videos\\MOG_backgroundsub\\"+"frame_"+str(nbframe)+".png", frame)
    # ===========================================================
    nbframe += 1
    # Breaking and ending if escape is pressed.
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        print('interrupting...')
        break
    
