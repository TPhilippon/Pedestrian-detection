#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:02:42 2017

@author: terencephilippon
"""

import cv2
import numpy as np
import sys, os, select

print("starting...")

# ===================================================================
################### Setting data and path ###########################
if os.name == 'posix':
    homepath = os.environ['HOME']
else: homepath = os.environ['HOMEPATH']

if os.name == 'nt':
    path = homepath+"\\Videos\\campus4c1.mp4"
else:
    path = homepath+'/Desktop/carTracking/'

# =================================================================== 

face_cascade = cv2.CascadeClassifier(path+'cars.xml')
vc = cv2.VideoCapture(path+'road.mp4')
 
if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False
 
while rval:
    rval, frame = vc.read()
 
    # car detection.
    cars = face_cascade.detectMultiScale(frame, 1.1, 2)
 
    ncars = 0
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        ncars = ncars + 1
 
    # show result
    cv2.imshow("Result",frame)
    cv2.waitKey(1);
vc.release()




