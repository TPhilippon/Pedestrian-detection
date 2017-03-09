# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:50:47 2017

@author: terencephilippon
"""

# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV 
from __future__ import print_function
import cv2
import numpy
import os
 
path = '/Users/terencephilippon/Documents/New-GithubTest-master/'

# capture frames from a video
cap = cv2.VideoCapture('/Users/terencephilippon/Documents/New-GithubTest-master/video-c.mp4')
cap.set(1, 5)

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('/Users/terencephilippon/Documents/New-GithubTest-master/cars.xml')

# loop runs if capturing has been initialized.
while True:
    # reads frames from a video
    ret, frames = cap.read()
    
    if ret == True:
         
        # convert to gray scale of each frames
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
         
     
        # Detects cars of different sizes in the input image
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)
         
        # To draw a rectangle in each cars
        for (x,y,w,h) in cars:
            cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
     
        # Display frames in a window 
        cv2.imshow('video2', frames)
         
        # Wait for Esc key to stop
        if cv2.waitKey(10) == 27:
            break
    else:
        break
 
# De-allocate any associated memory usage
cv2.destroyAllWindows()


