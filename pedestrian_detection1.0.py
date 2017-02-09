# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 15:43:56 2016

@author: terencephilippon
"""

# import the necessary packages
from __future__ import print_function
from matplotlib import pyplot
import matplotlib as mpl
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

### Read from the video camera (Mac OS)
camera = cv2.VideoCapture('/Users/terencephilippon/Python/VIDEO/Video Data/campus1.mp4')
### Read from video or camera (Windows)
#camera = cv2.VideoCapture(path)

# Choosing start frame.
camera.set(1, 45)

# initialize the first frame in the video stream.
firstFrame = None
line_counter = 0
frame_counter = 0
reset_counter = 0
prevXp = 250
tolerance = np.array([[3,3]])
# Array stock
#memory1 = np.zeros((1,2))
#memory2 = np.zeros((1,2))
points_stack = np.zeros((1,2))
t2 = np.zeros((1,2))
t1 = np.zeros((1,2))
points_stack_temp = np.zeros((1,2))
# initialize frame size for matrix.
(grabbed, frame) = camera.read()
matrix = np.zeros((frame.shape[0], frame.shape[1]))    # np.uint8
matrix = imutils.resize(matrix, width=500)
#camera.release()
#cv2.destroyAllWindows()

### MATPLOTLIB
# Plot matrix initialize color map and figure.
#fig = pyplot.figure(2)
#cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
#                                           ['white','orange','red'],
#                                           256)
###

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
#==============================================================================
#     # HOG
#==============================================================================
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8,8),
    		padding=(16, 16), scale=1.10)
    
#     draw the original bounding boxes (red) * Facultatif *
#    for (x, y, w, h) in rects:
#        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
 
    # Create array containing rectangles coordinates.
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    
#==============================================================================
#     ### If needed : apply non-maxima suppression to the bounding boxes using a
#     fairly large overlap threshold to try to maintain overlapping
#     boxes that are still people (if too much false detections appears).
#==============================================================================
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
#    print(pick)
    # draw the final bounding boxes (use rects or pick if non-maxima are applied).
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 1)
        
        # Set center point param. Correction on Y of the center to match feets position.
        Xcenter, Ycenter = (xA+xB)/2, (((yA+yB)/2)+50)
        point_to_add = np.array([[int(Xcenter), int(Ycenter)]])
    
        # Set pixel of matrice value + 1.
        matrix[Ycenter+100,Xcenter+50] = matrix[Ycenter+100,Xcenter+50]+1
        
        # Draw point and set param for line counter.
#        cv2.circle(frame, (Xcenter, Ycenter), 0, (0,0,255), 3)
#        Xp = 50+(xA+xB)/2
                
        # Memory and stacking.
#        center_memory = np.vstack([memory, (int(Xcenter), int(Ycenter))])
        points_stack = np.vstack([points_stack,point_to_add])
        points_stack_temp = np.vstack([points_stack_temp,point_to_add])
        t2 = points_stack_temp

#        print(Xpoint)

        # Count when object "cross" the line.
#        delta = Xp-prevXp

#        if points_temps >= 250 and prevXp < 250 and delta <= 5:
#            line_counter += 1
#        prevXp = Xp

    if t1.shape < t2.shape:
        t1.resize((t2.shape), refcheck=False)
    else:
        t2.resize((t1.shape), refcheck=False)
#    delta = t2 - t1
    t12 = np.hstack((t1,t2))
#    deltaXY = np.hstack((delta,t12))
    print(t12)
    for (x1,y1,x2,y2) in t12 :
        if (x1+50) < 250 and (x2+50) >= 250 and abs(x2-x1)<10 and abs(y2-y1)<10:
            line_counter +=1
            print(x1,y2,x2,y2)
        
        
    t1 = t2
    t2 = np.zeros((1,2))
    points_stack_temp = np.zeros((1,2))
    
#==============================================================================
#     ### draw the text with number of people détected on the frame and
#     # draw circle to the a size based on that number.
#==============================================================================
    number = len(pick)
#    cv2.circle(frame, (390, 240), number*3, (0,0,255), -1)
    cv2.putText(fullframe, "People detected : "+str(number),
                (10, fullframe.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(fullframe, "Sens --> : "+str(line_counter),
                (20, fullframe.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
#==============================================================================
#     ### Draw each point détected from the begenning.
#==============================================================================
    for XY in points_stack:
        XYx = int(XY[0])
        XYy = int(XY[1])
        cv2.circle(frame, (XYx, XYy), 1, (0,0,255), -1)
    
    # Display image of record + boxes + text counting people. Add colormap.
#    cv2.imshow("Before NMS", orig) 
    cv2.imshow("After NMS", fullframe); # to see camera frames
    cv2.imshow("Mat", matrix)
    
    # Plot on matplotlib figure.
#    img2 = pyplot.imshow(matrix,interpolation='nearest',
#                    cmap = cmap2,
#                    origin='upper')
#    pyplot.colorbar(img2,cmap=cmap2)
#    pyplot.show()
    
    # Update
#    print("frame n° : "+frame_counter)
#    print("reset counter == "+reset_counter)
    frame_counter += 1
    reset_counter += 1
    if reset_counter >= 5:
        reset_counter = 0
        memory = np.empty((1,2))
    
#    prevpick = pick
    # ================================================================
    # Wait for key to be pressed if needed to stop the process.
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        camera.release()
        cv2.destroyAllWindows()
        for i in range (1,5):
            print(i)
        print("interruption.")
        sys.exit()
    # Record key press and break if hit "esc".



#==============================================================================
# # *** end detection /// Plot of matrix /// ***
#==============================================================================


#cmaps = [('Plot')]
#nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
#gradient = np.linspace(0, 1, 256)
#gradient = np.vstack((gradient, gradient))

###
#fig = pyplot.figure(2)
#
#cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap',
#                                           ['white','orange','red'],
#                                           256)
#
#img2 = pyplot.imshow(matrix,interpolation='nearest',
#                    cmap = cmap2,
#                    origin='upper')
#
#pyplot.colorbar(img2,cmap=cmap2)
###




print("end.")




# ARCHIVES

    # color mapping
#    m_color = cv2.cvtColor(matrix, cv2.COLOR_BGR2HSV)
#    matrix = cv2.imread(matrix)
#    cv2.imwrite('matrix.png',matrix)
#    matrix_color = cv2.applyColorMap(matrix, cv2.COLORMAP_HOT)

#    matrix_grey = cv2.adaptiveThreshold(matrix, 20, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
#    matrix_grey = cv2.imread(matrix,cv2.IMREAD_GRAYSCALE)


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