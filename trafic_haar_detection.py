#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:16:39 2017

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






