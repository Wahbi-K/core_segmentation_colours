# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:15:57 2022

@author: Wahbi K. El-Bouri
"""

# Code for Infarct Core Segmentation Using Vitrea - Kausik Chaterjee Collaboration

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_file
import sys
import cv2

if __name__=='__main__':

    path = r'C:\Users\wahbi\OneDrive - The University of Liverpool\Infarct Core Segmentation Project\Vitrea\Test1\DIACOM\WSDTI1R4\SC3I4ZVV'
    
    sys.path.append(path)
    
    # Try for 1 image to start with - can loop this later or do it in parallel
    ds = pydicom.dcmread(path+ "\I1090000")
    
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 