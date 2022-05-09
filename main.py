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

from data_loading import dataLoader

if __name__=='__main__':

    path = r'C:/Users/rashe/Downloads/trywahbifile'

    sys.path.append(path)

    loader = dataLoader(path, num_images=50)

    #TODO: write processing unit to format the data
    #TODO: write GMM and thesholding module to classify pixels according to distro
    #TODO: write plotting module

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.pause(10)
