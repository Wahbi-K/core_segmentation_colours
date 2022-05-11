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
from model import Clusterer

DIR = r"C:/Users/rashe/Downloads/segmentation_test_data"


if __name__=='__main__':


    sys.path.append(DIR)

    loader = dataLoader(DIR, num_images=50)
    loader.load_in_images()
    loader.format_images()
    train_x, test_x = loader.train_test_split()

    import pdb; pdb.set_trace()

    clust = Clusterer(clustering_method='GMM')
    clust.fit(train_x)

    #Done: write processing unit to format the data
    #TODO: write GMM and thesholding module to classify pixels according to distro
    #TODO: write plotting module

    plt.imshow(loader.pixel_buffer.buffer[0], cmap=plt.cm.bone)
    plt.pause(10)
    loader.format_images()

    plt.imshow(loader.pixel_buffer.buffer[0], cmap=plt.cm.bone)
    plt.pause(10)
