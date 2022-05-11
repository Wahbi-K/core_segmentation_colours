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

from data_loading import dataLoader, reshape_images
from model import Clusterer
import constants

DIR = r"C:/Users/rashe/Downloads/segmentation_test_data"


if __name__=='__main__':


    sys.path.append(DIR)

    loader = dataLoader(DIR, num_images=50)
    loader.load_in_images()
    loader.format_images()
    train_x, test_x = loader.train_test_split()
    print("Data loaded and split into training and testing sets")

    clust = Clusterer(num_clusters = 50, clustering_method='GMM')
    clust.fit(train_x)
    pixel_labels = clust.predict(train_x)
    print("Clustering algorithm initialised and trained and labels predicted")

    pixel_labels = reshape_images(pixel_labels)

    #TODO: segmentation sucks. Figure out why it's so bad 
    print("Plotting image")
    plt.imshow(loader.pixel_buffer.buffer[0], cmap=plt.cm.bone)
    plt.pause(10)

    print("Plotting segmented image")
    plt.imshow(pixel_labels[0], cmap=plt.cm.bone)
    plt.pause(10)
