# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:15:57 2022

@author: Wahbi K. El-Bouri
"""

#
# Code for Infarct Core Segmentation Using Vitrea - Kausik Chaterjee Collaboration
import argparse
import time

import copy
import numpy as np
import matplotlib.pyplot as plt
import sys

from data_loading import dataLoader, reshape_images
from model import Clusterer
import constants
from inference import Inferer

TRAIN_DIR = r"C:\Users\wahbi\OneDrive - The University of Liverpool\Infarct Core Segmentation Project\Vitrea\Test1\DIACOM\WSDTI1R4\SC3I4ZVV"
TEST_DIR = r"C:\Users\wahbi\OneDrive - The University of Liverpool\Infarct Core Segmentation Project\Vitrea\Test1\DIACOM\WSDTI1R4\SC3I4ZVV"

# TRAIN_DIR = r"C:\Users\wahbi\OneDrive - The University of Liverpool\Infarct Core Segmentation Project\Vitrea\Test2\DIACOM\DOOBF10L\OGWZ55UX"
# TEST_DIR = TRAIN_DIR

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_images', default=5000, type=int)
    parser.add_argument('--num_test_images', default=5000, type=int)
    parser.add_argument('--num_clusters', default=12, type=int)
    parser.add_argument('--clustering_method', default='KNN', type=str)
    args = parser.parse_args()
    return args

def main(args):
    loader = dataLoader(TRAIN_DIR, num_images=args.num_train_images)
    loader.load_in_images()
    loader.format_images()
    train_x, test_x = loader.train_test_split()
    print("Data loaded and split into training and testing sets")

    clust = Clusterer(num_clusters=args.num_clusters, clustering_method=args.clustering_method)
    clust.fit(train_x)
    pixel_labels = clust.predict(train_x)
    print("Clustering algorithm initialised and trained and labels predicted")
    pixel_labels = reshape_images(pixel_labels)

    inferer = Inferer(clust, dir=TEST_DIR, num_images=args.num_test_images)
    seg_images = inferer.save_inferred_plots()
    
    return pixel_labels, clust, seg_images

if __name__=='__main__':
    num_clusters = 12
    t = time.time()
    sys.path.append(TRAIN_DIR)
    pixel_lables, clust, seg_images = main(parse_args())
    print('Elapsed time: ' + str(time.time()-t) +' s')
    
    mask = np.zeros(num_clusters)
    for j, image in enumerate(seg_images):
        for k in range(num_clusters):
            mask[k] = mask[k] + np.sum(np.sum((image==k)*1))
    
    # get into cm3
    # each pixel = 0.25mm3
    mask_dummy = copy.deepcopy(mask)
    mask_dummy = mask*0.25/1000
    TEST2 = 68 #pixels for 3 cm
    TEST1 = 64 #pixels for 3cm
    
    # plots
    i = 10
    plt.imshow(seg_images[i], cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 0)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 1)*1, cmap=plt.cm.bone) #b
    plt.imshow((seg_images[i] == 2)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 3)*1, cmap=plt.cm.bone) #b
    plt.imshow((seg_images[i] == 4)*1, cmap=plt.cm.bone) #b
    plt.imshow((seg_images[i] == 5)*1, cmap=plt.cm.bone) #b
    plt.imshow((seg_images[i] == 6)*1, cmap=plt.cm.bone)#b
    plt.imshow((seg_images[i] == 7)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 8)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 9)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 10)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 11)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 12)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 13)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 14)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 15)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 16)*1, cmap=plt.cm.bone)
    plt.imshow((seg_images[i] == 17)*1, cmap=plt.cm.bone)
