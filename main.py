# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:15:57 2022

@author: Wahbi K. El-Bouri
"""

# Code for Infarct Core Segmentation Using Vitrea - Kausik Chaterjee Collaboration
import argparse

import numpy as np
import matplotlib.pyplot as plt
import sys

from data_loading import dataLoader, reshape_images
from model import Clusterer
import constants
from inference import Inferer

TRAIN_DIR = r"C:/Users/rashe/Downloads/segmentation_test_data"
TEST_DIR = r"C:/Users/rashe/Downloads/segmentation_test_data"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', default=10, type=int)
    parser.add_argument('--num_clusters', default=10, type=int)
    parser.add_argument('--clustering_method', default='KNN', type=str)
    args = parser.parse_args()
    return args

def main(args):
    loader = dataLoader(TRAIN_DIR, num_images=args.num_images)
    loader.load_in_images()
    loader.format_images()
    train_x, test_x = loader.train_test_split()
    print("Data loaded and split into training and testing sets")

    clust = Clusterer(num_clusters=args.num_clusters, clustering_method=args.clustering_method)
    clust.fit(train_x)
    pixel_labels = clust.predict(train_x)
    print("Clustering algorithm initialised and trained and labels predicted")
    pixel_labels = reshape_images(pixel_labels)

    inferer = Inferer(clust, dir=TEST_DIR, num_images=50)
    inferer.save_inferred_plots()

if __name__=='__main__':

    sys.path.append(TRAIN_DIR)
    main(parse_args())
