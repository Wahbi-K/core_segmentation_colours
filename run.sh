#!/bin/bash

python main.py \
      --num_train_images 10 \
      --num_test_images 10 \
      --num_clusters 7 \
      --clustering_method 'KNN'
