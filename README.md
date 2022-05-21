# core_segmentation_colours
 Project with KC to segment DICOM images using colour maps

You can train and infer using a model by running the command found in run.sh

You can change the number of clusters you are using and experiment with what works
best.

You also must specify how many images you want to consider. If you want to
consider all images in a folder then just put a really big number for this. For
very large datasets though this may take a long time to train. Perhaps
consider training with a smaller number of images to begin with.

The setup expects training images to be kept in the TRAIN_DIR and images to be
segmented at test time to be kept in the TEST_DIR. Change these to the relevant
paths in main.py
