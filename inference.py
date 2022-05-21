import os
import numpy as np
import matplotlib.pyplot as plt

from data_loading import dataLoader, reshape_images

class Inferer:
    """class that allows us to carry out segmentations of test images"""
    def __init__(self, model, dir: str, num_images: int = 50):
        self.trained_model = model
        self.dir = dir
        self.num_images = num_images
        self.load_in_test_images()

    def load_in_test_images(self) -> None:
        """load in test images"""
        loader = dataLoader(self.dir, num_images = self.num_images)
        loader.load_in_images()
        loader.format_images()
        self.loader = loader

    def infer(self) -> np.array:
        """predict pixel membership"""
        pixel_labels = self.trained_model.predict(self.loader.pixel_buffer.format_buffer)
        return pixel_labels

    def construct_segmented_images(self) -> list:
        return reshape_images(self.infer())

    def save_inferred_plots(self) -> None:
        """saves out inferred images as png files"""
        results_folder = os.path.join(self.dir, "segmented_image")
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)

        for image, filename in zip(self.construct_segmented_images(), self.loader.file_list):
            plt.imshow(image, cmap=plt.cm.bone)
            plt.savefig(os.path.join(results_folder, filename+"_segmented.png"), format='png')
            plt.close()
