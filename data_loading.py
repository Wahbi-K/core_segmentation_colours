import os
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from pydicom import dcmread
from pydicom.data import get_testdata_file

from formatting import Formatter
import constants

@dataclass
class pixelBuffer:
    """Dataclass to store pixels in buffer-like objects"""
    buffer: list #kept this argument as we may want to add to an exisitng buffer
    index: int = -1

    def __post_init__(self):
        self.buffer = []

    def add_data(self, pixel_item: np.array) -> None:
        """Simply appends a data point to the data buffer being constructed
        """
        self.buffer.append(pixel_item)

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        return self.buffer[self.index]

    def __len__(self):
        return len(self.buffer)


class dataLoader(Formatter):
    """Class to load in pixel datasets to conduct segmentation on the images"""

    def __init__(self, dir: str, num_images: int):
        super().__init__()
        self.dir = dir
        self.num_images = num_images
        self.pixel_buffer = pixelBuffer(buffer=[]) #initialise pixel buffer

    def load_in_images(self) -> None:
        """method that allows us to load in the pixels as independent data
        points and treat them as iid. This allows for them to be shuffled and
        trained on by separating into training and test sets."""

        file_list = os.listdir(self.dir)
        num_files_in_dir = len([filename for filename in file_list if os.path.isfile(os.path.join(self.dir,filename))])
        #The smaller of the number specified or number of images that exist in file is selected
        limit = self.num_images if self.num_images <= num_files_in_dir else num_files_in_dir

        img_count = 0
        for filename in file_list:
            if filename.startswith('I'):
                ds = dcmread(os.path.join(self.dir, filename))
                self.pixel_buffer.add_data(ds.pixel_array)
    
                img_count+=1
                if img_count >= limit:
                    self.file_list = file_list[:limit]
                    break

        #File names are saved out so we can match these up again when we segment them
        self.file_list = file_list[:img_count]
        #Keep the actual number of images in the object as opposed to the specified
        self.num_images = img_count
        
    def transform_to_hu(medical_image, image):
        intercept = medical_image.RescaleIntercept
        slope = medical_image.RescaleSlope
        hu_image = image * slope + intercept

        return hu_image
    
    def window_image(image, window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
    
        return window_image

    def format_images(self) -> None:
        """flatten the images for training"""
        self.pixel_buffer.format_buffer = np.vstack([self.format(array, grey=False) for array in self.pixel_buffer.buffer])

    def train_test_split(self, split_ratio: float = 0.7) -> (np.array, np.array):
        """split pixels into train and test sets"""

        #This logic is to ensure that we keep full images in training sets but not necessary
        num_full_images = np.round((split_ratio*len(self.pixel_buffer.format_buffer))/constants.PIXEL_RESOLUTION[0]**2)
        train_size = num_full_images*constants.PIXEL_RESOLUTION[0]**2

        train_x, test_x = train_test_split(self.pixel_buffer.format_buffer, train_size=int(train_size), shuffle=True)
        return train_x, test_x

def reshape_images(data: np.array) -> list:
    """method that reconstructs the segmented image from the predictions"""
    resolution = constants.PIXEL_RESOLUTION[0]*constants.PIXEL_RESOLUTION[1]
    num_imgs = int(len(data)/resolution)
    segmented_image = []
    for n in range(num_imgs):
        segmented_image.append(data[resolution*n:resolution*(n+1)].reshape(constants.PIXEL_RESOLUTION[0],constants.PIXEL_RESOLUTION[1]))

    return segmented_image
