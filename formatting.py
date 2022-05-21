import numpy as np

from skimage.color import rgb2gray

class Formatter:
    """class written for formatting. Some functions are not essential here"""
    def __init__(self):
        self.format_type = None

    def format(self, array: np.array, grey=False) -> np.array:
        array_shape = array.shape
        if grey:
            return self.greyscale(array).reshape(array_shape[0]*array_shape[1], 1)
        return array.reshape(array_shape[0]*array_shape[1], array_shape[-1])

    def greyscale(self, array: np.array) -> np.array:
        return rgb2gray(array)
