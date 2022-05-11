import numpy as np

class Formatter:

    def __init__(self):
        self.format_type = None

    def format(self, array: np.array) -> np.array:
        array_shape = array.shape
        return array.reshape(array_shape[0]*array_shape[1], array_shape[-1])
