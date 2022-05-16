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
        buffer: list
        index: int = -1

        def __post_init__(self):
            self.buffer = []

        def add_data(self, pixel_item: np.array) -> None:
            self.buffer.append(pixel_item)

        def __iter__(self):
            return self

        def __next__(self):
            self.index += 1
            return self.buffer[self.index]

        def __len__(self):
            return len(self.buffer)


class dataLoader(Formatter):

    def __init__(self, dir: str, num_images: int):
        super().__init__()
        self.dir = dir
        self.num_images = num_images
        self.pixel_buffer = pixelBuffer(buffer=[])

    def load_in_images(self) -> None:

        file_list = os.listdir(self.dir)
        num_files_in_dir = len([filename for filename in file_list if os.path.isfile(os.path.join(self.dir,filename))])
        limit = self.num_images if self.num_images <= num_files_in_dir else num_files_in_dir

        img_count = 0
        for filename in file_list:
            ds = dcmread(os.path.join(self.dir, filename))
            self.pixel_buffer.add_data(ds.pixel_array)

            img_count+=1
            if img_count >= limit:
                break

    def format_images(self) -> None:
        self.pixel_buffer.format_buffer = np.vstack([self.format(array) for array in self.pixel_buffer.buffer])

    def train_test_split(self, split_ratio: float = 0.7) -> (np.array, np.array):
        num_full_images = np.round((split_ratio*len(self.pixel_buffer.format_buffer))/constants.PIXEL_RESOLUTION[0]**2)
        train_size = num_full_images*constants.PIXEL_RESOLUTION[0]**2
        train_x, test_x = train_test_split(self.pixel_buffer.format_buffer, train_size=int(train_size), shuffle=True)
        return train_x, test_x

def reshape_images(data: np.array) -> list:
    resolution = constants.PIXEL_RESOLUTION[0]*constants.PIXEL_RESOLUTION[1]
    num_imgs = int(len(data)/resolution)
    segmented_image = []
    for n in range(num_imgs):
        segmented_image.append(data[resolution*n:resolution*(n+1)].reshape(constants.PIXEL_RESOLUTION[0],constants.PIXEL_RESOLUTION[1]))

    return segmented_image


if __name__ == "__main__":

    dir = r"C:/Users/rashe/Downloads/segmentation_test_data"
    dl = dataLoader(dir, 1)
    import pdb; pdb.set_trace()
    dataset = dl.load_in_images()
