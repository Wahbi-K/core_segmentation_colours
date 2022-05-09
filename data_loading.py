import os

from dataclasses import dataclass

from pydicom import dcmread
from pydicom.data import get_testdata_file

@dataclass
class pixelBuffer:
        buffer: list
        index: int = -1

        def __post_init__(self):
            self.buffer = []

        def add_data(self, pixel_item):
            self.buffer.append(pixel_item)

        def __iter__(self):
            return self

        def __next__(self):
            self.index += 1
            return self.buffer[self.index]


class dataLoader:

    def __init__(self, dir, num_images):
        self.dir = dir
        self.num_images = num_images

    def load_in_images(self):

        file_list = os.listdir(self.dir)
        num_files_in_dir = len([filename for filename in file_list if os.path.isfile(os.path.join(self.dir,filename))])
        limit = self.num_images if self.num_images <= num_files_in_dir else num_files_in_dir

        img_count = 0
        pixel_list=pixelBuffer(buffer=[])
        for filename in file_list:
            ds = dcmread(os.path.join(self.dir, filename))
            pixel_list.add_data(ds.pixel_array)

            img_count+=1
            if img_count >= limit:
                break

        return pixel_list

if __name__ == "__main__":

    dir = r"C:/Users/rashe/Downloads/segmentation_test_data"
    dl = dataLoader(dir, 3)
    dl.load_in_images()
