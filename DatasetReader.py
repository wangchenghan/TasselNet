import os
import glob
import numpy as np
from skimage import io

class DatasetReader:
    def __init__(self, image_path, density_path):
        self.image_path = image_path
        self.density_path = density_path
                
        IMAGE_FORMAT = ['jpg', 'JPG', 'PNG', 'png', 'bmp', 'BMP']

        self.file_path_list = []
        for image_format in IMAGE_FORMAT:
            exists_images = glob.glob(os.path.join(self.image_path, '*.' + image_format))
            for image_path in exists_images:
                image_name = os.path.basename(image_path).split('.')[0]
                label_name = image_name + '.npy'
                density_path = os.path.join(self.density_path, label_name)
                density_path = density_path if os.path.exsist(density_path) else ''
                self.file_list.append({"image_path": image_path, "density_path": density_path})

    def generate_image_and_density(self):
        for image_path, density_map in self.file_path_list:
            image = io.imread(image_path)
            density_map = np.load(density_map) if density_map else None
            yield image, density_map

    def generate_slice_and_labels(self, window_size=31, batch_size=4):
        for image, density_map in self.generate_image_and_density():
            if window_size % 2 == 0:
                raise Exception("窗口大小应该是奇数")
            pad_size = window_size // 2
            result = []
            if len(image.shape) < 3:
                padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
            else:
                padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')
            padded_density_map = np.pad(density_map, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
            for i in range(density_map.shape[0]):
                for j in range(density_map.shape[1]):
                    ground_truth_value = padded_image[(2 * i + window_size) // 2, (2 * j + window_size) // 2]