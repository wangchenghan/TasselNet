import os
import glob
import numpy as np
from skimage import io, img_as_ubyte
from skimage.transform import resize 
from utils import *

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

    def generate_image_and_density(self, size=None):
        for image_path, density_map in self.file_path_list:
            image = io.imread(image_path)
            density_map = np.load(density_map) if density_map else None
            if size and len(size) == 2:
                image = img_as_ubyte(resize(image, size))
                if density_map:
                    density_map = desity_map_resize_v3(density_map, size)
            yield image, density_map

    def generate_slice_and_labels(self, window_size=31, batch_size=4, size=None):
        image_batch = np.zeros((batch_size, window_size, window_size, image.shape[2]), np.uint8)
        ground_truth_batch = np.zeros((batch_size, np.float))
        batch_index = 0
        for image, density_map in self.generate_image_and_density(size):
            if density_map and (density_map.shape[0] != image.shape[0] or density_map.shape[1] != image.shape[1]):
                raise Exception("图像的尺寸和密度图的尺寸不匹配，请重新生成，或在generate_slice_and_labels方法中指定一个新的尺寸（size）")
            if window_size % 2 == 0:
                raise Exception("窗口大小应该是奇数")
            pad_size = window_size // 2
            result = []
            if len(image.shape) < 3:
                padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
            else:
                padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')
            if density_map: 
                padded_density_map = np.pad(density_map, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        cropped_image = np.array(padded_image[i: i + window_size, j: j + window_size])
                        ground_truth_value = padded_density_map[(2 * i + window_size) // 2, (2 * j + window_size) // 2]
                        image_batch[batch_index] = cropped_image
                        ground_truth_batch[batch_index] = ground_truth_value
                        batch_index += 1
                        if batch_index >= batch_size:
                            yield image_batch, ground_truth_batch
                            batch_index = 0
            else: 
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        cropped_image = np.array(padded_image[i: i + window_size, j: j + window_size])
                        image_batch[batch_index] = cropped_image
                        ground_truth_batch[batch_index] = -1
                        batch_index += 1
                        if batch_index >= batch_size:
                            yield image_batch, ground_truth_batch
                            batch_index = 0
