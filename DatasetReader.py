# -- coding:utf-8 --
import os
import glob
import numpy as np
from skimage import io, img_as_ubyte
from skimage.transform import resize 
from utils import *

class DatasetReader:
    def __init__(self, image_directory_path, density_map_directory_path=''):
        self.image_directory_path = image_directory_path
        self.density_map_directory_path = density_map_directory_path

        IMAGE_FORMAT = ['jpg', 'JPG', 'PNG', 'png', 'bmp', 'BMP']

        self.file_path_list = []
        self.image_name_list = []
        for image_format in IMAGE_FORMAT:
            exists_images = glob.glob(os.path.join(self.image_directory_path, '*.' + image_format))
            for image_path in exists_images:
                image_name = os.path.basename(image_path).split('.')[0]
                self.image_name_list.append(image_name)
                label_name = image_name + '.npy'
                density_map_path = os.path.join(self.density_map_directory_path, label_name)
                density_map_path = density_map_path if os.path.exists(density_map_path) else ''
                self.file_path_list.append((image_path, density_map_path))

    def get_order_image_name(self):
        return self.image_name_list 

    def generate_image_and_density(self, size=None, filter_content=0):
        for image_path, density_map_path in self.file_path_list[:2]:
            image = io.imread(image_path)
            density_map = np.load(density_map_path) if density_map_path else None
            if size and len(size) == 2:
                image = img_as_ubyte(resize(image, size))
                # image = resize(image, size)
                if not(density_map is None):
                    density_map = desity_map_resize_v3(density_map, size)
            if not (density_map is None) and (density_map.shape[0] != image.shape[0] or density_map.shape[1] != image.shape[1]):
                raise Exception("图像的尺寸和密度图的尺寸不匹配，请重新生成，或在generate_slice_and_labels方法中指定一个新的尺寸（size）")
            if filter_content == 0:
                yield image, density_map
            elif filter_content == 1:
                yield image
            elif filter_content == 2:
                yield density_map
            

    def generate_slice_and_labels(self, window_size=31, batch_size=4, size=None, accumulate_area=False, filter_content=0):
        image_batch = np.zeros((batch_size, window_size, window_size, 3), np.uint8)
        ground_truth_batch = np.zeros((batch_size), np.float)
        batch_index = 0
        for image, density_map in self.generate_image_and_density(size):
            if window_size % 2 == 0:
                raise Exception("窗口大小应该是奇数")
            pad_size = window_size // 2
            result = []
            if len(image.shape) < 3:
                padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
            else:
                padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')
            if not (density_map is None): 
                padded_density_map = np.pad(density_map, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        cropped_image = np.array(padded_image[i: i + window_size, j: j + window_size])
                        if accumulate_area:
                            ground_truth_value = np.sum(padded_density_map[i: i + window_size, j: j + window_size])
                        else:
                            ground_truth_value = padded_density_map[(2 * i + window_size) // 2, (2 * j + window_size) // 2]
                        image_batch[batch_index,:,:,:] = cropped_image
                        ground_truth_batch[batch_index] = ground_truth_value
                        batch_index += 1
                        if batch_index >= batch_size:
                            if filter_content == 0:
                                yield image_batch, ground_truth_batch
                            elif filter_content == 1:
                                yield image_batch
                            elif filter_content == 2:
                                yield ground_truth_batch
                            image_batch = np.zeros((batch_size, window_size, window_size, 3), np.uint8)
                            ground_truth_batch = np.zeros((batch_size), np.float)
                            batch_index = 0
            else: 
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        cropped_image = np.array(padded_image[i: i + window_size, j: j + window_size])
                        image_batch[batch_index,:,:,:] = cropped_image
                        ground_truth_batch[batch_index] = -1
                        batch_index += 1
                        if batch_index >= batch_size:
                            if filter_content == 0:
                                yield image_batch, ground_truth_batch
                            elif filter_content == 1:
                                yield image_batch
                            elif filter_content == 2:
                                yield ground_truth_batch
                            image_batch = np.zeros((batch_size, window_size, window_size, 3), np.uint8)
                            ground_truth_batch = np.zeros((batch_size), np.float)
                            batch_index = 0

if __name__ == '__main__':
    image_path = "/Users/Biomind/Documents/周文静/TasselNet/spruce/202011/train"
    density_map_path = "/Users/Biomind/Documents/周文静/TasselNet/spruce/202011_den/train"
    size = (224, 224)
    date_reader = DatasetReader(image_path, density_map_path)
    # image_and_density_generator = date_reader.generate_image_and_density(size)
    # for i in range(1):
    #     image, density_map = image_and_density_generator.__next__()
    #     print(image)
    #     print(density_map)
    # slice_and_label_generator = date_reader.generate_slice_and_labels(31, 4, size)
    # print("#" * 100)
    # for i in range(1):
    #     slice_batch, label = slice_and_label_generator.__next__()
    #     io.imshow(slice_batch[3])
    #     # print(slice_batch)
    #     print(label)
    # slice_and_label_generator = date_reader.generate_slice_and_labels(31, 4, size, True)
    # for i in range(1):
    #     slice_batch, label = slice_and_label_generator.__next__()
    #     io.imshow(slice_batch[3])
    #     # print(slice_batch)
    #     print(label)
    slice_and_label_generator = date_reader.generate_slice_and_labels(31, 4, size, True, filter_content=2)
    for i in range(5):
        slice_batch = slice_and_label_generator.__next__()
        print(slice_batch)