# -- coding:utf-8 --
import numpy as np
import pandas as pd
import os
import matplotlib.image as mpimg
from skimage import io
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image
from skimage import io, img_as_ubyte
import glob
import cv2
import math
import random
from xml.dom import minidom

def read_label_xml(filename):
    # 使用minidom解析器打开 XML 文档
    DOMTree = minidom.parse(filename)
    collection = DOMTree.documentElement

    coordinates = []
    objects = collection.getElementsByTagName('object')
    size = collection.getElementsByTagName('size')
    width = int(size[0].getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size[0].getElementsByTagName('height')[0].childNodes[0].data)
    for single_object in objects:
        # 类别名称
        name = single_object.getElementsByTagName('name')[0].childNodes[0].data
        bndbox = single_object.getElementsByTagName('bndbox')[0]
        # 矩形坐标
        xmin = int(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data)
        coordinates.append((xmin, ymin, xmax, ymax, name))

    return {'shape': [width, height], 'coordinates': coordinates}

def gaussian_kernel_2d_opencv(kernel_size = 3,sigma = 0):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    return np.multiply(kx,np.transpose(ky))

def label2density(label):
    width, height = label['shape']
    density_map = np.zeros((height, width), np.float)
    coordinates = label['coordinates']
    for coordinate in coordinates:
        xmin = coordinate[0]
        ymin = coordinate[1]
        xmax = coordinate[2]
        ymax = coordinate[3]

        kernel_size = max(xmax - xmin, ymax - ymin)
        kernel = gaussian_kernel_2d_opencv(kernel_size, kernel_size / 10)
        # print(np.sum(kernel))
        xcenter = (xmax + xmin) // 2
        ycenter = (ymax + ymin) // 2
        half_kernel_size = kernel_size // 2
        xmin = xcenter - half_kernel_size
        xmax = xcenter + half_kernel_size
        ymin = ycenter - half_kernel_size
        ymax = ycenter + half_kernel_size
        # print('*****************************************')
        # print(kernel.shape)
        for i in range(max(xmin, 0), min(xmax, height - 1)):
            for j in range(max(ymin, 0), min(ymax, width - 1)):
                density_map[i][j] += kernel[i - xmin][j - ymin]
    return density_map

def label_path2density_map(label_path):
    label = read_label_xml(label_path)
    density_map = label2density(label)
    return density_map


def normalization(data):
    _range = np.max(data) - np.min(data)
    return 255 * ((data - np.min(data)) / _range)

def generate_density_dataset(file_list, output_path, slice_number=1):
    for file in file_list:
        image_path = file['image']
        label_path = file['label']

        density_map = label_path2density_map(label_path)

        image_name = os.path.basename(image_path).split('.')[0] + '.png'
        np.save(os.path.join(output_path, image_name.split('.')[0] + '.npy'), density_map)
        scaled_density_map = normalization(density_map).astype(np.uint8)
        io.imsave(os.path.join(output_path, image_name), scaled_density_map)
        print(image_name + ' done!')

if __name__ == '__main__':
    img_path = 'spruce/202011/train'
    gt_path = 'spruce/202011/train'
    out_path = 'spruce/202011_den'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    N = 4
    TRAIN_RATE = 0.7

    IMAGE_FORMAT = ['jpg', 'JPG', 'PNG', 'png', 'bmp', 'BMP']

    file_list = []
    for image_format in IMAGE_FORMAT:
        exists_images = glob.glob(os.path.join(img_path, '*.' + image_format))
        for image_path in exists_images:
            image_name = os.path.basename(image_path).split('.')[0]
            label_name = image_name + '.xml'
            label_path = os.path.join(gt_path, label_name)
            if os.path.exists(label_path):
                file_list.append({'image': image_path, 'label': label_path})

    length = len(file_list)
    print('共有图像及标注：' + str(length) + '组')
    random.shuffle(file_list)

    num_of_train = int(TRAIN_RATE * len(file_list))
    train_list = file_list[:num_of_train]
    validation_list = file_list[num_of_train:]

    train_output_path = os.path.join(out_path, 'train')
    validation_output_path = os.path.join(out_path, 'val')
    print('训练集保存位置：' + train_output_path)
    print('验证集保存位置：' + validation_output_path)

    generate_density_dataset(train_list, train_output_path, N)
    # generate_density_dataset(validation_list, validation_output_path, N)
