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
    print(dir(collection))
    objects = collection.getElementsByTagName('object')
    size = collection.getElementsByTagName('size')
    width = size.getElementsByTagName('width')
    height = size.getElementsByTagName('height')
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
    density_map = np.zeros((width, height), np.float)
    coordinates = label['coordinate']
    for coordinate in coordinates:
        xmin = coordinate[0]
        ymin = coordinate[1]
        xmax = coordinate[2]
        ymax = coordinate[3]

        diameter = np.max(xmax - xmin, ymax - ymin)
        kernel_size = np.max(xmax - xmin, ymax - ymin)
        kernel = gaussian_kernel_2d_opencv(kernel_size, 1)
        for i in range(max(xmin, 0), min(xmax, width)):
            for j in range(max(ymin, 0), min(xmin, height)):
                density_map[i][j] += kernel[i - xmin][j - ymin]
    return density_map

def label_path2density_map(label_path):
    label = read_label_xml(label_path)
    density_map = label2density(label)
    return density_map

def generate_density_dataset(file_list, output_path, slice_number=1):
    for file in file_list:
        image_path = file['image']
        label_path = file['label']

        density_map = label_path2density_map(label_path)

        image_name = os.path.basename(image_path).split('.')[0] + '.png'
        io.imsave(density_map, os.path.join(output_path, image_name))


        # for j in range(1, 4):
        #         px = 1
        #         px2 = 1
        #         for k in range(1, 4):
        #             print('global' + str(global_step))
        #         # print('j' + str(j))
        #         # print('k' +str(k))
        #         print('----------')
        #         if (global_step == 4 & j == 3 & k == 4):
        #             print('global' + str(global_step))
        #         final_image = img[py - 1: py + p_h - 1, px - 1: px + p_w - 1, :]
        #         final_gt = den_map[py2 - 1: py2 + d_map_ph - 1, px2 - 1: px2 + d_map_pw - 1]
        #         px = px + p_w
        #         px2 = px2 + d_map_pw
        #         if final_image.shape[2] < 3:
        #             final_image = np.tile(final_image, [1, 1, 3])
        #         image_final_name = out_path + mode + '_img/' 'IMG_' + str(i) + '_' + str(count) + '.jpg'
        #         #gt_final_name = out_path + mode + '_gt/' + 'GT_IMG_' + str(i) + '_' + str(count)
        #         gt_final_name = out_path + mode + '_gt/' + 'IMG_' + str(i) + '_' + str(count) + '.csv'
        #         Image.fromarray(final_image).convert('RGB').save(image_final_name)
        #         #np.save(gt_final_name, final_gt)
        #         # dataframe = pd.DataFrame(final_gt, index=False)
        #         # dataframe.to_csv("test.csv", index=False, sep=',')
        #         np.savetxt(gt_final_name, final_gt, fmt='%f', delimiter=',')
        #         #归一化至（0，1）
        #         temp = np.max(final_gt) - np.min(final_gt)
        #         nomalized_final_gt = np.clip((final_gt * 1 / temp) - (np.min(final_gt) / temp), 0, 1)
        #         # print(np.max(nomalized_final_gt), np.min(nomalized_final_gt))
        #         io.imsave(gt_final_name.split('.')[0] + '.png', nomalized_final_gt)
        #         # Image.fromarray(final_gt).convert('L').save(gt_final_name.split('.')[0] + '.png')
        #         count = count + 1
        #     py = py + p_h
        #     py2 = py2 + d_map_ph
        # global_step = global_step + 1



if __name__ == '__main__':
    img_path = 'datasets/spruce/202011/train'
    gt_path = 'datasets/spruce/202011/train'
    #train_gt = 'datasets/ShanghaiTech_Crowd_Counting_Dataset/part_A_final/train_data/ground_truth'
    out_path = 'datasets/spruce/202011_den'
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
    generate_density_dataset(validation_list, validation_output_path, N)
