# -- coding:utf-8 --
from skimage import io
from skimage.transform import resize 
from skimage.util import crop
import numpy as np

def generate_slices(image, window_size=32, slide=1):
    if window_size % 2 == 0:
        raise Exception("窗口大小应该是奇数")
    pad_size = window_size // 2
    result = []
    if len(image.shape) < 3:
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
    else:
        padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            cropped_image = np.array(padded_image[i: i + window_size, j: j + window_size])
            result.append(cropped_image)
    return result

def generate_slices_labels(label_image, window_size=32, slide=1):
    if window_size % 2 == 0:
        raise Exception("窗口大小应该是奇数")
    pad_size = window_size // 2
    result = []
    padded_image = np.pad(label_image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
    for i in range(label_image.shape[0]):
        for j in range(label_image.shape[1]):
            ground_truth_value = padded_image[(2 * i + window_size) // 2, (2 * j + window_size) // 2]
            result.append(ground_truth_value)
    return result

def desity_map_resize_v1(desity_array, size):
    # 小密度图转化可以用；大密度图转化过程会出现内存问题，中间矩阵过大
    """
    src_size = (M1, M2)
    dest_size = (N1, N2)
    process: (M1, M2) -> (M1 * N1, M2 * N2) -> (N1, N2)
    """
    if len(size) != 2:
        raise Exception("Wrong size for resize density map, only two demension is needed")
    origin_width = desity_array.shape[0]
    origin_height = desity_array.shape[1]
    resized_width = size[0]
    resized_height = size[1]
    middle_width = origin_width * resized_width
    middle_height = origin_height * resized_height

    middle_array = np.zeros((middle_width, middle_height), np.float)
    resized_array = np.zeros((resized_width, resized_height), np.float)

    for i in range(origin_width):
        for j in range(origin_height):
            middle_array[resized_width * i: resized_width * (i + 1), resized_height * j: resized_height * (j + 1)] = float(desity_array[i][j]) / ( resized_width * resized_height )
    for i in range(resized_width):
        for j in range(resized_height):
            resized_array[i][j] = np.sum(middle_array[origin_width * i: origin_width * (i + 1), origin_height * j: origin_height * (j + 1)])
    return resized_array

def desity_map_resize_v2(desity_array, size):
    # 计算量过大，也不适用
    """
    src_size = (M1, M2)
    dest_size = (N1, N2)
    process: (M1, M2) -> (N1, N2)
    """
    if len(size) != 2:
        raise Exception("Wrong size for resize density map, only two demension is needed")
    origin_width = desity_array.shape[0]
    origin_height = desity_array.shape[1]
    resized_width = size[0]
    resized_height = size[1]
    resized_array = np.zeros((resized_width, resized_height), np.float)
    for i in range(resized_width):
        for j in range(resized_height):
            for m in range(origin_width * i, origin_width * (i + 1)):
                for n in range(origin_height * j, origin_height * (j + 1)):
                    resized_array[i][j] += float(desity_array[m // resized_width][n // resized_height]) / ( resized_width * resized_height )
    return resized_array

def desity_map_resize_v3(desity_array, size):
    # 正常缩放 + 数值缩放
    # 结果和v1, v2不一致，待定，目前以v1, v2的结果为基准，不过运算很快
    """
    src_size = (M1, M2)
    dest_size = (N1, N2)
    process: (M1, M2) -> (N1, N2)
    """
    if len(size) != 2:
        raise Exception("Wrong size for resize density map, only two demension is needed")
    resized_width = size[0]
    resized_height = size[1]
    resized_array = resize(desity_array, (resized_width, resized_height))
    resized_array = resized_array * ( np.sum(desity_array) / np.sum(resized_array))
    return resized_array

def desity_map_resize_v4(desity_array, size):
    # not finished, 未完待续
    """
    src_size = (M1, M2)
    dest_size = (N1, N2)
    process: (M1, M2) -> (N1, N2)
    """
    if len(size) != 2:
        raise Exception("Wrong size for resize density map, only two demension is needed")
    origin_width = desity_array.shape[0]
    origin_height = desity_array.shape[1]
    resized_width = size[0]
    resized_height = size[1]
    resized_array = np.zeros((resized_width, resized_height), np.float)

    return resized_array

if __name__ == '__main__':
    # im = io.imread('/Users/Biomind/Documents/周文静/TasselNet/企业微信20210423035005.png')
    # # print(im)
    # part = crop(im, 500, copy=True)
    # print(im.shape)
    # padded_im = np.pad(im ,((40,40), (40,40), (0, 0)),'constant')
    # print(padded_im.shape)
    # # io.imshow(part)
    a = np.array([[1,2], [4,5]])
    b = desity_map_resize_v4(a, (3,3))
    print(np.sum(a))
    print(np.sum(b))
    print(b)