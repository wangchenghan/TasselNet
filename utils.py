from skimage import io
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

def desity_map_resize(image, size):
    if len(size) != 2:
        raise Exception("wrong size for resize density map")
    width = size[0]
    height = size[1]
    

if __name__ == '__main__':
    im = io.imread('/Users/Biomind/Documents/周文静/TasselNet/企业微信20210423035005.png')
    # print(im)
    part = crop(im, 500, copy=True)
    print(im.shape)
    padded_im = np.pad(im ,((40,40), (40,40), (0, 0)),'constant')
    print(padded_im.shape)
    # io.imshow(part)