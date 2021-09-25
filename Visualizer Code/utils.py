import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import algorithms
from scipy.ndimage import zoom
from skimage.transform import resize
from PIL import Image
import caffe
import cv2 as cv
import math
import random
import pickle
def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = skimage.img_as_float(skimage.io.imread(filename, as_gray=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img

def net_prediction(net, image, backward = False, target = None):
    image_mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', image_mean)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    predictions = output['prob'][0]
    print(predictions)
    if backward:
        if not target: diffs = net.backward()
        else: diffs = net.backward([target])
        return predictions, diffs
    return predictions

def normalizer(arr):
    return (arr-np.amin(arr))/(np.amax(arr) - np.amin(arr))

def scale_normalize(arr):
    map = np.ndarray.flatten(arr)
    sort_map = np.sort(map)
    ret = np.zeros(arr.shape)
    num = map.shape[0]
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ret[i][j] = np.searchsorted(sort_map, arr[i][j])/num
    return ret

def visualizer(map_orig, lower_limit = 0, upper_limit = 1):
    map = np.copy(map_orig)
    map = normalizer(map)
    map[map < lower_limit] = 0
    map[map < upper_limit] = 0
    plt.title('Lower Limit: '+str(lower_limit) + 'Upper Limit: '+ str(upper_limit))
    plt.imshow(map, cmap='jet', interpolation='None', alpha=1)
    plt.show()
    plt.close('all')

def grad_histogram(arr, interval = 0.02, count = True):
    flat_arr = np.ndarray.flatten(arr)
    print("Mean:", str(np.mean(flat_arr)))
    print("Standard Deviation:", str(np.std(flat_arr)))
    x_index = interval
    if count:
        x_index = np.arange(0, 1, interval)
        x_index[0] = -0.000001
        print(x_index)
        y_index = []
        for x in x_index:
            y_index.append(((flat_arr > x)&(flat_arr <= x+interval)).sum())
        print(y_index)
    plt.hist(flat_arr, x_index)
    plt.show()
    plt.close('all')