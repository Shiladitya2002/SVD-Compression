# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
import math
import random
import pickle
#from caffe import GradientOptimizer, FindParams


# Press the green button in the gutter to run the script.

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

def normalizer(arr):
   return (arr-np.amin(arr))/(np.amax(arr) - np.amin(arr))

def scale_normalizer(arr):
   map = np.ndarray.flatten(arr)
   sort_map = np.sort(map)
   ret = np.zeros(arr.shape)
   num = map.shape[0]
   for i in range(arr.shape[0]):
       for j in range(arr.shape[1]):
           ret[i][j] = np.searchsorted(sort_map, arr[i][j])/num
   return ret

def visualizer(map_orig, cutoff = 0):
   map = np.copy(map_orig)
   map[map < cutoff] = 0
   map = normalizer(map)
   plt.title('cutoff: '+str(cutoff))
   plt.imshow(map, cmap='jet', interpolation='None', alpha=1)
   plt.show()
   plt.close('all')

def sp_noise(image, prob, ifsp):
   output = np.zeros(image.shape)
   thres = 1 - prob/2
   num = 0
   for i in range(image.shape[0]):
       for j in range(image.shape[1]):
           rdn = random.random()
           if rdn < prob/2:
               if ifsp:
                   output[i][j][0] = 0.0
                   output[i][j][1] = 0.0
                   output[i][j][2] = 0.0
               else:
                   output[i][j][0] = random.random()
                   output[i][j][1] = random.random()
                   output[i][j][2] = random.random()
           elif rdn > thres:
               if ifsp:
                   output[i][j][0] = 1.0
                   output[i][j][1] = 1.0
                   output[i][j][2] = 1.0
               else:
                   output[i][j][0] = random.random()
                   output[i][j][1] = random.random()
                   output[i][j][2] = random.random()
           else:
               #print(image[i][j])
               output[i][j][0] = image[i][j][0]
               output[i][j][1] = image[i][j][1]
               output[i][j][2] = image[i][j][2]
   #print(np.mean(output))
   return output

def multiply(DATA_SOURCE, image, mode, layer):
    for i in range(1, 9):
        layer = "group2_block" + str(i) + "_conv0_v"
        filename = DATA_SOURCE + "\\GradCAM\\" + image + "-" + mode + "\\" + layer
        with open(filename, 'rb') as fp: map1 = pickle.load(fp)
        map1 = np.absolute(map1)
        map1 = normalizer(map1)
        map = map * map1

        layer = "group2_block" + str(i) + "_conv0"
        filename = DATA_SOURCE + "\\GradCAM\\" + image + "-" + mode + "\\" + layer
        with open(filename, 'rb') as fp: map1 = pickle.load(fp)
        map1 = np.absolute(map1)
        map1 = normalizer(map1)
        map = map * map1

        layer = "group2_block" + str(i) + "_conv1_v"
        filename = DATA_SOURCE + "\\GradCAM\\" + image + "-" + mode + "\\" + layer
        with open(filename, 'rb') as fp: map1 = pickle.load(fp)
        map1 = np.absolute(map1)
        map1 = normalizer(map1)
        map = map * map1

        layer = "group2_block" + str(i) + "_conv1"
        filename = DATA_SOURCE + "\\GradCAM\\" + image + "-" + mode + "\\" + layer
        with open(filename, 'rb') as fp: map1 = pickle.load(fp)
        map1 = np.absolute(map1)
        map1 = normalizer(map1)
        map = map * map1

if __name__ == '__main__':
    UNCOMP_RES56 = "D:\\Coding Applications\\ENC-master\\init\\decomp_init\\res56\\models_res56"
    #COMP_RES56 = "D:\\Coding Applications\\ENC-master\\init\\decomp_init\\res56\\comp_res56\\deploy"

    filename = r"D:\Documents\UC Berkeley\2020-2021\Abbasi-Asl Lab\Data\SaliencyCAM\SaliencyCAM-cat_2-compressed"
    with open(filename, 'rb') as fp: map1 = pickle.load(fp)
    filename = r"D:\Documents\UC Berkeley\2020-2021\Abbasi-Asl Lab\Data\SaliencyCAM\SaliencyCAM-cat_2-uncompressed"
    with open(filename, 'rb') as fp: map2 = pickle.load(fp)
    print(np.sum(np.absolute(normalizer(map1)-normalizer(map2))))
    print(np.std(normalizer(map1)))
    print(np.std(normalizer(map2)))
    print(np.sum(np.absolute(normalizer(map2))))

    DATA_SOURCE = R"D:\Documents\UC Berkeley\2020-2021\Abbasi-Asl Lab\Data"
    image = "cat_2"
    mode = "compressed"
    discrepancy = 0
    for i2 in range(0,3):
        for i in range(1, 9):
            layer = "group"+str(i2)+"_block" + str(i) + "_conv0_v"
            filename = DATA_SOURCE + "\\GradCAM\\" + image + "-" + mode + "\\" + layer
            with open(filename, 'rb') as fp: map1 = pickle.load(fp)
            map1 = np.absolute(map1)
            map1 = normalizer(map1)

            layer = "group"+str(i2)+"_block" + str(i) + "_conv0"
            filename = DATA_SOURCE + "\\GradCAM\\" + image + "-" + mode + "\\" + layer
            with open(filename, 'rb') as fp: map2 = pickle.load(fp)
            map2 = normalizer(map2)
            discrepancy += np.sum(np.absolute(map1-map2))

            layer = "group"+str(i2)+"_block" + str(i) + "_conv1_v"
            filename = DATA_SOURCE + "\\GradCAM\\" + image + "-" + mode + "\\" + layer
            with open(filename, 'rb') as fp: map3 = pickle.load(fp)
            map3 = np.absolute(map3)
            map3 = normalizer(map3)

            layer = "group"+str(i2)+"_block" + str(i) + "_conv1"
            filename = DATA_SOURCE + "\\GradCAM\\" + image + "-" + mode + "\\" + layer
            with open(filename, 'rb') as fp: map4 = pickle.load(fp)
            map4 = normalizer(map4)
            discrepancy += np.sum(np.absolute(map3-map4))
        
    print(discrepancy)   

    #map1 = np.ndarray.flatten(map1)
    #map2 = np.ndarray.flatten(map2)
    #plt.hist(map1, bins=500, alpha = 0.5, label = 'uncompressed')
    #plt.hist(map2, bins=500, alpha=0.5, label='compressed')

    #plt.legend(loc ='upper right')
    #plt.xlabel('Gradient Bins')
    #plt.ylabel('Pixel Count')
    #plt.show()
    #plt.close('all')
    print('Process Finished')

