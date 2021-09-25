import numpy as np
import matplotlib.pyplot as plt
import utils
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

def noise_gradual(net, input_location, true_label, num_iter=10, sp = False, filename='', show = False):
    image = utils.load_image(input_location)
    plt.title('Original')
    plt.imshow(image)
    plt.show()
    plt.close('all')

    vals = []
    for i in range(0, num_iter+1):
        prob = ((1/num_iter)*i)**3
        new_image = sp_noise(image, prob, sp)
        if show:
            plt.title('Noise Lvl: '+str(prob))
            plt.imshow(new_image)
            plt.show()
            plt.close('all')

        prediction = utils.net_prediction(net, new_image)
        accuracy = prediction[true_label]
        vals.append(accuracy)

        print([prob, accuracy])

    return vals

def feature_map(net, input_location, layer):
    image = utils.load_image(input_location)
    plt.title('Original')
    plt.imshow(image)
    plt.show()
    plt.close('all')

    prediction = utils.net_prediction(net, image)
    print(net.blobs.keys())
    data = np.copy(net.blobs[layer].data)
    plt_dim = int(math.ceil(math.sqrt(data.shape[1])))
    f, axarr = plt.subplots(plt_dim, plt_dim)
    for i in range(0,data.shape[1]):
        print(np.mean(utils.normalizer(data[0][i])))
        axarr[i//plt_dim][i%plt_dim].imshow(utils.normalizer(data[0][i]), cmap='Greys', interpolation='None')
    plt.show()
    plt.close('all')

def saliency_grad(net,input_location, true_label, filename = '', show=True):
    image = utils.load_image(input_location)
    plt.title('Original')
    plt.imshow(image)
    if show: plt.show()
    plt.close('all')
    net.blobs['prob'].diff[0][true_label] = 1
    predictions, diffs = utils.net_prediction(net, image, backward = True)
    diff_map = np.absolute(diffs['data'][0])
    sal_map = np.sum(diff_map, axis=0)
    norm_map = utils.normalizer(sal_map)

    plt.imshow(norm_map, cmap = 'jet', interpolation='None')
    if show: plt.show()
    plt.close('all')

    weight_data = np.copy(net.params["fc"][0].data)
    weights = weight_data[true_label]
    cam_nummap = np.zeros(net.blobs["group2_block8_sum"].data[0][0].shape)
    for i in range(0, len(weights)):
        cam_nummap += weights[i] * net.blobs["group2_block8_sum"].data[0][i]
    cam_nummap = np.abs(cam_nummap)
    cam_nummap = utils.normalizer(cam_nummap)
    transformed_cam = cv.resize(cam_nummap, net.blobs['data'].data.shape[2:4])
    print(transformed_cam.shape)

    plt.imshow(transformed_cam, cmap='jet', interpolation='hanning', alpha=1)
    if show: plt.show()
    plt.close('all')

    salcam_map = np.multiply(transformed_cam, sal_map)
    if filename != '':
        with open(filename, 'wb') as fp: pickle.dump(salcam_map, fp)

    salcam_map = utils.normalizer(salcam_map)
    print(np.amax(salcam_map))
    if show: utils.grad_histogram(salcam_map)
    plt.imshow(salcam_map, cmap='jet', interpolation='None')
    if show: plt.show()
    plt.close('all')

def integrated_gradients(net, input_location, num_iter, true_label, filename='', show=True):
    image = utils.load_image(input_location)
    image = np.where(image==0, 0.01, image)
    plt.title('Original')
    plt.imshow(image)
    if show: plt.show()
    plt.close('all')

    net.blobs['prob'].diff[0][true_label] = 1
    increment = 1/num_iter
    gradients = []

    for i in range(0, num_iter+1):
        #Create Image
        beta = i*increment
        new_image = cv.convertScaleAbs(255 * image, 1, beta)
        new_image = new_image/255
        #Find Gradient
        predictions, diffs = utils.net_prediction(net, new_image, True)
        gradients.append(diffs['data'][0])
    #Integral
    np.amax(gradients)
    integral = np.sum(gradients[:-1] + gradients[1:], axis=0)
    np.amax(integral)
    integral = np.transpose(integral, (1, 2, 0))

    image_mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    #transformer.set_mean('data', image_mean)
    transformed_image = transformer.preprocess('data', image)
    transformed_image = np.transpose(transformed_image, (1,2,0))
    print(np.amax(transformed_image))
    print(np.amin(transformed_image))

    integral = np.multiply(transformed_image/(2*num_iter), integral)

    #Gradient Mapping
    gradmap = np.sum(integral, axis = 2)
    if filename != '':
        with open(filename,'wb') as fp: pickle.dump(gradmap, fp)

    print('Std Dev: ', str(np.std(gradmap)))
    print('Max: ', str(np.amax(gradmap)))
    print('Min: ', str(np.amin(gradmap)))
    print('Range: ', str(np.amax(gradmap)-np.amin(gradmap)))

    map_norm = utils.normalizer(gradmap)
    if show:
        utils.grad_histogram(map_norm)
        utils.visualizer(map_norm)
        utils.visualizer(map_norm, 0.5)
        utils.visualizer(map_norm, 0.53)
        utils.visualizer(map_norm, 0.56)
        utils.visualizer(map_norm, 0.59)
        utils.visualizer(map_norm, 0.62)

def visualize_gradcam(net, input_location, layer, true_label, scaleorder = False, filename = ''):
    image = utils.load_image(input_location)
    plt.imshow(image)
    plt.show()
    plt.close('all')

    net.blobs['prob'].diff[0][true_label] = 1
    predictions, diffs = utils.net_prediction(net, image, True, layer)
    print(diffs.keys())

    gradients = diffs[layer][0]
    gradients = np.nan_to_num(gradients)
    gradients = np.abs(gradients)
    weights = [np.average(gradient) for gradient in gradients]

    feature_maps = net.blobs[layer].data[0]
    gradcam_map = np.zeros(feature_maps[0].shape)
    for i in range(0, feature_maps.shape[0]):
        gradcam_map = gradcam_map + feature_maps[i]*weights[i]
    #gradcam_map = np.absolute(gradcam_map)
    if not scaleorder:
        gradcam_map[gradcam_map <0] = 0
        gradcam_map = utils.normalizer(gradcam_map)
        utils.grad_histogram(gradcam_map, 0.02)

        plt.imshow(gradcam_map, cmap='jet', interpolation='hanning', alpha=1)
        plt.show()
        plt.close('all')
    else:
        gradcam_map[gradcam_map < 0] = 0
        gradcam_map = utils.scale_normalize(gradcam_map)
        plt.imshow(gradcam_map, cmap='jet', interpolation='None', alpha=1)
        plt.show()
        plt.close('all')

def visualize_cam(net, input_location, filename = ''):
    weight_data = np.copy(net.params["fc"][0].data)
    print(weight_data)
    image = utils.load_image(input_location)
    predictions = utils.net_prediction(net, image)
    classification = np.argmax(predictions)
    print(classification)
    weights = weight_data[classification]
    cam_nummap = np.zeros(net.blobs["group2_block8_sum"].data[0][0].shape)
    print(net.blobs["group2_block8_sum"].data[0][0].shape)
    print(weights)
    for i in range(0, len(weights)):
        cam_nummap += weights[i]*net.blobs["group2_block8_sum"].data[0][i]
    print(cam_nummap)
    cam_nummap = np.abs(cam_nummap)
    max = np.amax(cam_nummap)
    min = np.amin(cam_nummap)
    cam_nummap = (cam_nummap-min)/(max-min)
    plt.axis('off')
    plt.imshow(cam_nummap, cmap='jet', interpolation='hanning', alpha=1)
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.show()
    print("Complete")

def visualize_weights(net, layer_name, padding=4, filename='', show=True, scale_factor=1):
    # The parameters are a list of [weights, biases]
    data = np.copy(net.params[layer_name][0].data)
    # N is the total number of convolutions
    N = data.shape[0] * data.shape[1]
    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size_x = data.shape[2]
    filter_size_y = data.shape[3]
    # Size of the result image including padding
    result_size_x = filters_per_row * (filter_size_x + padding) - padding
    result_size_y = filters_per_row * (filter_size_y + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size_x, result_size_y))
    #result.fill(-1*scale_factor)
    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[0]):
        for c in range(data.shape[1]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size_x):
                for j in range(filter_size_y):
                    result[filter_y * (filter_size_x + padding) + i, filter_x * (filter_size_y + padding) + j] = data[
                        n, c, i, j]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    print(min)
    print(max)
    def scale(x):
        return (x+scale_factor) / (2*scale_factor)
    #result = (result - min) / (max - min)
    result = scale(result)
    # Plot figure
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(result, cmap='viridis', interpolation='nearest')

    # Save plot if filename is set
    if filename != '':
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    if show: plt.show()
    return

def filter_generate(net, limit, network_name, comp_or_uncomp):
    identifier = comp_or_uncomp + "-" + network_name
    at = 0
    for layer_name in net.params:
        if at == limit:
            return
        visualize_weights(net, layer_name, 4, "Images/"+identifier+"/"+identifier+"-"+layer_name+".png", False)
        at += 1
    return