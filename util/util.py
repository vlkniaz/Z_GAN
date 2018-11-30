from __future__ import print_function
import torch
import numpy as np
import colorsys
from PIL import Image
import os


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def fruxel2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.squeeze(image_numpy, axis=0)
    if len(image_numpy.shape) == 4:
        image_numpy = np.squeeze(image_numpy, axis=0)
    image_numpy = np.transpose(image_numpy, (1, 2, 0))

    mode = 'depth'
    fruxel_depth = 128
    fruxel = image_numpy
    out_size = image_numpy.shape[0] 

    if mode=='seg':
        img = np.sum(fruxel, axis=2)*255
    elif mode =='depth':
        img = np.zeros((out_size, out_size, 3), dtype='uint8')
        for z in range(fruxel_depth-1,-1,-1):
            mask = fruxel[:,:,z] >= 0.5 
            z1 = fruxel_depth - 1 - z
            color = np.array(colorsys.hsv_to_rgb((z1*240.0/fruxel_depth)/360,1,1))
            img[mask,:] = color*255

    return img.astype(imtype)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def thermal_tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy =  np.transpose(image_numpy, (1, 2, 0))
    tmin = np.min(image_numpy)
    tmax = np.max(image_numpy)
    print('tmin: ' + str(tmin))
    print('tmax: ' + str(tmax))
    d = tmax - tmin
    image_numpy = (image_numpy - tmin) / d * 255.0
    return image_numpy.astype(imtype)

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def thermal_rel_tensor2im(image_tensor, label_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    #if image_numpy.shape[0] == 1:
    #    image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy =  np.transpose(image_numpy, (1, 2, 0))

    label_numpy = label_tensor[0].cpu().float().numpy()
    label_numpy =  np.transpose(label_numpy, (1, 2, 0))

    image_numpy = np.reshape(image_numpy, (512,512,1))

    label_numpy_s = np.reshape(label_numpy[:,:,0], (512,512,1))
    result_numpy = label_numpy_s * 30.0 + image_numpy * 10.0
    result_numpy = np.dstack((result_numpy, result_numpy, result_numpy))

    image_numpy = result_numpy
    tmin = np.min(image_numpy)
    tmax = np.max(image_numpy)
    print('tmin: ' + str(tmin))
    print('tmax: ' + str(tmax))
    d = tmax - tmin
    image_numpy = (image_numpy - tmin) / d * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
