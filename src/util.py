#!/usr/bin/env python
import numpy as np
import torch
# from skimage import data, exposure, img_as_float
from PIL import ImageEnhance, Image

def normalize(imgs):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    shape = imgs.shape

    imgs /= 255.0
    imgs -= mean
    imgs /= std

    if len(shape) == 4:
        imgs = np.transpose(imgs, (0, 3, 1, 2))
    else:
        imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
    return imgs

def denormalize(imgs):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    shape = imgs.shape

    if len(shape) == 4:
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    else:
        imgs = np.transpose(imgs, (0, 1, 3, 4, 2))

    imgs *= std
    imgs += mean
    imgs *= 255.0
    imgs = imgs.astype(int)

    return imgs

# def adjust_contrast(img):
#     img_gamma = exposure.rescale_intensity(img)
#     print(img_gamma.shape)
#     return img_gamma

def adjust_contrast(img):
    img = Image.fromarray(np.uint8(img))
    enhancer = ImageEnhance.Contrast(img)
    img_new = enhancer.enhance(1)
    enhancer = ImageEnhance.Color(img_new)
    img_new = enhancer.enhance(1.5)
    return img_new

def metric_frame(output, target):
    correct = 0
    num_frame = target.size()[0]
    _, prediction = torch.max(output.data, 1)
    correct += (prediction == target.data).sum()
    return correct

def metric_interaction(prediction, target):
    pass
