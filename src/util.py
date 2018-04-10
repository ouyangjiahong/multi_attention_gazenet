#!/usr/bin/env python
import numpy as np
import torch

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

    # imgs_trans = np.zeros((shape[0], shape[3], shape[1], shape[2]))
    # for i in range(shape[0]):
    #     img = imgs[i]
    #     img = img / 255.0
    #     mean=[0.485, 0.456, 0.406]
    #     std=[0.229, 0.224, 0.225]
    #     img = (img - mean) / std
    #     img = np.transpose(img, (2, 0, 1))
    #     imgs_trans[i,:,:,:] = img
    # return imgs_trans

def metric_frame(output, target):
    correct = 0
    num_frame = target.size()[0]
    _, prediction = torch.max(output.data, 1)
    correct += (prediction == target.data).sum()
    return correct

def metric_interaction(prediction, target):
    pass
