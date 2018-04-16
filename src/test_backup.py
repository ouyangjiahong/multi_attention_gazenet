#!/usr/bin/env python
import os
import time
import torch
import pdb
import shutil
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import numpy as np
import cv2
# from skimage.io import imsave
import matplotlib.pyplot as plt
# from skimage.transform import resize

from model_backup import GazeClassifier
from util import *
from logger import Logger
import gazenetGenerator_nogaze as gaze_gen

# for real-time prediction
def predict(img_seq, model):
    dim = len(img_seq.shape)
    if dim != 5:
        if dim != 4:
            raise ValueError("In prediction mode, the input images size should be (bs, ts, 224, 224, 3) or (ts, 224, 224, 3)")
        else:
            img_seq = np.expand_dims(img_seq, axis=0)
    img_size = (img_seq.shape[2], img_seq.shape[3])

    img_seq = normalize(img_seq)
    img_seq = np.reshape(img_seq,(-1,3,) + img_size)
    img_seq_var = torch.autograd.Variable(torch.Tensor(img_seq).cuda())
    output_var = model(img_seq_var)
    prediction = F.softmax(output_var, dim=1)
    _, label = torch.max(prediction, 1)
    print(label)
    return label, prediction


def main():
    num_classes = 6
    batch_size = 6
    gaze_gen_batch_size = 1
    gaze_gen_time_steps = 1
    epochs = 50
    cnn_feat_size = 256     # AlexNet
    gaze_size = 3
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    eval_freq = 3       # epoch
    print_freq = 1      # iteration
    dataset_path = '../../gaze-net-classification/data1'
    img_size = (224,224)
    arch = 'alexnet'


    model = GazeClassifier(arch=arch)
    model.cuda()
    model = load_checkpoint(model)

    # define generator
    trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)
    train_data = trainGenerator.flow_from_directory(dataset_path, subset='training', crop=False,target_size=img_size,
                                batch_size=gaze_gen_batch_size, crop_with_gaze=True,time_steps=gaze_gen_time_steps,
                                crop_with_gaze_size=img_size[0],class_mode='categorical')
    # small dataset, error using validation split
    val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', crop=False,target_size=img_size,
                                batch_size=gaze_gen_batch_size, crop_with_gaze=True,time_steps=gaze_gen_time_steps,
                                crop_with_gaze_size=img_size[0], class_mode='categorical')


    # start predict
    for i in range(10):
        print("start a new interaction")
        # img_seq: (ts,224,224,3), gaze_seq: (ts, 3), ouput: (ts, 6)
        # [img_seq, gaze_seq], target = next(val_data)
        img_seq, target = next(train_data)
        restart = True
        # print("input size")
        # print(img_seq.shape)

        print(target)
        for j in range(img_seq.shape[0]):
            predict(img_seq[j], model)
            # print(target[j])
            # restart = False
            img = img_seq[j,:,:,:,:]
            img = np.squeeze(img, axis=0)
            # cv2.circle(img, (int(gazes[j,1]), int(gazes[j,2])), 10, (255,0,0),-1)
            cv2.imshow('ImageWindow', img)
            cv2.waitKey(33)
        # predict(img_seq[5:10], gaze_seq[5:10], extractor_model, model, restart=False)


def load_checkpoint(model, filename='../model/classification/model_best.pth.tar'):
    if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    return model

if __name__ == '__main__':
    main()
