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
# from skimage.io import imsave
import matplotlib.pyplot as plt
# from skimage.transform import resize

from model import FeatureExtractor, SpatialAttentionModel
from util import *
from logger import Logger
import gazeWholeGenerator as gaze_gen

# for real-time prediction
def predict(img_seq, gaze_seq, extractor_model, model, restart=False):
    dim = len(img_seq.shape)
    if dim != 4:
        raise ValueError("In prediction mode, the input images size should be (ts, 224, 224, 3)")
    ts = img_seq.shape[0]
    bs = 1
    img_size = (img_seq.shape[1], img_seq.shape[2])

    img_seq = normalize(img_seq)
    img_seq = np.reshape(img_seq, (bs*ts,3,) + img_size)
    img_seq_var = torch.autograd.Variable(torch.Tensor(img_seq))
    gaze_seq_var = torch.autograd.Variable(torch.Tensor(gaze_seq))
    gaze_seq_var = gaze_seq_var.unsqueeze(0)        # no bs dim

    # extract cnn feature
    cnn_feat_var = extractor_model(img_seq_var)
    cnn_feat_var = cnn_feat_var.view((bs, ts, -1, cnn_feat_var.size()[2]*cnn_feat_var.size()[3]))
    cnn_feat_var = cnn_feat_var.permute(0, 1, 3, 2)     # (bs, ts, 36, 256)

    prediction = model(cnn_feat_var, gaze_seq_var, restart=restart)
    prediction = prediction.view((bs*ts, -1))

    # get output result
    prediction = F.softmax(prediction, dim=1)
    _, prediction = torch.max(prediction.data, 1)       # (ts,)
    print(prediction)
    return prediction


def main():
    # define parameters
    num_class = 6
    batch_size = 1
    time_step = 32
    cnn_feat_size = 256     # AlexNet
    gaze_size = 3
    gaze_lstm_hidden_size = 64
    gaze_lstm_projected_size = 128
    dataset_path = '../data/gaze_dataset'
    # dataset_path = '../../gaze-net/gaze_dataset'
    img_size = (224, 224)

    # define model
    arch = 'alexnet'
    extractor_model = FeatureExtractor(arch=arch)
    extractor_model.features = torch.nn.DataParallel(extractor_model.features)
    extractor_model.cuda()      # uncomment this line if using cpu
    extractor_model.eval()

    model = SpatialAttentionModel(num_class, cnn_feat_size,
                        gaze_size, gaze_lstm_hidden_size, gaze_lstm_projected_size)
    model.cuda()

    # load model from checkpoint
    model = load_checkpoint(model)

    # define generator, TODO: change with the new one
    trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0)
    train_data = trainGenerator.flow_from_directory(dataset_path, subset='training', crop=False,
                    batch_size=batch_size, target_size= img_size, class_mode='sequence_pytorch')

    # start predict
    for i in range(3):
        print("start a new interaction")
        # img_seq: (ts,224,224,3), gaze_seq: (ts, 3), ouput: (ts, 6)
        [img_seq, gaze_seq], _ = next(train_data)
        predict(img_seq[:5], gaze_seq[:5], extractor_model, model, restart=True)
        predict(img_seq[5:10], gaze_seq[5:10], extractor_model, model, restart=False)


def load_checkpoint(model, filename='../model/spatial/checkpoint.pth.32.tar'):
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