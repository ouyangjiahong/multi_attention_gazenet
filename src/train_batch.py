import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torchvision.models as models
import torch.nn as nn
import torch.nn.parallel
import numpy as np

from model import *
from util import *


def main():
    bs = 2
    ts = 32
    img_shape = (3, 224, 224)
    # set up the feature extractor
    arch = 'alexnet'
    # arch = 'vgg16'
    extractor_model = FeatureExtractor(arch=arch)
    extractor_model.features = torch.nn.DataParallel(extractor_model.features)
    extractor_model.cuda()      # uncomment this line if using cpu
    extractor_model.eval()

    # pre-processing for the image, should add data generator loader here
    imgs = 255 * np.random.rand(bs, ts, 224, 224, 3)
    imgs = normalize(imgs)

    # compute the features for a batch of images, [bs, ts, 256, 36]
    imgs = np.reshape(imgs, (bs*ts,) + img_shape)
    imgs_var = torch.autograd.Variable(torch.Tensor(imgs))
    cnn_feat_var = extractor_model(imgs_var)
    print(cnn_feat_var.size())
    cnn_feat_var = cnn_feat_var.view((bs, ts, -1, cnn_feat_var.size()[2]*cnn_feat_var.size()[3]))
    cnn_feat_var = cnn_feat_var.permute(0, 1, 3, 2)
    print("cnn fetaure extractor")
    print(cnn_feat_var.size())          # (bs, ts, 36, 256)

    # test spatial attention layer
    spatial_attention_layer = SpatialAttentionLayer(lstm_hidden_size = 64,
                                cnn_feat_size = 256, projected_size = 64)
    spatial_attention_layer.cuda()
    spatial_attention_layer.train()

    lstm_hidden = torch.randn(bs, ts, 64).cuda()
    lstm_hidden_var = torch.autograd.Variable(lstm_hidden)
    spatial_weight = spatial_attention_layer(lstm_hidden_var, cnn_feat_var)
    print("spatial weight")         # (bs, ts, grid_num)
    print(spatial_weight.size())
    # print(spatial_weight.sum(dim=2))
    spatial_feat_var = cnn_feat_var * spatial_weight.unsqueeze(3)   # (bs, ts, 256, 36)
    spatial_feat_var = spatial_feat_var.sum(2)      # (bs, ts, 256)
    print("spatial feat")
    print(spatial_feat_var.size())

    # test temporal attention layer, loop for each bach
    temporal_attention_layer = TemporalAttentionLayer(lstm_hidden_size = 64,
                                spatial_feat_size = 256, projected_size = 32)
    temporal_attention_layer.cuda()
    temporal_attention_layer.train()

    t_cnt = 0
    while(True):
        if t_cnt >= ts:         # in real-time, don't know the time steps
            print("finish the interaction")
            break
        lstm_hidden = torch.randn(bs, 64).cuda()    # should be h[t-1]
        lstm_hidden_var = torch.autograd.Variable(lstm_hidden)
        temporal_weight = temporal_attention_layer(lstm_hidden_var, spatial_feat_var[:,t_cnt,:])
        print("temporal weight")         # (bs)
        print(temporal_weight.size())
        t_cnt += 1



    lstm_hidden = torch.randn(bs, ts, 64).cuda()
    lstm_hidden_var = torch.autograd.Variable(lstm_hidden)
    temporal_weight = temporal_attention_layer(lstm_hidden_var, spatial_feat_var)
    print("temporal weight")         # (ts)
    print(temporal_weight.size())
    # print(temporal_weight.sum(dim=1))
    temporal_feat_var = spatial_feat_var * temporal_weight.unsqueeze(2)
    print("temporal feat")
    print(temporal_feat_var.size())



if __name__ == '__main__':
    main()
