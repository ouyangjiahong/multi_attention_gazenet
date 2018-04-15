#!/usr/bin/env python
import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.vgg import model_urls
import torch.nn as nn
import torch.nn.parallel
import numpy as np
# from skimage.io import imsave
# from skimage.transform import resize


class FeatureExtractor(nn.Module):
    '''
    call the CNN model as a feature extractor
    '''
    def __init__(self, arch):
        super(FeatureExtractor, self).__init__()
        self.arch = arch   # 'alexnet' or 'vgg16'
        self.features = self.load_pretrained_model()

    def forward(self, img):
        feat = self.features(img)
        return feat

    def load_pretrained_model(self):
        print("Load model from torchvision.models")
        if self.arch == 'alexnet':
            pretrained_model = models.__dict__[self.arch](pretrained=True)
            pretrained_model = pretrained_model.features    # only keep the conv layers
            #TODO: change it back
            # pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1]) # remove the last maxpool
            # print(pretrained_model)
        else:
            raise Exception("Please download the model to ~/.torch and change the params")
        return pretrained_model

class SpatialAttentionLayer(nn.Module):
    '''
    compute the spatial attention for each frame
    '''
    def __init__(self, lstm_hidden_size, cnn_feat_size, projected_size):
        '''
        lstm_hidden_size: number of LSTM hidden nodes(LSTM for gaze sequence)
        cnn_feat_size: number of CNN features for each region
        projected_size: output feature size
        '''
        super(SpatialAttentionLayer, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.cnn_feat_size = cnn_feat_size
        self.projected_size = projected_size
        self.linear_lstm = nn.Linear(lstm_hidden_size, projected_size)
        self.linear_cnn = nn.Linear(cnn_feat_size, projected_size)
        self.linear_weight = nn.Linear(projected_size, 1, bias=False)

    def forward(self, lstm_hidden, cnn_feat):
        '''
        lstm_hidden: (bs, lstm_hidden_size)
        cnn_feat: (bs, grid_num, cnn_feat_size)
        zi = wh * tanh(Wv * V + Wg * H)
        weight : ai = sigmoid(zi)
        '''
        bs = cnn_feat.size()[0]         #normally should be 1
        # 6*6=36, corresponding to the size of CNN output feature size
        grid_num = cnn_feat.size()[1]
        # print("grid num")
        # print(grid_num)
        weight = []
        for i in range(grid_num):
            H = self.linear_lstm(lstm_hidden)
            V = self.linear_cnn(cnn_feat[:,i,:])
            feat_sum = H + V
            feat_sum = F.tanh(feat_sum)                 # (bs, projected_size)
            feat_sum = self.linear_weight(feat_sum)     # (bs, 1)
            weight.append(feat_sum)
        weight = torch.cat(weight, dim=0)               # (bs * grid_num)
        spatial_weight = F.softmax(weight.view(bs, grid_num), dim=1)
        return spatial_weight

class TemporalAttentionLayer(nn.Module):
    '''
    compute the temporal attention for each sequence
    '''
    def __init__(self, lstm_hidden_size, spatial_feat_size, projected_size):
        '''
        lstm_hidden_size: number of LSTM hidden nodes(LSTM for image feature sequence)
        spatial_feat_size: number of features from spatial attention layer
        projected_size: output feature size
        '''
        super(TemporalAttentionLayer, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.spatial_feat_size = spatial_feat_size
        self.projected_size = projected_size
        self.linear_lstm = nn.Linear(lstm_hidden_size, projected_size)
        self.linear_spatial_feat = nn.Linear(spatial_feat_size, projected_size)
        self.linear_weight = nn.Linear(projected_size, 1, bias=False)

    def forward(self, lstm_hidden, spatial_feat):
        '''
        lstm_hidden: (bs, ts, lstm_hidden_size)
        spatial_feat: (bs, ts, spatial_feat_size)
        zi = wh * tanh(Wv * V + Wg * H)
        weight : ai = sigmoid(zi)
        '''
        bs = spatial_feat.size()[0]         #normally should be 1
        ts = spatial_feat.size()[1]

        weight = []
        for i in range(ts):
            H = self.linear_lstm(lstm_hidden[:,i,:])
            V = self.linear_spatial_feat(spatial_feat[:,i,:])
            feat_sum = H + V
            feat_sum = F.tanh(feat_sum)                 # (bs, projected_size)
            feat_sum = self.linear_weight(feat_sum)     # (bs, 1)
            weight.append(feat_sum)
        weight = torch.cat(weight, dim=0).view(bs, ts)       # (bs, ts)
        temporal_weight = F.softmax(weight, dim=1)      # (bs, ts)
        return temporal_weight

class SpatialAttentionModel(nn.Module):
    '''
    build the whole model
    '''
    def __init__(self, num_class, cnn_feat_size, gaze_size,
                gaze_lstm_hidden_size, spatial_projected_size):
        '''
        num_frame: set frame number for each interaction
        cnn_feat_size: number of CNN features for each image
        gaze_size: 3, (p, x, y)
        gaze_lstm_hidden_size: number of gaze LSTM hidden size
        spatial_projected_size: output feature size of spatial attention
        '''
        super(SpatialAttentionModel, self).__init__()
        self.num_class = num_class
        # self.num_frame = num_frame
        self.cnn_feat_size = cnn_feat_size
        self.gaze_size = gaze_size
        self.gaze_lstm_hidden_size = gaze_lstm_hidden_size
        self.spatial_projected_size = spatial_projected_size
        self.gaze_lstm_layer = self.build_gaze_lstm()
        self.spatial_attention_layer = SpatialAttentionLayer(gaze_lstm_hidden_size,
                                            cnn_feat_size, spatial_projected_size)
        self.mlp_layer = self.build_mlp_classifier()
        self.init_hidden = self.init_gaze_lstm_hidden()
        self.init_cell = self.init_gaze_lstm_cell()
        self.gaze_lstm_hidden = None
        self.gaze_lstm_cell = None

    def build_gaze_lstm(self):
        lstm = nn.LSTM(self.gaze_size, self.gaze_lstm_hidden_size, batch_first=True)
        return lstm

    def build_mlp_classifier(self):
        input_size = self.cnn_feat_size + self.gaze_lstm_hidden_size
        # print(input_size)
        mlp = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.Linear(input_size, input_size),
                nn.Linear(input_size, self.num_class))
        return mlp

    def init_gaze_lstm_hidden(self):
        return nn.Sequential(
                nn.Linear(3, self.gaze_lstm_hidden_size),
                nn.Tanh(),
                nn.Linear(self.gaze_lstm_hidden_size, self.gaze_lstm_hidden_size),
                nn.Tanh())

    def init_gaze_lstm_cell(self):
        return nn.Sequential(
                nn.Linear(3, self.gaze_lstm_hidden_size),
                nn.Tanh(),
                nn.Linear(self.gaze_lstm_hidden_size, self.gaze_lstm_hidden_size),
                nn.Tanh())

    def init_gaze_lstm_state(self, gaze_seq):
        mean_gaze = torch.mean(gaze_seq, 1).squeeze(1)  # (bs, 3)
        self.gaze_lstm_hidden = F.tanh(self.init_hidden(mean_gaze)).unsqueeze(0)  # (1, bs, h)
        self.gaze_lstm_cell = F.tanh(self.init_cell(mean_gaze)).unsqueeze(0) # (1, bs, h)

    def forward(self, cnn_feat_seq, gaze_seq, restart=True):
        '''
        cnn_feat_seq: (bs, num_frame, 36, 256)
        gaze_seq: (bs, num_frame, 3)
        '''
        num_frame = cnn_feat_seq.size()[1]
        bs = cnn_feat_seq.size()[0]
        if restart == True or self.gaze_lstm_cell is None:
            self.init_gaze_lstm_state(gaze_seq)

        # start the loop for region weight
        pred_all = []
        grid_side = int(np.sqrt(cnn_feat_seq.size()[2]))
        for i in range(num_frame):
            # calculate the weight
            spatial_weight = self.spatial_attention_layer(self.gaze_lstm_hidden.squeeze(dim=0),
                                        cnn_feat_seq[:,i,:,:]) # (bs, 36)
            spatial_feat = cnn_feat_seq[:,i,:,:] * spatial_weight.unsqueeze(2)   # (bs, 256, 36)
            spatial_feat = spatial_feat.sum(1)      # (bs, 256)

            if False:           # save weight image
                spatial_weight_vis = 2550 * spatial_weight_all[:,i,:,:]
                spatial_weight_vis = spatial_weight_vis.cpu().numpy()
                # print(spatial_weight_vis)
                spatial_weight_vis = spatial_weight_vis.astype(np.uint8)
                spatial_weight_vis = resize(spatial_weight_vis[0], (224, 224))
                # print('save weight')
                img_name = '../vis/' + str('%03d'%i) + '.jpg'
                print(img_name)
                imsave(img_name, spatial_weight_vis)

            # update the lstm, h: (bs, hidden_num) + f: (bs, 256)
            gaze = gaze_seq[:, i, :].unsqueeze(1)
            # if i == 0:
            #     gaze = gaze_seq[:, 0, :]
            #     gaze = gaze.unsqueeze(1)
            # else:
            #     gaze = gaze_seq[:, :i, :]
            gaze_lstm_output, (self.gaze_lstm_hidden, self.gaze_lstm_cell) = self.gaze_lstm_layer(gaze,
                                            (self.gaze_lstm_hidden, self.gaze_lstm_cell))

            # concate lstm and feat for mlp, (bs, 256 + 64), self.gaze_lstm_hidden: (1,1,64)
            feat_concat = torch.cat((spatial_feat, self.gaze_lstm_hidden[0]), dim=1)

            pred = self.mlp_layer(feat_concat)
            pred_all.append(pred.unsqueeze(0))

        pred_all = torch.cat(pred_all, 1)

        return pred_all

class MultipleAttentionModel(nn.Module):
    '''
    build the whole model
    '''
    def __init__(self, num_class, cnn_feat_size, gaze_size,
                gaze_lstm_hidden_size, spatial_projected_size,
                temporal_projected_size, queue_size, extractor=False):
        '''
        num_frame: set frame number for each interaction
        cnn_feat_size: number of CNN features for each image
        gaze_size: 3, (p, x, y)
        gaze_lstm_hidden_size: number of gaze LSTM hidden size
        spatial_projected_size: output feature size of spatial attention
        temporal_projected_size: output feature size of temporal attention
        '''
        super(MultipleAttentionModel, self).__init__()
        self.num_class = num_class
        # self.num_frame = num_frame
        self.cnn_feat_size = cnn_feat_size
        self.gaze_size = gaze_size
        self.gaze_lstm_hidden_size = gaze_lstm_hidden_size
        self.spatial_projected_size = spatial_projected_size
        self.temporal_projected_size = temporal_projected_size
        self.queue_size = queue_size
        self.gaze_lstm_layer = self.build_gaze_lstm()
        self.spatial_attention_layer = SpatialAttentionLayer(gaze_lstm_hidden_size,
                                            cnn_feat_size, spatial_projected_size)
        self.temporal_attention_layer = TemporalAttentionLayer(gaze_lstm_hidden_size,
                                            cnn_feat_size, temporal_projected_size)
        self.mlp_layer = self.build_mlp_classifier()
        self.init_hidden = self.init_gaze_lstm_hidden()
        self.init_cell = self.init_gaze_lstm_cell()
        self.gaze_lstm_hidden = None
        self.gaze_lstm_cell = None
        self.temporal_feat_counter = None
        self.queue_hidden = None
        self.queue_spatial = None
        self.extractor = extractor
        if self.extractor:
            self.feat_extractor = self.build_feature_extractor()

    def build_feature_extractor(self):
        arch = 'alexnet'
        extractor_model = FeatureExtractor(arch=arch)
        for i, param in enumerate(extractor_model.features.parameters()):
            if i <= 5:      # only fine tune the last two conv layers
                param.requires_grad = False
        # print(extractor_model)
        return extractor_model

    def build_gaze_lstm(self):
        lstm = nn.LSTM(self.gaze_size, self.gaze_lstm_hidden_size, batch_first=True)
        return lstm

    def build_mlp_classifier(self):
        input_size = self.cnn_feat_size + self.gaze_lstm_hidden_size
        # print(input_size)
        mlp = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.Linear(input_size, input_size),
                nn.Linear(input_size, self.num_class))
        return mlp

    def init_gaze_lstm_hidden(self):
        return nn.Sequential(
                nn.Linear(3, self.gaze_lstm_hidden_size),
                nn.Tanh(),
                nn.Linear(self.gaze_lstm_hidden_size, self.gaze_lstm_hidden_size),
                nn.Tanh())

    def init_gaze_lstm_cell(self):
        return nn.Sequential(
                nn.Linear(3, self.gaze_lstm_hidden_size),
                nn.Tanh(),
                nn.Linear(self.gaze_lstm_hidden_size, self.gaze_lstm_hidden_size),
                nn.Tanh())

    def init_gaze_lstm_state(self, gaze_seq):
        mean_gaze = torch.mean(gaze_seq, 1).squeeze(1)  # (bs, 3)
        self.gaze_lstm_hidden = F.tanh(self.init_hidden(mean_gaze)).unsqueeze(0)  # (1, bs, h)
        self.gaze_lstm_cell = F.tanh(self.init_cell(mean_gaze)).unsqueeze(0) # (1, bs, h)

    def forward(self, cnn_feat_seq, gaze_seq, restart=True):
        '''
        cnn_feat_seq: (bs, num_frame, 36, 256) or img_seq: (bs*ts, 3, 224, 224)
        gaze_seq: (bs, num_frame, 3)
        '''
        num_frame = gaze_seq.size()[1]
        bs = gaze_seq.size()[0]
        if self.extractor == True:
            cnn_feat_seq = self.feat_extractor(cnn_feat_seq)
            cnn_feat_seq = cnn_feat_seq.view((bs, num_frame, -1, cnn_feat_seq.size()[2]*cnn_feat_seq.size()[3]))
            cnn_feat_seq = cnn_feat_seq.permute(0, 1, 3, 2)

        if restart == True or self.gaze_lstm_cell is None:
            self.init_gaze_lstm_state(gaze_seq)
            self.queue_hidden = []
            self.queue_spatial = []

        # start the loop for region weight
        pred_all = []
        grid_side = int(np.sqrt(cnn_feat_seq.size()[2]))
        for i in range(num_frame):
            # calculate the image feature using spatial weight
            spatial_weight = self.spatial_attention_layer(self.gaze_lstm_hidden.squeeze(dim=0),
                                        cnn_feat_seq[:,i,:,:]) # (bs, 36)
            spatial_feat = cnn_feat_seq[:,i,:,:] * spatial_weight.unsqueeze(2)   # (bs, 36, 256)
            spatial_feat = spatial_feat.sum(1)      # (bs, 256)
            self.queue_spatial.append(spatial_feat)                             #[(bs, 256)]
            self.queue_hidden.append(self.gaze_lstm_hidden.squeeze(dim=0))      #[(bs, 64)]

            # calculate the video feature using temporal weight
            ts = len(self.queue_spatial)
            if ts > self.queue_size:        # update queue
                self.queue_spatial.pop(0)
                self.queue_hidden.pop(0)
                ts = self.queue_size
            queue_hidden_var = torch.cat(self.queue_hidden, dim=0).view(ts, bs, -1)     # (ts, bs, 64)
            queue_hidden_var = queue_hidden_var.transpose(1, 0)      # (bs, ts, 64)
            queue_spatial_var = torch.cat(self.queue_spatial, dim=0).view(ts, bs, -1)     # (ts, bs, 256)
            queue_spatial_var = queue_spatial_var.transpose(1, 0)      # (bs, ts, 256)
            temporal_weight = self.temporal_attention_layer(queue_hidden_var, queue_spatial_var)   # (bs, ts)
            # print(temporal_weight.size())
            # print(queue_spatial_var.size())
            temporal_feat = queue_spatial_var * temporal_weight.unsqueeze(2)      # (bs, ts, 256)
            temporal_feat = temporal_feat.sum(1)      # (bs, 256)

            # update the lstm, h: (bs, hidden_num) + f: (bs, 256)
            gaze = gaze_seq[:, i, :].unsqueeze(1)
            gaze_lstm_output, (self.gaze_lstm_hidden, self.gaze_lstm_cell) = self.gaze_lstm_layer(gaze,
                                            (self.gaze_lstm_hidden, self.gaze_lstm_cell))

            # concate lstm and feat for mlp, (bs, 256 + 64)
            feat_concat = torch.cat((temporal_feat, self.gaze_lstm_hidden[0]), dim=1)

            pred = self.mlp_layer(feat_concat)
            pred_all.append(pred.unsqueeze(0))

        pred_all = torch.cat(pred_all, 1)

        return pred_all
