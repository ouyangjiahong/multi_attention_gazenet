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

from model import FeatureExtractor, SpatialAttentionModel, MultipleAttentionModel
from util import *
from logger import Logger
import gazeWholeGenerator as gaze_gen


def train(train_data, extractor_model, model, criterion, optimizer, epoch, logger, para):
    bs = para['bs']
    img_size = para['img_size']
    num_class = para['num_class']
    print_freq = para['print_freq']

    model.train()

    print("training interaction number")
    train_num = len(train_data)
    # train_num = 2
    # print(train_num)
    end = time.time()
    for i in range(train_num):
        # get data, img_seq: (ts,224,224,3), gaze_seq: (ts, 3), ouput: (ts, 6)
        [img_seq, gaze_seq], target_seq = next(train_data)
        img_seq = img_seq[:35]          # just for speed up
        gaze_seq = gaze_seq[:35]
        target_seq = target_seq[:35]
        ts = img_seq.shape[0]

        img_seq = normalize(img_seq)
        img_seq = np.reshape(img_seq, (bs*ts,3,) + img_size)
        img_seq_var = torch.autograd.Variable(torch.Tensor(img_seq), requires_grad=True)
        gaze_seq_var = torch.autograd.Variable(torch.Tensor(gaze_seq), requires_grad=True)
        target_seq_var = torch.autograd.Variable(torch.Tensor(target_seq)).long()
        gaze_seq_var = gaze_seq_var.unsqueeze(0)        # no bs dim

        # extract cnn feature
        # print(img_seq_var.size())
        cnn_feat_var = extractor_model(img_seq_var)
        cnn_feat_var = cnn_feat_var.view((bs, ts, -1, cnn_feat_var.size()[2]*cnn_feat_var.size()[3]))
        cnn_feat_var = cnn_feat_var.permute(0, 1, 3, 2)
        # print("cnn fetaure extractor")
        # print(cnn_feat_var.size())          # (bs, ts, 36, 256)

        optimizer.zero_grad()
        prediction = model(cnn_feat_var, gaze_seq_var)
        prediction = prediction.view((bs*ts, num_class))

        # print(target_seq_var)
        target_seq_var = target_seq_var.view(bs*ts)

        loss = criterion(prediction, target_seq_var)
        # print(loss.data)
        loss.backward()
        optimizer.step()
        time_cnt = time.time() - end
        end = time.time()

        if i % print_freq == 0:
            prediction = F.softmax(prediction, dim=1)
            acc_frame = metric_frame(prediction, target_seq_var)
            acc_frame = acc_frame / (1.0 * ts)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {time_cnt:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Accuracy {acc:.4f}\t'.format(
                   epoch, i, train_num, time_cnt=time_cnt, loss=loss.data[0],
                   acc=acc_frame))
            global_step = epoch * train_num + i
            logger.scalar_summary('train/loss', loss.data[0], global_step)
            logger.scalar_summary('train/acc', acc_frame, global_step)


def validate(val_data, extractor_model, model, criterion, epoch, logger, para,
                visualize=False, vis_data_path=''):
    bs = para['bs']
    img_size = para['img_size']
    num_class = para['num_class']
    # print_freq = para['print_freq']

    model.eval()

    print("validation interaction number")
    val_num = len(val_data)
    # val_num = 2
    # print(val_num)
    end = time.time()
    loss_sum = 0
    ts_all = 0
    acc_all = 0
    for i in range(val_num):
        # get data, img_seq: (ts,224,224,3), gaze_seq: (ts, 3), ouput: (ts, 6)
        [img_seq, gaze_seq], target_seq = next(val_data)
        img_seq = img_seq[:30]          # just for speed up
        gaze_seq = gaze_seq[:30]
        target_seq = target_seq[:30]
        ts = img_seq.shape[0]
        ts_all += ts

        img_seq = normalize(img_seq)
        img_seq = np.reshape(img_seq, (bs*ts,3,) + img_size)
        img_seq_var = torch.autograd.Variable(torch.Tensor(img_seq))
        gaze_seq_var = torch.autograd.Variable(torch.Tensor(gaze_seq))
        target_seq_var = torch.autograd.Variable(torch.Tensor(target_seq)).long()
        gaze_seq_var = gaze_seq_var.unsqueeze(0)        # no bs dim

        # extract cnn feature
        # print(img_seq_var.size())
        cnn_feat_var = extractor_model(img_seq_var)
        # print(cnn_feat_var.size())
        cnn_feat_var = cnn_feat_var.view((bs, ts, -1, cnn_feat_var.size()[2]*cnn_feat_var.size()[3]))
        cnn_feat_var = cnn_feat_var.permute(0, 1, 3, 2)
        # print("cnn fetaure extractor")
        # print(cnn_feat_var.size())          # (bs, ts, 36, 256)

        prediction = model(cnn_feat_var, gaze_seq_var)
        prediction = prediction.view((bs*ts, num_class))

        # print(target_seq_var)
        target_seq_var = target_seq_var.view(bs*ts)

        # loss and accuracy
        loss = criterion(prediction, target_seq_var)
        loss_sum += loss.data[0]
        # print(loss.data)

        prediction = F.softmax(prediction, dim=1)
        acc_frame = metric_frame(prediction, target_seq_var)
        acc_all += acc_frame
        time_cnt = time.time() - end
        end = time.time()

        acc_cur = acc_frame / float(ts)
        print('Epoch: [{0}]\t'
              'Loss {loss:.4f}\t'
              'Accuracy {acc:.4f}\t'.format(
               epoch, i, val_num, time_cnt=time_cnt, loss=loss.data[0], acc=acc_cur))

        # visualize
        if visualize and acc_cur < 0.5:
            visualization(i, acc_cur, img_seq, gaze_seq, target_seq_var, prediction, vis_data_path)

    loss_avg = loss_sum / float(val_num)
    acc_avg = acc_all / float(ts_all)

    print('Epoch: [{0}]\t'
          'Loss {loss:.4f}\t'
          'Accuracy {acc:.4f}\t'.format(
           epoch, i, val_num, time_cnt=time_cnt, loss=loss_avg, acc=acc_avg))
    logger.scalar_summary('val/loss', loss_avg, epoch)
    logger.scalar_summary('val/acc', acc_avg, epoch)
    return acc_avg


def visualization(iter, acc_cur, img_seq, gaze_seq, target_seq_var, prediction, vis_data_path):
    subdir_path = vis_data_path + str(iter) + '_' + str('%.3f'%acc_cur) + '/'
    print(subdir_path)
    os.makedirs(subdir_path)

    num_frame = img_seq.shape[0]
    img_seq = denormalize(img_seq)

    _, prediction = torch.max(prediction, 1)
    prediction = prediction.data.cpu().numpy()
    target_seq = target_seq_var.data.cpu().numpy()
    for i in range(num_frame):
        target = target_seq[i]
        predict = prediction[i]
        img = img_seq[i,:,:,:]
        gaze = gaze_seq[i,:]

        x = img.shape[1] * gaze[1]
        y = img.shape[0] * (1 - gaze[2])
        left = int(max(0, x-5))
        right = int(min(x+5, img.shape[1]-1))
        above = int(max(0, y-5))
        bottom = int(min(y+5, img.shape[0]-1))
        img[above:bottom, left:right, 0] = 0
        img[above:bottom, left:right, 1] = 	0
        img[above:bottom, left:right, 2] = 	1

        img_path = subdir_path + str('%3d'%i) + '_' + str(target) + \
                    '_' + str(predict) + '.jpg'
        # imsave(img_path, img)

def main():
    # define parameters
    TRAIN = True
    num_class = 6
    batch_size = 1
    time_step = 32
    epochs = 50
    cnn_feat_size = 256     # AlexNet
    gaze_size = 3
    gaze_lstm_hidden_size = 64
    gaze_lstm_projected_size = 128
    temporal_projected_size = 128
    queue_size = 32
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    eval_freq = 1       # epoch
    print_freq = 1      # iteration
    dataset_path = '../data/gaze_dataset'
    # dataset_path = '../../gaze-net/gaze_dataset'
    img_size = (224, 224)
    log_path = '../log'
    logger = Logger(log_path, 'multiple')

    # define model
    arch = 'alexnet'
    extractor_model = FeatureExtractor(arch=arch)
    extractor_model.features = torch.nn.DataParallel(extractor_model.features)
    extractor_model.cuda()      # uncomment this line if using cpu
    extractor_model.eval()

    model = MultipleAttentionModel(num_class, cnn_feat_size,
                        gaze_size, gaze_lstm_hidden_size, gaze_lstm_projected_size,
                        temporal_projected_size, queue_size)
    model.cuda()

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum = momentum, weight_decay=weight_decay)

    # define generator
    trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0)
    train_data = trainGenerator.flow_from_directory(dataset_path, subset='training', crop=False,
                    batch_size=batch_size, target_size= img_size, class_mode='sequence_pytorch')
    # small dataset, error using validation split
    # val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation', crop=False,
    #             batch_size=batch_size, target_size= img_size, class_mode='sequence_pytorch')
    val_data = train_data

    # img_seq: (ts,224,224,3), gaze_seq: (ts, 3), ouput: (ts, 6)
    # [img_seq, gaze_seq], output = next(train_data)
    # print("gaze data shape")
    # print(img_seq.shape)
    # print(gaze_seq.shape)
    # print(output.shape)

    # start Training
    para = {'bs': batch_size, 'img_size': img_size, 'num_class': num_class,
            'print_freq': print_freq}
    if TRAIN:
        print("get into training mode")
        best_acc = 0

        for epoch in range(epochs):
            adjust_learning_rate(optimizer, epoch, learning_rate)
            print('Epoch: {}'.format(epoch))
            # train for one epoch
            train(train_data, extractor_model, model, criterion, optimizer, epoch, logger, para)

            # evaluate on validation set
            if epoch % eval_freq == 0 or epoch == epochs - 1:
                acc = validate(val_data, extractor_model, model, criterion, epoch, logger, para, False)
                is_best = acc > best_acc
                best_acc = max(acc, best_acc)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best)
    else:
        model = load_checkpoint(model)
        print("get into testing and visualization mode")
        print("visualization for training data")
        vis_data_path = '../vis/train/'
        if not os.path.exists(vis_data_path):
            os.makedirs(vis_data_path)
        acc = validate(train_data, extractor_model, model, criterion, -1, \
                        logger, para, True, vis_data_path)
        print("visualization for validation data")
        vis_data_path = '../vis/val/'
        if not os.path.exists(vis_data_path):
            os.makedirs(vis_data_path)
        acc = validate(val_data, extractor_model, model, criterion, -1, \
                        logger, para, True, vis_data_path)

def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='../model/spatial/checkpoint.pth.tar'):
    print("save checkpoint")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../model/spatial/model_best.pth.tar')

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
