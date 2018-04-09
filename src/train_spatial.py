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
import torch.nn.parallel
import numpy as np

from model import FeatureExtractor, SpatialAttentionModel
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
        # img_seq = img_seq[:30]          # just for speed up
        # gaze_seq = gaze_seq[:30]
        # target_seq = target_seq[:30]
        ts = img_seq.shape[0]

        img_seq = normalize(img_seq)
        img_seq = np.reshape(img_seq, (bs*ts,3,) + img_size)
        img_seq_var = torch.autograd.Variable(torch.Tensor(img_seq).cuda(), requires_grad=True)
        gaze_seq_var = torch.autograd.Variable(torch.Tensor(gaze_seq).cuda(), requires_grad=True)
        target_seq_var = torch.autograd.Variable(torch.Tensor(target_seq).cuda()).long()
        gaze_seq_var = gaze_seq_var.unsqueeze(0)        # no bs dim

        # extract cnn feature
        # print(img_seq_var.size())
        cnn_feat_var = extractor_model(img_seq_var)
        # print(cnn_feat_var.size())
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


def validate(val_data, extractor_model, model, criterion, epoch, logger, para):
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
        img_seq_var = torch.autograd.Variable(torch.Tensor(img_seq).cuda())
        gaze_seq_var = torch.autograd.Variable(torch.Tensor(gaze_seq).cuda())
        target_seq_var = torch.autograd.Variable(torch.Tensor(target_seq).cuda()).long()
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

        loss = criterion(prediction, target_seq_var)
        loss_sum += loss.data[0]
        # print(loss.data)
        acc_frame = metric_frame(prediction, target_seq_var)
        acc_all += acc_frame
        time_cnt = time.time() - end
        end = time.time()

    loss_avg = loss_sum/float(val_num)
    acc_avg = acc_all/float(ts_all)

    print('Epoch: [{0}]\t'
          'Loss {loss:.4f}\t'
          'Accuracy {acc:.4f}\t'.format(
           epoch, i, val_num, time_cnt=time_cnt, loss=loss_avg, acc=acc_avg))
    logger.scalar_summary('val/loss', loss_avg, epoch)
    logger.scalar_summary('val/acc', acc_avg, epoch)
    return acc_avg

def metric_frame(output, target):
    correct = 0
    num_frame = target.size()[0]
    _, prediction = torch.max(output.data, 1)
    correct += (prediction == target.data).sum()
    return correct

def metric_interaction(prediction, target):
    pass

def main():
    # define parameters
    num_class = 6
    batch_size = 1
    time_step = 32
    epochs = 50
    cnn_feat_size = 256     # AlexNet
    gaze_size = 3
    gaze_lstm_hidden_size = 64
    gaze_lstm_projected_size = 128
    learning_rate = 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    eval_freq = 1       # epoch
    print_freq = 1      # iteration
    # dataset_path = '../data/gaze_dataset'
    dataset_path = '../../gaze-net/gaze_dataset'
    img_size = (224, 224)
    log_path = '../log'
    logger = Logger(log_path, 'spatial')

    # define model
    arch = 'alexnet'
    extractor_model = FeatureExtractor(arch=arch)
    extractor_model.features = torch.nn.DataParallel(extractor_model.features)
    extractor_model.cuda()      # uncomment this line if using cpu
    extractor_model.eval()

    model = SpatialAttentionModel(num_class, cnn_feat_size,
                        gaze_size, gaze_lstm_hidden_size, gaze_lstm_projected_size)
    model.cuda()

    # define loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate,
                                momentum = momentum, weight_decay=weight_decay)

    # define generator
    trainGenerator = gaze_gen.GazeDataGenerator(validation_split=0.2)
    train_data = trainGenerator.flow_from_directory(dataset_path, subset='training',
                    batch_size=batch_size, target_size= img_size, class_mode='sequence_pytorch')
    # small dataset, error using validation split
    val_data = trainGenerator.flow_from_directory(dataset_path, subset='validation',
                batch_size=batch_size, target_size= img_size, class_mode='sequence_pytorch')
    # val_data = train_data

    # img_seq: (ts,224,224,3), gaze_seq: (ts, 3), ouput: (ts, 6)
    # [img_seq, gaze_seq], output = next(train_data)
    # print("gaze data shape")
    # print(img_seq.shape)
    # print(gaze_seq.shape)
    # print(output.shape)

    # start Training
    para = {'bs': batch_size, 'img_size': img_size, 'num_class': num_class,
            'print_freq': print_freq}
    best_acc = 0
    # validate(val_data, extractor_model, model, criterion, 0, logger, para)

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        print 'Epoch: {}'.format(epoch)
        # train for one epoch
        train(train_data, extractor_model, model, criterion, optimizer, epoch, logger, para)

        # evaluate on validation set
        if epoch % eval_freq == 0 or epoch == epochs - 1:
            acc = validate(val_data, extractor_model, model, criterion, epoch, logger, para)
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best)


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


if __name__ == '__main__':
    main()
