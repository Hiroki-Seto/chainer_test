# condig:utf-8

import os
import json
import shutil
import argparse
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import chainer.datasets as t

from chainer.datasets import cifar


class VGG(chainer.Chain):

    def __init__(self, n_class=10):
        super(VGG, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 64, 3, pad=1)
            self.bn1_1 = L.BatchNormalization(64)
            self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1)
            self.bn1_2 = L.BatchNormalization(64)

            self.conv2_1 = L.Convolution2D(64, 128, 3, pad=1)
            self.bn2_1 = L.BatchNormalization(128)
            self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1)
            self.bn2_2 = L.BatchNormalization(128)

            self.conv3_1 = L.Convolution2D(128, 256, 3, pad=1)
            self.bn3_1 = L.BatchNormalization(256)
            self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_2 = L.BatchNormalization(256)
            self.conv3_3 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_3 = L.BatchNormalization(256)
            self.conv3_4 = L.Convolution2D(256, 256, 3, pad=1)
            self.bn3_4 = L.BatchNormalization(256)

            self.fc4 = L.Convolution2D(256, 1024, 1, pad=0)
            self.fc5 = L.Convolution2D(1024, 1024, 1, pad=0)
            self.fc6 = L.Linear(None, n_class)

    def __call__(self, x):
        h = F.relu(self.bn1_1(self.conv1_1(x)))
        h = F.relu(self.bn1_2(self.conv1_2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn2_1(self.conv2_1(h)))
        h = F.relu(self.bn2_2(self.conv2_2(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.relu(self.bn3_1(self.conv3_1(h)))
        h = F.relu(self.bn3_2(self.conv3_2(h)))
        h = F.relu(self.bn3_3(self.conv3_3(h)))
        h = F.relu(self.bn3_4(self.conv3_4(h)))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, ratio=0.25)

        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5)
        h = F.dropout(F.relu(self.fc5(h)), ratio=0.5)
        h = self.fc6(h)
        return h


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config")

    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r") as f:
        d = json.load(f)
    opt_str = d["optimizer"]


    train, test = cifar.get_cifar10()
    batchsize = d["batch_size"]

    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(test, batchsize, False, False)

    model = VGG()

    max_epoch = d["max_epoch"]
    model = L.Classifier(model)
    gpu_id = 0

    if gpu_id >= 0:
        model.to_gpu(gpu_id)

    lr = d["lr"]

    if opt_str == "SGD":
            optimizer = optimizers.MomentumSGD(lr=lr)
    elif opt_str == "rms":
            optimizer = optimizers.RMSprop(lr=lr)
    elif opt_str == "adam":
            optimizer = optimizers.Adam(alpha=lr)

    base_dir = os.path.join("./", opt_str)
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out=base_dir)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.run()

    file_name = config_path.split("/")[-1]
    shutil.copy(config_path, os.path.join(base_dir, file_name))
    print("Process Finished.")

