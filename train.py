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

from model_builder import Builder


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
    
    builder = Builder(model_type="VGG")
    model = builder.build()

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

    base_dir = os.path.join("./log", opt_str)
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

