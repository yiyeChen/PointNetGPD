#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from scipy.stats import mode
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath("__file__"))))

########### Yiye: Setting
load_model = path.join(
    path.dirname(path.dirname(path.abspath(__file__))),
    "data/pointnetgpd_3class.model"
)
cuda = True
gpu = 0

model = torch.load(load_model, map_location="cpu")
model.device_ids = [gpu]
print("load model {}".format(load_model))

if cuda:
    model = torch.load(load_model, map_location="cuda:{}".format(gpu))
    if gpu != -1:
        torch.cuda.set_device(gpu)
        model = model.cuda()
    else:
        device_id = [0, 1]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
if isinstance(model, torch.nn.DataParallel):
    model = model.module


def test_network(model_, local_pc):
    local_pc = local_pc.T
    local_pc = local_pc[np.newaxis, ...]
    local_pc = torch.FloatTensor(local_pc)
    if cuda:
        local_pc = local_pc.cuda()
    output, _ = model_(local_pc)  # N*C
    output = output.softmax(1)
    pred = output.data.max(1, keepdim=True)[1].cpu()
    output = output.cpu()
    return pred[0], output.data.numpy()


def main():
    repeat = 10
    num_point = 500
    model.eval()
    torch.set_grad_enabled(False)

    # load pc(should be in local gripper coordinate)
    # local_pc: (N, 3)
    # local_pc = np.load("test.npy")
    local_pc = np.random.random([500, 3])  # test only
    predict = []
    for _ in range(repeat):
        if len(local_pc) >= num_point:
            local_pc = local_pc[np.random.choice(len(local_pc), num_point, replace=False)]
        else:
            local_pc = local_pc[np.random.choice(len(local_pc), num_point, replace=True)]

        # run model
        predict.append(test_network(model, local_pc)[0])
    print("voting: ", predict)
    predict = mode(predict).mode[0]

    # output
    print("Test result:", predict)


if __name__ == "__main__":
    main()
