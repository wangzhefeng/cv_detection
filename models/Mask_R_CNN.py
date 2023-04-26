# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Mask_R_CNN.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042419
# * Description : description
# * Link        : https://mp.weixin.qq.com/s/UBPbPhewk2sBa8td9wy-CA
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model pre-trained on COCO
model = torchvision \
    .models \
    .detection \
    .fasterrcnn_resnet50_fpn(pretrained = True) \
    .to(device)

# 1 class scratch+ background
num_classes = 2

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace th epre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    num_classes,
)











# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
