# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-28
# * Version     : 0.1.042822
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from ultralytics import YOLO

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model
model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# model training
model.train(
    data = "coco128.yaml", 
    epochs = 100, 
    patience = 50,
    batch = 16,
    imgsz = 640,
    save = True,
    save_period = -1,
    cache = False,
    device = None,
    workers = 8,
    project = None,
    name = None,
    exist_ok = False,
    pretrained = False,
    optimizer = "SGD",
    verbose = False,
    seed = 0,
    deterministic = True,
    single_cls = False,
    rect = False,
    cos_lr = False,
    clone_mosaic = 0,
    resume = False,
    amp = True,
    lr0 = 0.01,
    lrf = 0.01,
    momentum = 0.937,
    weight_decay = 0.0005,
    warmup_epochs = 3.0,
    warmup_momentum = 0.8,
    warmup_bias_lr = 0.1,
    box = 7.5,
    cls = 0.5,
    df1 = 1.5,
    pose = 12.0,
    kobj = 2.0,
    label_smoothing = 0.0,
    nbs = 64,
    overlap_mask = True,
    mask_ratio = 4,
    dropout = 0.0,
    val = True,
)






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
