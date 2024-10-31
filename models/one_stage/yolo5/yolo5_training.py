# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolo5_training.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-27
# * Version     : 0.1.042718
# * Description : description
# * Link        : https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# ------------------------------
# model
# ------------------------------
# load pretrained
model = torch.hub.load("ultralytics/yolov5", "yolov5s", autoshape = False)
# load scratch
model = torch.hub.load("ultralytics/yolov5", "yolov5s", autoshape = False, pretrained = False)








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
