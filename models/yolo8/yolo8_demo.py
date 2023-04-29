# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolo8.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-26
# * Version     : 0.1.042621
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


"""
# create yolo model from scratch
model = YOLO("yolov8n.yaml")

# Training
model = YOLO("yolov8n.pt")
results = model.train(data = "coco128.yaml", epochs = 3)
results = model.val()
success = model.export(format = "onnx")

# Predict-object detection
results = model("https://ultralytics.com/images/bus.jpg")
"""




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
