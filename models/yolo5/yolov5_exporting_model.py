# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolov5_exporting_model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-27
# * Version     : 0.1.042723
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# models
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.pt")  # PyTorch
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.torchscript")  # TorchScript
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.onnx")  # ONNX
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s_openvino_model/")  # OpenVINO
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.engine")  # # TensorRT
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.mlmodel")  # CoreML(macOS-only)
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s.tflite")  # TFLite
model = torch.hub.load("ultralytics/yolov5", "custom", path = "yolov5s_paddle_model/")  # PaddlePaddle





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
