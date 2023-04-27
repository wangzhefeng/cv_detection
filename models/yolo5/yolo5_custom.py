# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolo5_custom.py
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


# local model
model = torch.hub.load("ultralytics/yolov5", "custom", path = "path/to/best.pt")
# local repo
model = torch.hub.load("path/to/yolov5", "custom", path = "path/to/best.pt", source = "local")





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
