# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolo.py
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
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


model = torch.hub.load("ultralytics/yolov5", "yolov5s")

"""
cd yolov5 && python train.py --img 320 --batch 16 --epochs 50 --data carScr.yaml --weights last.pt
cd yolov5 && python train.py --img 320 --batch 32 --epochs 10 --data carScr.yaml --weights yolo5s.pt --cache --evolve
"""



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
