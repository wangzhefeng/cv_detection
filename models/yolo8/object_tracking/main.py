# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-29
# * Version     : 0.1.042900
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))

from ultralytics import YOLO
from ultralytics.yolo.utils import set_logging
from PIL import Image

set_logging(verbose = False)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# model
model = YOLO("yolov8n.pt")

# model predict
preds = model.track(source = 0, save = True, show = True)

for p in preds[::20]:
    print(p.boxes.id)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
