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
if os.path.join(_path, "../../..") not in sys.path:
    sys.path.append(os.path.join(_path, "../../.."))

from ultralytics import YOLO
from PIL import Image

from data.example_image import get_example_image

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# model
model = YOLO("yolov8n.pt")

# image
img_name = "park.jpg"
img_path, pil_img = get_example_image(img_name)

# model predict
preds = model.predict(source = img_path, save_txt = True, save = True)

# 查看预测结果
Image.open(model.predictor.save_dir/img_name)

# 摄像头作为输入
model.predict(source = 0, show = True)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
