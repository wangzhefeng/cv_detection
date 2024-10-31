# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolo5_multiGPU.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-27
# * Version     : 0.1.042718
# * Description : description
# * Link        : https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#multi-gpu-inference
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import glob
import threading
from pathlib import Path

import cv2
from PIL import Image, ImageGrab
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = os.path.join(Path(__file__).parent.parent.parent, "data/images")

# ------------------------------
# model
# ------------------------------
model_v5s_1 = torch.hub.load(repo_or_dir = "ultralytics/yolov5", model = "yolov5s", device = 0)
model_v5s_2 = torch.hub.load(repo_or_dir = "ultralytics/yolov5", model = "yolov5s", device = 1)

# ------------------------------
# images
# ------------------------------
imgs = []
# cloud image
img_cloud = "https://ultralytics.com/images/zidane.jpg"
imgs.append(img_cloud)
# local data
for img_name in ["zidane.jpg", "bus.jpg"]:
    img_dir = os.path.join(data_dir, img_name)
    if not os.path.exists(img_dir):
        torch.hub.download_url_to_file("https://ultralytics.com/images/" + img_name, img_dir)
    imgs.append(Image.open(img_dir))  # PIL image
    # imgs.append(cv2.imread("bus.jpg")[..., ::-1])  # OpenCV image (BGR to RGB)
# screenshot inference
img_screenshot = ImageGrab.grab()  # take a screenshot
imgs.append(img_screenshot)

# ------------------------------
# inference
# ------------------------------
def run(model, img):
    results = model(img)
    results.save()


threading.Thread(
    target = run, 
    args = [
        model_v5s_1, 
        "https://ultralytics.com/images/zidane.jpg",
    ], 
    daemon = True
).start()
threading.Thread(
    target = run, 
    args = [
        model_v5s_2, 
        "https://ultralytics.com/images/bus.jpg",
    ], 
    daemon = True
).start()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
