# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042419
# * Description : description
# * Link        : https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading/#before-you-start
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import io
import base64
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
# yolov5n
# --------
# model_v5n = torch.hub.load("ultralytics/yolov5", "yolov5n")

# yolov5s
# --------
model_v5s = torch.hub.load(
    repo_or_dir = "ultralytics/yolov5", 
    model = "yolov5s", 
    device = "cpu", 
    _verbose = False, 
    # channels = 4,
    # classes = 10,
    force_reload = False,  # force reload
)
# model_v5s.conf = 0.25  # NMS confidence threshold
# model_v5s.iou = 0.45  # NMS IoU threshold
# model_v5s.agnostic = False  # NMS class-agnostic
# model_v5s.multi_label = False  # NMS multiple labels per box
# model_v5s.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
# model_v5s.max_det = 1000  # maximum number of detections per image
# model_v5s.amp = False  # Automatic Mixed Precision (AMP) inference

# yolov5m,...
# --------
# model_v5m = torch.hub.load("ultralytics/yolov5", "yolov5m")
# model_v5l = torch.hub.load("ultralytics/yolov5", "yolov5l")
# model_v5x = torch.hub.load("ultralytics/yolov5", "yolov5x")
# model_v5n6 = torch.hub.load("ultralytics/yolov5", "yolov5n6")
# model_v5s6 = torch.hub.load("ultralytics/yolov5", "yolov5s6")
# model_v5m6 = torch.hub.load("ultralytics/yolov5", "yolov5m6")
# model_v5l6 = torch.hub.load("ultralytics/yolov5", "yolov5l6")
# model_v5x6 = torch.hub.load("ultralytics/yolov5", "yolov5x6")

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
tensor_preds = model_v5s(imgs)  # default size 640
# tensor_preds.print()
# tensor_preds.show()
# tensor_preds.save()

# Box-Cropped result
crops = tensor_preds.crop(save = True)

# tensor predictions result
tensor_preds_img1 = tensor_preds.xyxy[0]
tensor_preds_img2 = tensor_preds.xyxy[1]
tensor_preds_img3 = tensor_preds.xyxy[2]
tensor_preds_img4 = tensor_preds.xyxy[3]
print(f"\ntensor preds of img1:\n {tensor_preds_img1}")
print(f"\ntensor preds of img2:\n {tensor_preds_img2}")
print(f"\ntensor preds of img3:\n {tensor_preds_img3}")
print(f"\ntensor preds of img4:\n {tensor_preds_img4}")

# pandas prediction result
pandas_preds = tensor_preds.pandas()
print(f"\nres_pandas:\n {pandas_preds}")
pandas_preds_img1 = pandas_preds.xyxy[0].sort_values("xmin").to_json(orient = "records")
pandas_preds_img2 = pandas_preds.xyxy[1].sort_values("xmin").to_json(orient = "records")
pandas_preds_img3 = pandas_preds.xyxy[2].sort_values("xmin").to_json(orient = "records")
pandas_preds_img4 = pandas_preds.xyxy[3].sort_values("xmin").to_json(orient = "records")
print(f"\npandas preds of img1:\n {pandas_preds_img1}")
print(f"\npandas preds of img2:\n {pandas_preds_img2}")
print(f"\npandas preds of img3:\n {pandas_preds_img3}")
print(f"\npandas preds of img4:\n {pandas_preds_img4}")






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
