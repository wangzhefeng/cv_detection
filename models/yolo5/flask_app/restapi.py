# -*- coding: utf-8 -*-

# ***************************************************
# * File        : restapi.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-27
# * Version     : 0.1.042722
# * Description : description
# * Link        : https://github.com/ultralytics/yolov5/tree/master/utils/flask_rest_api
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import io
import argparse

import torch
from flask import Flask, request
from PIL import Image

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
DETECTION_URL = "/v1/object-detection/<model>"

app = Flask(__name__)
models = {}

@app.route(DETECTION_URL, method = ["POST"])
def predict(model):
    if request.method != "POST":
        return
    
    if request.files.get("image"):
        # method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))
        # method 2
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        # inference
        if model in models:
            results = models[model](im, size = 640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient = "records")




# 测试代码 main 函数
def main():
    # arg parse
    parser = argparse.ArgumentParser(description = "Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default = 5000, type = int, help = "port number")
    parser.add_argument("--model", nargs = "+", default = ["yolov5s"], help = "model(s) to run, i.e. --model yolov5n yolov5s")
    opt = parser.parse_args()
    # model load
    for m in opt.model:
        models[m] = torch.hub.load(
            "ultralytics/yolov5", 
            m, 
            force_reload = True, 
            skip_validation = True
        )
    # app run  
    app.run(host = "0.0.0.0", port = opt.port)  # debug=True causes Restarting with stat

if __name__ == "__main__":
    main()
