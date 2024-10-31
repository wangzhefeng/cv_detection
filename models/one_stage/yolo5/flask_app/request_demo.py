# -*- coding: utf-8 -*-

# ***************************************************
# * File        : request_demo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-27
# * Version     : 0.1.042722
# * Description : description
# * Link        : https://github.com/ultralytics/yolov5/tree/master/utils/flask_rest_api
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import pprint
from flask import request

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"
IMAGE = "../../data/images/zidane.jpg"


# read image
with open(IMAGE, "rb") as f:
    image_data = f.read()

# request
response = request.post(
    DETECTION_URL, 
    files = {
        "image": image_data,
    }
).json()

pprint.print(response)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
