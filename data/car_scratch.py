# -*- coding: utf-8 -*-

# ***************************************************
# * File        : car_scratch.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-26
# * Version     : 0.1.042614
# * Description : description
# * Link        : https://mp.weixin.qq.com/s/UBPbPhewk2sBa8td9wy-CA
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os

from bs4 import BeautifulSoup
import requests

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data path
img_dir = "/Users/zfwang/learn/machinelearning/data/image_seg_det/car_scratch"


def get_car_scratch():
    # request
    url = "https://stock.adobe.com/in/search/images?k=car%20scratch"
    response = requests.get(url)
    # soup
    soup = BeautifulSoup(response.text, "html.parser")
    # images
    images = soup.find_all("img")
    
    data = {}
    for image in images:
        # name
        name = image["alt"]
        if name == "":
            continue
        # link
        try:
            link = image["data-lazy"]
        except:
            link = image["src"]
        if link[-3:] in ["svg", "gif"]:
            continue
        
        if not data:
            data[name] = link
            continue
    
        if name in data.keys():
            name = name + "_1"
        
        data[name] = link
    
    for name, link in data.items():
        # get images
        image_path = os.path.join(
            img_dir, 
            name.replace(" ", "-") \
                .replace("//-", "") \
                .replace("-.-", "-") \
                .replace(".-", "-") \
                .replace("...-", "-") \
                .replace("...", "-") + ".jpg"
        )
        if not os.path.exists(image_path):
            with open(image_path, "wb") as f:
                img = requests.get(link)
                f.write(img.content)
                print(f"downloaded: {name, link}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
