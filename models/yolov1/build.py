# -*- coding: utf-8 -*-

# ***************************************************
# * File        : build.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-01
# * Version     : 0.1.110100
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

from loss import build_criterion
from yolov1 import YOLOv1

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def build_yolov1(args, cfg, device, num_classes = 80, trainable = False, deploy = False):
    """
    构建 YOLOv1 网络

    Args:
        args (_type_): _description_
        cfg (_type_): _description_
        device (_type_): _description_
        num_classes (int, optional): _description_. Defaults to 80.
        trainable (bool, optional): _description_. Defaults to False.
        deploy (bool, optional): _description_. Defaults to False.
    """
    print("==============================")
    print(f"Build {args.model.upper()} ...")

    print("==============================")
    print(f"Model Configuration: \n {cfg}")
    # ------------------------------
    # 构建 YOLOv1
    # ------------------------------
    model = YOLOv1(
        cfg = cfg,
        device = device,
        img_size = args.img_size,
        num_classes = num_classes,
        conf_thresh = args.conf_thresh,
        nms_thresh = args.nms_thresh,
        trainable = trainable,
        deploy = deploy,
    )
    # ------------------------------
    # 初始化 YOLOv1 的 pred 层参数
    # ------------------------------
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
    
    # obj pred
    b = model.obj_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad = True)
    
    # cls pred
    b = model.cls_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad = True)
    
    # reg pred
    b = model.reg_pred.bias.view(-1, )
    b.data.fill_(1.0)
    model.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad = True)
    
    w = model.reg_pred.weight
    w.data.fill_(0.0)
    model.reg_pred.weight = torch.nn.Parameter(w, requires_grad = True)
    # ------------------------------
    # 构建用于计算标签分配和计算损失的 Criterion 类
    # ------------------------------
    criterion = build_criterion(cfg, device, num_classes) if trainable else None
   
    return model, criterion




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
