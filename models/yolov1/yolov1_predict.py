# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolov1_predict.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-11-01
# * Version     : 0.1.110116
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

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class ObjPred(nn.Module):
    """
    预测层(prediction layer)
    边界框置信度预测(1)
    """

    def __init__(self, head_dim):
        super(ObjPred, self).__init__()
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size = 1)
    
    def forward(self, x):
        out = self.obj_pred(x)  # [B, 13, 13, 1]

        return out


class ClsPred(nn.Module):
    """
    预测层(prediction layer)
    类别预测概率(num_classes)
    """

    def __init__(self, head_dim, num_classes):
        super(ClsPred, self).__init__() 
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size = 1)
    
    def forward(self, x):
        out = self.cls_pred(x)  # [B, 13, 13, num_classes]

        return out


class RegPred(nn.Module):
    """
    预测层(prediction layer)
    位置参数预测(4)
    """

    def __init__(self, head_dim):
        super(RegPred, self).__init__()
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size = 1)
    
    def forward(self, x):
        out = self.reg_pred(x)  # [B, 13, 13, 4]

        return out


def build_pred(head_dim, num_classes):
    print('==============================')
    print('PredLayer:')
    # build pred layer
    obj_pred = ObjPred(head_dim)
    cls_pred = ClsPred(head_dim, num_classes)
    reg_pred = RegPred(head_dim)

    return obj_pred, cls_pred, reg_pred




# 测试代码 main 函数
def main():
    head_dim = 512
    num_classes = 20
    obj_pred, cls_pred, reg_pred = build_pred(head_dim, num_classes)
    print(obj_pred)
    print(cls_pred)
    print(reg_pred)

if __name__ == "__main__":
    main()
