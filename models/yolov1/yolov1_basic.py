# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolo.basic.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 1.0.103101
# * Description : 常用的基础模块
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    """
    convolution

    Args:
        c1: 输入通道数
        c2: 输出通道数
        k: 卷积核尺寸 
        p0: 补零的尺寸
        s1: 卷积的步长
        d1: 卷积膨胀系数
        g (_type_): _description_
        bias (bool, optional): _description_. Defaults to False.
    """
    conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

    return conv


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type is not None:
        return nn.Identity()
    else:
        raise NotImplementedError('Activation {} not implemented.'.format(act_type))


def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is not None:
        return nn.Identity()
    else:
        raise NotImplementedError('Normalization {} not implemented.'.format(norm_type))


class Conv(nn.Module):
    
    def __init__(self, 
                 c1,                     # 输入通道数
                 c2,                     # 输出通道数 
                 k = 1,                  # 卷积核尺寸 
                 p = 0,                  # 补零的尺寸
                 s = 1,                  # 卷积的步长
                 d = 1,                  # 卷积膨胀系数
                 act_type = 'lrelu',     # 激活函数的类别
                 norm_type = 'BN',       # 归一化层的类别
                 depthwise = False       # 是否使用depthwise卷积
                 ):
        super(Conv, self).__init__()
        
        convs = []
        add_bias = False if norm_type else True
        # 构建 depthwise + pointwise 卷积
        if depthwise:
            # 首先，搭建 depthwise 卷积
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias)) 
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type)) 
            # 然后，搭建 pointwise 卷积
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
        # 构建普通的标准卷积
        else:
            # conv2d
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            # batch norm
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            # activation
            if act_type:
                convs.append(get_activation(act_type))
        
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
