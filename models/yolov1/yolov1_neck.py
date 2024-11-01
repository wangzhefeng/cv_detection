# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolov1_neck.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 1.0.103101
# * Description : YOLOv1 的颈部网络: SPP
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

import torch
import torch.nn as nn

try:
    from .yolov1_basic import Conv
except:
    from yolov1_basic import Conv

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

 
class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling(空间金字塔池化)
    该代码参考 YOLOv5 的官方代码实现 https://github.com/ultralytics/yolov5
    """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 expand_ratio = 0.5, 
                 pooling_size = 5, 
                 act_type = 'lrelu', 
                 norm_type = 'BN'):
        super(SPPF, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.cv1 = Conv(in_dim, inter_dim, k = 1, act_type = act_type, norm_type = norm_type)
        self.cv2 = Conv(inter_dim * 4, out_dim, k = 1, act_type = act_type, norm_type = norm_type)
        self.m = nn.MaxPool2d(kernel_size = pooling_size, stride = 1, padding = pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)                               # [B, 13, 13, 512] -> [B, 13, 13, 256]
        y1 = self.m(x)                                # [B, 13, 13, 256]
        y2 = self.m(y1)                               # [B, 13, 13, 256]
        y3 = self.m(y2)                               # [B, 13, 13, 256]
        concat = torch.cat((x, y1, y2, y3), dim = 1)  # [B, 13, 13, 1024]
        out = self.cv2(concat)                        # [B, 13, 13, 512]
        
        return out


def build_neck(cfg, in_dim, out_dim):
    """
    搭建 Neck 网络
    """
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'sppf':
        neck = SPPF(
            in_dim = in_dim,
            out_dim = out_dim,
            expand_ratio = cfg['expand_ratio'], 
            pooling_size = cfg['pooling_size'],
            act_type = cfg['neck_act'],
            norm_type = cfg['neck_norm']
        )
    else:
        raise NotImplementedError('Neck {} not implemented.'.format(cfg['neck']))

    return neck




# 测试代码 main 函数
def main():
    cfg = {
        'neck': 'sppf',                 # 使用 SPP 作为颈部网络
        'expand_ratio': 0.5,            # SPP 的模型参数
        'pooling_size': 5,              # SPP 的模型参数
        'neck_act': 'lrelu',            # SPP 的模型参数
        'neck_norm': 'BN',              # SPP 的模型参数
        'neck_depthwise': False,        # SPP 的模型参数
    }
    neck = build_neck(cfg, in_dim = 512, out_dim = 512)
    print(neck)
    print(neck.out_dim)

if __name__ == "__main__":
    main()
