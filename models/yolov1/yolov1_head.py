# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolov1_head.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 1.0.103101
# * Description : description
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


class DecoupledHead(nn.Module):
    """
    解偶检测头
    """

    def __init__(self, cfg, in_dim, out_dim, num_classes = 80):
        super(DecoupledHead, self).__init__()
        print('==============================')
        print('Head: Decoupled Head')
        self.in_dim = in_dim
        self.num_cls_head = cfg['num_cls_head']
        self.num_reg_head = cfg['num_reg_head']
        self.act_type = cfg['head_act']
        self.norm_type = cfg['head_norm']
        # ------------------ 类别检测头 ------------------
        cls_feats = []
        self.cls_out_dim = max(out_dim, num_classes)
        for i in range(cfg['num_cls_head']):
            if i == 0:
                cls_feats.append(Conv(
                    in_dim, 
                    self.cls_out_dim, 
                    k=3, p=1, s=1, 
                    act_type = self.act_type,
                    norm_type = self.norm_type,
                    depthwise = cfg['head_depthwise']
                ))
            else:
                cls_feats.append(Conv(
                    self.cls_out_dim, 
                    self.cls_out_dim, 
                    k=3, p=1, s=1, 
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    depthwise=cfg['head_depthwise']
                ))
        # ------------------ 回归检测头 ------------------
        reg_feats = []
        self.reg_out_dim = max(out_dim, 64)
        for i in range(cfg['num_reg_head']):
            if i == 0:
                reg_feats.append(Conv(
                    in_dim, 
                    self.reg_out_dim, 
                    k=3, p=1, s=1, 
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    depthwise=cfg['head_depthwise']
                ))
            else:
                reg_feats.append(Conv(
                    self.reg_out_dim, 
                    self.reg_out_dim, 
                    k=3, p=1, s=1, 
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    depthwise=cfg['head_depthwise']
                ))
        # ------------------------------
        # 类别、回归检测头
        # ------------------------------
        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

    def forward(self, x):
        """
        x: (torch.Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats


def build_head(cfg, in_dim, out_dim, num_classes=80):
    """
    构建解偶检测头

    输入参数的解释：
        cfg:         (Dict)  网络的config变量
        in_dim:      (Int)   输入特征图的通道数
        out_dim:     (Int)   输出特征图的通道数
        num_classes: (Int)   检测类别的数量
    """
    head = DecoupledHead(cfg, in_dim, out_dim, num_classes) 

    return head
 


# 测试代码 main 函数
def main():
    # ------------------------------
    # 检测头
    # ------------------------------
    # 检测头的结构参数
    cfg = {
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_act': 'silu',
        'head_norm': 'BN',
        'head_depthwise': False,
        # 'reg_max': 16,
    }

    # 输入特征的参数
    batch_size   = 2
    feat_channel = 512
    feat_height  = 13
    feat_width   = 13
    # 随机生成一张图像
    feature = torch.randn(batch_size, feat_channel, feat_height, feat_width)
    print(f"input size: {feature.shape}")
    
    # 搭建检测头
    model = build_head(
        cfg = cfg,
        in_dim = feat_channel,
        out_dim = 512,
        num_classes = 20,
    )
    print(model)
    
    # 模型推理
    cls_feat, reg_feat = model(feature)
    # 查看模型的输出的shape
    print(cls_feat.shape)
    print(reg_feat.shape)
    # ------------------------------
    # 预测层
    # ------------------------------
    # 置信度预测
    obj_pred = nn.Conv2d(512, 1, kernel_size = 1)
    # 类别预测
    cls_pred = nn.Conv2d(512, 20, kernel_size = 1)
    # 位置参数预测
    reg_pred = nn.Conv2d(512, 4, kernel_size = 1)

if __name__ == "__main__":
    main()
