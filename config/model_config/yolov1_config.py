# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolov1_config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 1.0.103100
# * Description : YOLOv1 Config
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = [
    "yolov1_cfg"
]

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


yolov1_cfg = {
    # input
    'trans_type': 'ssd',            # 使用 SSD 风格的数据增强
    'multi_scale': [0.5, 1.5],      # 多尺度的范围
    # backbone
    'backbone': 'resnet18',         # 使用 ResNet-18 作为主干网络
    'pretrained': True,             # 加载预训练权重
    'stride': 32,  # P5             # 网络的最大输出步长
    'max_stride': 32,               # TODO
    # neck
    'neck': 'sppf',                 # 使用 SPP 作为颈部网络
    'expand_ratio': 0.5,            # SPP 的模型参数
    'pooling_size': 5,              # SPP 的模型参数
    'neck_act': 'lrelu',            # SPP 的模型参数
    'neck_norm': 'BN',              # SPP 的模型参数
    'neck_depthwise': False,        # SPP 的模型参数
    # head
    'head': 'decoupled_head',       # 使用解耦检测头
    'head_act': 'lrelu',            # 检测头所需的参数
    'head_norm': 'BN',              # 检测头所需的参数
    'num_cls_head': 2,              # 解耦检测头的分类分支所包含的卷积层数
    'num_reg_head': 2,              # 解耦检测头的回归分支所包含的卷积层数
    'head_depthwise': False,        # 检测头所需的参数
    # loss weight
    'loss_obj_weight': 1.0,         # obj 置信度损失的权重
    'loss_cls_weight': 1.0,         # cls 类别损失的权重
    'loss_box_weight': 5.0,         # box 边界框位置参数损失的权重
    # training configuration
    'no_aug_epoch': -1,             # 关闭马赛克增强和混合增强的节点
    'trainer_type': 'yolov8',       # TODO
    # optimizer
    'optimizer': 'sgd',             # 使用 SGD 优化器
    'momentum': 0.937,              # SGD 优化器的 momentum 参数
    'weight_decay': 5e-4,           # SGD 优化器的 weight_decay 参数
    'clip_grad': 10,                # 梯度剪裁参数
    # model EMA
    'ema_decay': 0.9999,            # 模型 EMA 参数
    'ema_tau': 2000,                # 模型 EMA 参数
    # lr schedule
    'scheduler': 'linear',          # 使用线性学习率衰减策略
    'lr0': 0.01,                    # 初始学习率
    'lrf': 0.01,                    # 最终的学习率 = lr0 * lrf
    'warmup_momentum': 0.8,         # Warmup 阶段，优化器的 momentum 参数的初始化
    'warmup_bias_lr': 0.1,          # Warmup 阶段，优化器为模型的 bias 参数设置的学习率初始值
}




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
