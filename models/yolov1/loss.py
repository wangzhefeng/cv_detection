# -*- coding: utf-8 -*-

# ***************************************************
# * File        : loss.py
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
import torch.nn.functional as F

from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized
from matcher import YoloMatcher

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Criterion:
    """
    Criterion 类用于完成训练阶段的 <标签分配> 和 <损失计算>
    """

    def __init__(self, cfg, device, num_classes = 80) -> None:
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = cfg["loss_obj_weight"]
        self.loss_cls_weight = cfg["loss_cls_weight"]
        self.loss_box_weight = cfg["loss_box_weight"]
        # 标签分配(制作正样本)
        self.matcher = YoloMatcher(num_classes = num_classes)
    
    def loss_objectness(self, pred_obj, gt_obj):
        """
        计算 objectness 损失
        """
        loss_obj = F.binary_cross_entropy_with_logits(
            pred_obj, gt_obj, reduction = "none"
        )

        return loss_obj

    def loss_classes(self, pred_cls, gt_label):
        """
        计算 classification 损失
        """
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_cls, gt_label, reduction = "none"
        )

        return loss_cls

    def loss_bboxes(self, pred_box, gt_box):
        """
        计算 bbox regression 损失
        """
        ious = get_ious(
            pred_box, gt_box, 
            box_mode = "xyxy", iou_type = "giou"
        )
        loss_box = 1.0 - ious

        return loss_box

    def __call__(self, outputs, targets):
        device = outputs["pred_cls"][0].device
        stride = outputs["stride"]
        fmp_size = outputs["fmp_size"]
        pred_obj = outputs["pred_obj"].view(-1)                    # [B, M, C]->[BM,]
        pred_cls = outputs["pred_cls"].view(-1, self.num_classes)  # [B, M, C]->[BM, C]
        pred_box = outputs["pred_box"].view(-1, 4)                 # [B, M, C]->[BM, 4]
        # ------------------------------
        # 标签分配
        # ------------------------------
        # 标签
        gt_objectness, gt_classes, gt_bboxes = self.matcher(
            fmp_size = fmp_size,
            stride = stride,
            targets = targets,
        )
        # 将标签的 shape 处理成和预测的 shape 相同的形式，以便后续计算损失
        gt_objectness = gt_objectness.view(-1).to(device).float()
        gt_classes = gt_classes.view(-1, self.num_classes).to(device).float()
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()
        # 正样本标记
        pos_masks = (gt_objectness > 0)
        # 正样本数量
        num_fgs = pos_masks.sum()
        # 如果使用多卡做分布式训练，需要在多张卡上计算正样本数量的均值
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fgs)
        num_fgs = (num_fgs / get_world_size()).clamp(1.0)
        # ------------------------------
        # 计算损失
        # ------------------------------
        # 计算 objectness 损失，即边界框的置信度、或有无物体的置信度的损失
        loss_obj = self.loss_objectness(pred_obj, gt_objectness)
        loss_obj = loss_obj.sum() / num_fgs

        # 计算 classification 损失，即类别的置信度的损失
        pred_cls_pos = pred_cls[pos_masks]
        gt_classes_pos = gt_classes[pos_masks]
        loss_cls = self.loss_classes(pred_cls_pos, gt_classes_pos)
        loss_cls = loss_cls.sum() / num_fgs

        # 计算 box regression 损失，即边界框回归的损失
        pred_box_pos = pred_box[pos_masks]
        gt_bboxes_pos = gt_bboxes[pos_masks]
        loss_box = self.loss_bboxes(pred_box_pos, gt_bboxes_pos)
        loss_box = loss_box.sum() / num_fgs
        
        # 计算总损失
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box
        # ------------------------------
        # 输出
        # ------------------------------
        loss_dict = dict(
            loss_obj = loss_obj,
            loss_cls = loss_cls,
            loss_box = loss_box,
            losses = losses,
        )

        return loss_dict


def build_criterion(cfg, device, num_classes):
    criterion = Criterion(
        cfg = cfg,
        device = device,
        num_classes = num_classes,
    )

    return criterion




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
