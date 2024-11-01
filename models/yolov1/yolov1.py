# -*- coding: utf-8 -*-

# ***************************************************
# * File        : YOLOv1.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 1.0.103100
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
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from typing import List

import numpy as np
import torch
import torch.nn as nn

from yolov1_backbone import build_backbone
from yolov1_neck import build_neck
from yolov1_head import build_head
from yolov1_predict import build_pred

from utils.misc import multiclass_nms

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class YOLOv1(nn.Module):
    
    def __init__(self, 
                 cfg, 
                 device, 
                 img_size = None,
                 num_classes = 20, 
                 conf_thresh = 0.01,
                 nms_thresh = 0.5,
                 trainable = False,
                 deploy = False):
        super(YOLOv1, self).__init__()
        # ------------------------------
        # 基础参数
        # ------------------------------
        self.cfg = cfg                   # 模型配置文件
        self.img_size = img_size         # 输入图像大小
        self.device = device             # 设备：cuda 或 cpu
        self.num_classes = num_classes   # 类别的数量
        self.conf_thresh = conf_thresh   # 得分阈值
        self.nms_thresh = nms_thresh     # NMS 阈值
        self.trainable = trainable       # 训练的标记
        self.stride = 32                 # 网络的最大步长
        self.deploy = deploy
        # ------------------------------
        # 主干网络(backbone network)
        # ------------------------------
        self.backbone, feat_dim = build_backbone(cfg["backbone"], trainable & cfg["pretrained"])
        # ------------------------------
        # 颈部网络(neck network)
        # ------------------------------
        self.neck = build_neck(cfg, feat_dim, out_dim = 512)
        head_dim = self.neck.out_dim
        # ------------------------------
        # 检测头(detection head)
        # ------------------------------
        self.head = build_head(cfg, head_dim, head_dim, num_classes)
        # ------------------------------
        # 预测层(prediction layer)
        # ------------------------------
        # # 置信度预测
        # self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size = 1)
        # # 类别预测
        # self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size = 1)
        # # 位置参数预测
        # self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size = 1)
        self.obj_pred, self.cls_pred, self.reg_pred = build_pred(head_dim, num_classes)
    
    def create_grid(self, fmp_size):
        """
        用于生成网格坐标矩阵，其中每个元素都是特征图上的像素坐标
        """ 
        # 特征图的宽和高
        hs, ws = fmp_size
        # 生成网格的 x 坐标和 y 坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # 将 x, y 两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim = -1).float()
        # [H, W, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        
        return grid_xy
    
    def decode_boxes(self, pred_reg: torch.Tensor, fmp_size: List[int, int]):
        """
        解算边界框坐标：将 YOLO 预测的 (tx, ty)、(tw, th) 转换为 bbox 的左上角坐标 (x1, y1) 和右下角坐标 (x2, y2)

        输入:
            pred_reg: (torch.Tensor) -> [B, HxW, 4] or [HxW, 4]，网络预测的 tx,ty,tw,th
            fmp_size: (List[int, int])，包含输出特征图的宽度和高度两个参数
        输出:
            pred_box: (torch.Tensor) -> [B, HxW, 4] or [HxW, 4]，解算出的边界框坐标
        """
        # 生成网格坐标矩阵
        grid_cell = self.create_grid(fmp_size)
        # 计算预测边界框的中心坐标(tx, ty)和宽高(w, h)
        # cx = (gridx + sigmoid(tx))*stride, cy = (gridy + sigmoid(ty))*stride
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) + self.stride
        # w = exp(tw)*stride, h = exp(th)*stride
        pred_wh = torch.exp(pred_reg[..., 2:]) * self.stride
        # 将所有 bbox 的中心点坐标和宽高换算成 x1 y1 x2 y2 形式
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim = -1)

        return pred_box

    def postprocess(self, bboxes, scores):
        """
        后处理环节，包括 <阈值筛选> 和 <非极大值抑制(NMS)> 两个环节
        
        输入:
            bboxes: (numpy.array) -> [HxW, 4]
            scores: (numpy.array) -> [HxW, num_classes]
        输出:
            bboxes: (numpy.array) -> [N, 4]
            score:  (numpy.array) -> [N,]
            labels: (numpy.array) -> [N,]
        """
        # 将得分较高的类别作为预测的类别标签
        labels = np.argmax(scores, axis = 1)
        # 预测标签所对应的得分
        scores = scores[(np.arange(scores.shape[0]), labels)]
        # 阈值筛选
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        # 非极大值抑制(nms)
        scores, labels, bboxes = multiclass_nms(
            scores,
            labels,
            bboxes,
            self.nms_thresh,
            self.num_classes,
            False
        )

        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        """
        YOLOv1 前向推理
        """
        # 主干网络
        feat = self.backbone(x)
        # 颈部网络
        feat = self.neck(feat)
        # 检测头
        cls_feat, reg_feat = self.head(feat)
        # 预测层
        obj_pred = self.obj_pred(cls_feat)
        cls_pred = self.cls_pred(cls_pred)
        reg_pred = self.reg_pred(reg_pred)
        fmp_size = obj_pred.shape[-2:]  # [H, W]
        # 对 pred 的 size 做一些 view 调整，便于后续的处理
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flattne(1, 2)
        # 测试时，默认 batch=1，因此不需要 batch 这个维度，用 [0] 将其去除
        obj_pred = obj_pred[0]  # [H*W, 1]
        cls_pred = cls_pred[0]  # [H*W, NC]
        reg_pred = reg_pred[0]  # [H*W, 4]
        
        # 每个边界框的得分
        scores = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())
        
        # 解算边界框，并归一化边界框: [H*W, 4]
        bboxes = self.decode_boxes(reg_pred, fmp_size)
        
        # 部署
        if self.deploy:
            # [n_anchors_all, 4+C]
            outputs = torch.cat([bboxes, scores], dim = -1)
            return outputs
        else:
            # 将 bbox 和 score 预测都放在 CPU 处理上，以便进行后处理
            scores = scores.cpu().numpy()
            bboxes = bboxes.cpu().numpy()
            # 后处理
            bboxes, scores, labels = self.postprocess(bboxes, scores)
        
        return bboxes, scores, labels

    def forward(self, x):
        """
        YOLOv1 的主体运算函数
        """
        if not self.trainable:
            return self.inference(x)
        else:
            # 主干网络
            feat = self.backbone(x)  # [B, 416, 416, 3]->[B, 13, 13, 512]
            # 颈部网络
            feat = self.neck(feat)  # [B, 13, 13, 512]->[B, 13, 13, 512]
            # 检测头
            cls_feat, reg_feat = self.head(feat)  # [B, 13, 13, 512] -> [B, 13, 13, 512]
            # 预测层
            obj_pred = self.obj_pred(cls_feat)  # [B, 13, 13, 1]
            cls_pred = self.cls_pred(cls_feat)  # [B, 13, 13, 20]
            reg_pred = self.reg_pred(reg_feat)  # [B, 13, 13, 4]
            fmp_size = obj_pred.shape[-2:]
            # 对 pred 的 size 做一些 view 调整，便于后续的处理
            # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flattne(1, 2)
            
            # 解算边界框坐标
            box_pred = self.decode_boxes(reg_pred, fmp_size)
            
            # 网络输出
            outputs = {
                "pred_obj": obj_pred,  # torch.Tensor [B, M, 1]
                "pred_cls": cls_pred,  # torch.Tensor [B, M, C]
                "pred_box": box_pred,  # torch.Tensor [B, M, 4]
                "stride": self.stride, # Int
                "fmp_size": fmp_size,
            }
            return outputs




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
