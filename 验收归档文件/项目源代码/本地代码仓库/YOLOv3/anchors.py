"""
YOLOv3锚框机制实现
包含锚框生成、匹配策略和坐标编码/解码
"""
import torch
import torch.nn as nn
import numpy as np


class AnchorGenerator:
    """
    YOLOv3锚框生成器
    在每个网格位置生成预定义的锚框
    """
    def __init__(self, anchors, input_size=416):
        """
        Args:
            anchors: 锚框尺寸列表 [(w1, h1), (w2, h2), (w3, h3)]
            input_size: 输入图像尺寸
        """
        self.anchors = anchors
        self.input_size = input_size
        self.num_anchors = len(anchors)
    
    def generate_anchors(self, grid_size, device='cpu'):
        """
        在指定网格尺寸上生成所有锚框
        Args:
            grid_size: 网格尺寸 (如13, 26, 52)
            device: 设备
        Returns:
            anchor_grid: (grid_size, grid_size, num_anchors, 4) [x, y, w, h]
        """
        # 计算步长
        stride = self.input_size / grid_size
        
        # 将锚框尺寸缩放到当前网格
        scaled_anchors = [(w / stride, h / stride) for w, h in self.anchors]
        
        # 创建网格坐标
        grid_x = torch.arange(grid_size, dtype=torch.float32, device=device)
        grid_y = torch.arange(grid_size, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        # 扩展维度以包含锚框数量
        grid_x = grid_x.unsqueeze(2).repeat(1, 1, self.num_anchors)  # (grid_size, grid_size, num_anchors)
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.num_anchors)
        
        # 创建锚框宽高
        anchor_w = torch.zeros_like(grid_x)
        anchor_h = torch.zeros_like(grid_y)
        
        for i, (w, h) in enumerate(scaled_anchors):
            anchor_w[:, :, i] = w
            anchor_h[:, :, i] = h
        
        # 组合成锚框 [x, y, w, h]
        anchor_grid = torch.stack([grid_x, grid_y, anchor_w, anchor_h], dim=-1)
        
        return anchor_grid
    
    def generate_all_anchors(self, grid_sizes, device='cpu'):
        """
        为所有尺度生成锚框
        Args:
            grid_sizes: 网格尺寸列表 [13, 26, 52]
            device: 设备
        Returns:
            所有尺度的锚框列表
        """
        all_anchors = []
        for grid_size in grid_sizes:
            anchors = self.generate_anchors(grid_size, device)
            all_anchors.append(anchors)
        return all_anchors


class AnchorMatcher:
    """
    锚框匹配器
    负责在训练时将真实框与锚框进行匹配
    """
    def __init__(self, anchors_config, iou_threshold=0.5, ignore_threshold=0.4):
        """
        Args:
            anchors_config: 所有尺度的锚框配置
            iou_threshold: 正样本IoU阈值
            ignore_threshold: 忽略样本IoU阈值
        """
        self.anchors_config = anchors_config
        self.iou_threshold = iou_threshold
        self.ignore_threshold = ignore_threshold
        
        # 为每个尺度创建锚框生成器
        self.anchor_generators = []
        for anchors in anchors_config:
            self.anchor_generators.append(AnchorGenerator(anchors))
    
    def compute_iou(self, box1, box2):
        """
        计算两个框的IoU
        Args:
            box1: (N, 4) [x1, y1, x2, y2]
            box2: (M, 4) [x1, y1, x2, y2]
        Returns:
            iou: (N, M)
        """
        # 扩展维度进行广播
        box1 = box1.unsqueeze(1)  # (N, 1, 4)
        box2 = box2.unsqueeze(0)  # (1, M, 4)
        
        # 计算交集
        inter_x1 = torch.max(box1[..., 0], box2[..., 0])
        inter_y1 = torch.max(box1[..., 1], box2[..., 1])
        inter_x2 = torch.min(box1[..., 2], box2[..., 2])
        inter_y2 = torch.min(box1[..., 3], box2[..., 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def match_anchors(self, gt_boxes, gt_labels, grid_sizes, input_size=416):
        """
        匹配锚框与真实框
        Args:
            gt_boxes: 真实框 (N, 4) [x1, y1, x2, y2] 归一化坐标
            gt_labels: 真实标签 (N,)
            grid_sizes: 网格尺寸 [13, 26, 52]
            input_size: 输入尺寸
        Returns:
            匹配结果列表，每个尺度一个
        """
        device = gt_boxes.device
        matches = []
        
        for scale_idx, grid_size in enumerate(grid_sizes):
            # 生成当前尺度的锚框
            anchor_generator = self.anchor_generators[scale_idx]
            anchor_grid = anchor_generator.generate_anchors(grid_size, device)
            
            # 初始化匹配结果
            batch_size = 1  # 假设批次大小为1，实际使用时需要处理批次
            num_anchors = len(self.anchors_config[scale_idx])
            
            # 目标张量 [batch, grid, grid, anchors, 5+num_classes]
            # 5 = [x, y, w, h, objectness]
            target_shape = (batch_size, grid_size, grid_size, num_anchors, 5 + len(torch.unique(gt_labels)))
            targets = torch.zeros(target_shape, device=device)
            
            # 对每个真实框找最佳匹配锚框
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                # 将归一化坐标转换为网格坐标
                gt_x_center = gt_box[0] * grid_size
                gt_y_center = gt_box[1] * grid_size
                gt_w = (gt_box[2] - gt_box[0]) * grid_size
                gt_h = (gt_box[3] - gt_box[1]) * grid_size
                
                # 找到对应的网格单元
                grid_x = int(gt_x_center)
                grid_y = int(gt_y_center)
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    # 计算与当前尺度所有锚框的IoU
                    anchor_ious = []
                    for anchor_idx, (anchor_w, anchor_h) in enumerate(self.anchors_config[scale_idx]):
                        # 计算锚框与真实框的IoU（使用宽高比较）
                        anchor_w_scaled = anchor_w / (input_size / grid_size)
                        anchor_h_scaled = anchor_h / (input_size / grid_size)
                        
                        # 简化的IoU计算（基于宽高）
                        iou = min(gt_w / anchor_w_scaled, anchor_w_scaled / gt_w) * \
                              min(gt_h / anchor_h_scaled, anchor_h_scaled / gt_h)
                        anchor_ious.append(iou)
                    
                    # 选择IoU最高的锚框
                    best_anchor_idx = torch.argmax(torch.tensor(anchor_ious))
                    
                    if anchor_ious[best_anchor_idx] > self.ignore_threshold:
                        # 设置目标值
                        targets[0, grid_y, grid_x, best_anchor_idx, 0] = gt_x_center - grid_x  # dx
                        targets[0, grid_y, grid_x, best_anchor_idx, 1] = gt_y_center - grid_y  # dy
                        targets[0, grid_y, grid_x, best_anchor_idx, 2] = torch.log(gt_w / anchor_w_scaled + 1e-6)  # dw
                        targets[0, grid_y, grid_x, best_anchor_idx, 3] = torch.log(gt_h / anchor_h_scaled + 1e-6)  # dh
                        targets[0, grid_y, grid_x, best_anchor_idx, 4] = 1.0  # objectness
                        targets[0, grid_y, grid_x, best_anchor_idx, 5 + gt_label] = 1.0  # class
            
            matches.append(targets)
        
        return matches


class YOLOv3AnchorSystem:
    """
    YOLOv3完整锚框系统
    整合锚框生成和匹配功能
    """
    def __init__(self, input_size=416):
        self.input_size = input_size
        
        # YOLOv3标准锚框配置
        self.anchors_config = [
            [(116, 90), (156, 198), (373, 326)],  # 13x13 大目标
            [(30, 61), (62, 45), (59, 119)],      # 26x26 中等目标
            [(10, 13), (16, 30), (33, 23)]        # 52x52 小目标
        ]
        
        self.grid_sizes = [13, 26, 52]
        
        # 创建锚框生成器
        self.anchor_generators = []
        for anchors in self.anchors_config:
            self.anchor_generators.append(AnchorGenerator(anchors, input_size))
        
        # 创建锚框匹配器
        self.anchor_matcher = AnchorMatcher(self.anchors_config)
    
    def generate_all_anchors(self, device='cpu'):
        """生成所有尺度的锚框"""
        all_anchors = []
        for i, grid_size in enumerate(self.grid_sizes):
            anchors = self.anchor_generators[i].generate_anchors(grid_size, device)
            all_anchors.append(anchors)
        return all_anchors
    
    def match_targets(self, gt_boxes, gt_labels):
        """匹配目标"""
        return self.anchor_matcher.match_anchors(gt_boxes, gt_labels, self.grid_sizes, self.input_size)
    
    def decode_predictions(self, predictions, scale_idx, device='cpu'):
        """
        解码预测结果
        Args:
            predictions: 模型预测 (batch, channels, grid, grid)
            scale_idx: 尺度索引 (0, 1, 2)
            device: 设备
        Returns:
            解码后的边界框
        """
        batch_size, _, grid_size, _ = predictions.shape
        num_anchors = len(self.anchors_config[scale_idx])
        num_classes = predictions.shape[1] // num_anchors - 5
        
        # 重塑预测张量
        predictions = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_size, grid_size)
        predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()  # (batch, grid, grid, anchors, 5+classes)
        
        # 生成锚框
        anchor_grid = self.anchor_generators[scale_idx].generate_anchors(grid_size, device)
        
        # 解码坐标
        pred_xy = torch.sigmoid(predictions[..., :2])  # 中心点偏移
        pred_wh = predictions[..., 2:4]  # 宽高缩放
        pred_conf = torch.sigmoid(predictions[..., 4:5])  # 置信度
        pred_cls = torch.sigmoid(predictions[..., 5:])  # 类别概率
        
        # 计算实际坐标
        stride = self.input_size / grid_size
        pred_xy = (pred_xy + anchor_grid[..., :2]) * stride  # 转换为像素坐标
        pred_wh = torch.exp(pred_wh) * anchor_grid[..., 2:4] * stride  # 转换为像素尺寸
        
        # 组合结果
        decoded = torch.cat([pred_xy, pred_wh, pred_conf, pred_cls], dim=-1)
        
        return decoded


# 创建默认锚框系统
def create_anchor_system(input_size=416):
    """创建YOLOv3锚框系统"""
    return YOLOv3AnchorSystem(input_size)
