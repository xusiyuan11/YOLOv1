"""
YOLOv3后处理模块
包含预测解码、NMS和多尺度结果融合
"""
import torch
import torch.nn.functional as F
import numpy as np
from anchors import YOLOv3AnchorSystem


class YOLOv3PostProcessor:
    """
    YOLOv3后处理器
    负责将模型输出转换为最终的检测结果
    """
    def __init__(self, 
                 num_classes=20,
                 input_size=416,
                 conf_threshold=0.5,
                 nms_threshold=0.4,
                 max_detections=100):
        """
        Args:
            num_classes: 类别数量
            input_size: 输入图像尺寸
            conf_threshold: 置信度阈值
            nms_threshold: NMS IoU阈值
            max_detections: 最大检测数量
        """
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # 创建锚框系统
        self.anchor_system = YOLOv3AnchorSystem(input_size)
        self.grid_sizes = [13, 26, 52]
    
    def __call__(self, predictions, input_shape=None):
        """
        处理模型预测
        Args:
            predictions: 模型预测列表 [(B,75,13,13), (B,75,26,26), (B,75,52,52)]
            input_shape: 原始输入图像尺寸 (H, W)，用于坐标还原
        Returns:
            detections: 检测结果列表，每个batch一个 [(N,6), ...] [x1,y1,x2,y2,conf,class]
        """
        return self.process_predictions(predictions, input_shape)
    
    def process_predictions(self, predictions, input_shape=None):
        """
        处理预测结果
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        
        # 解码所有尺度的预测
        all_detections = []
        
        for batch_idx in range(batch_size):
            batch_predictions = [pred[batch_idx:batch_idx+1] for pred in predictions]
            
            # 解码当前batch的预测
            decoded_results = self._decode_predictions(batch_predictions, device)
            
            # 应用NMS
            final_detections = self._apply_nms(decoded_results)
            
            # 坐标还原
            if input_shape is not None:
                final_detections = self._rescale_boxes(final_detections, input_shape)
            
            all_detections.append(final_detections)
        
        return all_detections
    
    def _decode_predictions(self, predictions, device):
        """
        解码预测结果
        Args:
            predictions: 单个batch的预测 [(1,75,13,13), (1,75,26,26), (1,75,52,52)]
            device: 设备
        Returns:
            decoded_boxes: 解码后的边界框 (N, 6) [x1,y1,x2,y2,conf,class]
        """
        all_boxes = []
        
        for scale_idx, (pred, grid_size) in enumerate(zip(predictions, self.grid_sizes)):
            # 解码当前尺度
            scale_boxes = self._decode_scale(pred, scale_idx, grid_size, device)
            if len(scale_boxes) > 0:
                all_boxes.append(scale_boxes)
        
        if len(all_boxes) == 0:
            return torch.zeros(0, 6, device=device)
        
        # 合并所有尺度的结果
        all_boxes = torch.cat(all_boxes, dim=0)
        return all_boxes
    
    def _decode_scale(self, predictions, scale_idx, grid_size, device):
        """
        解码单个尺度的预测
        Args:
            predictions: (1, 75, grid, grid)
            scale_idx: 尺度索引
            grid_size: 网格大小
            device: 设备
        Returns:
            boxes: (N, 6) [x1,y1,x2,y2,conf,class]
        """
        batch_size = predictions.shape[0]
        num_anchors = 3
        
        # 重塑预测张量 (1, grid, grid, 3, 25)
        pred = predictions.view(batch_size, num_anchors, 5 + self.num_classes, grid_size, grid_size)
        pred = pred.permute(0, 3, 4, 1, 2).contiguous()
        pred = pred.view(grid_size, grid_size, num_anchors, 5 + self.num_classes)
        
        # 获取锚框
        anchors = self.anchor_system.anchors_config[scale_idx]
        stride = self.input_size / grid_size
        
        # 创建网格坐标
        grid_x = torch.arange(grid_size, dtype=torch.float32, device=device)
        grid_y = torch.arange(grid_size, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        
        boxes = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                for a in range(num_anchors):
                    # 提取预测值
                    prediction = pred[i, j, a]
                    
                    # 解码坐标
                    x = (torch.sigmoid(prediction[0]) + j) * stride
                    y = (torch.sigmoid(prediction[1]) + i) * stride
                    w = torch.exp(prediction[2]) * anchors[a][0]
                    h = torch.exp(prediction[3]) * anchors[a][1]
                    
                    # 置信度
                    obj_conf = torch.sigmoid(prediction[4])
                    
                    # 类别概率
                    class_probs = torch.sigmoid(prediction[5:])
                    class_conf, class_pred = torch.max(class_probs, dim=0)
                    
                    # 最终置信度
                    final_conf = obj_conf * class_conf
                    
                    # 过滤低置信度
                    if final_conf > self.conf_threshold:
                        # 转换为边界框格式 [x1, y1, x2, y2]
                        x1 = x - w / 2
                        y1 = y - h / 2
                        x2 = x + w / 2
                        y2 = y + h / 2
                        
                        # 限制在图像范围内
                        x1 = torch.clamp(x1, 0, self.input_size)
                        y1 = torch.clamp(y1, 0, self.input_size)
                        x2 = torch.clamp(x2, 0, self.input_size)
                        y2 = torch.clamp(y2, 0, self.input_size)
                        
                        # 添加到结果
                        box = torch.tensor([x1, y1, x2, y2, final_conf, class_pred], 
                                         dtype=torch.float32, device=device)
                        boxes.append(box)
        
        if len(boxes) == 0:
            return torch.zeros(0, 6, device=device)
        
        return torch.stack(boxes)
    
    def _apply_nms(self, boxes):
        """
        应用非极大值抑制
        Args:
            boxes: (N, 6) [x1,y1,x2,y2,conf,class]
        Returns:
            filtered_boxes: NMS后的边界框
        """
        if len(boxes) == 0:
            return boxes
        
        # 按类别分组进行NMS
        unique_classes = torch.unique(boxes[:, 5])
        final_boxes = []
        
        for cls in unique_classes:
            # 获取当前类别的框
            class_mask = boxes[:, 5] == cls
            class_boxes = boxes[class_mask]
            
            # 按置信度排序
            _, order = torch.sort(class_boxes[:, 4], descending=True)
            class_boxes = class_boxes[order]
            
            # 应用NMS
            keep_indices = self._nms(class_boxes[:, :4], class_boxes[:, 4], self.nms_threshold)
            
            if len(keep_indices) > 0:
                keep_indices = torch.tensor(keep_indices, dtype=torch.long)
                final_boxes.append(class_boxes[keep_indices])
        
        if len(final_boxes) == 0:
            return torch.zeros(0, 6, device=boxes.device)
        
        final_boxes = torch.cat(final_boxes, dim=0)
        
        # 限制最大检测数量
        if len(final_boxes) > self.max_detections:
            _, indices = torch.sort(final_boxes[:, 4], descending=True)
            final_boxes = final_boxes[indices[:self.max_detections]]
        
        return final_boxes
    
    def _nms(self, boxes, scores, threshold):
        """
        非极大值抑制实现
        Args:
            boxes: (N, 4) [x1,y1,x2,y2]
            scores: (N,) 置信度分数
            threshold: IoU阈值
        Returns:
            keep: 保留的索引
        """
        if len(boxes) == 0:
            return []
        
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按分数排序
        _, order = torch.sort(scores, descending=True)
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # 计算IoU
            xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
            yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
            xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
            yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            # 保留IoU小于阈值的框
            mask = iou <= threshold
            order = order[1:][mask]
        
        return keep
    
    def _rescale_boxes(self, boxes, input_shape):
        """
        将边界框坐标还原到原始图像尺寸
        Args:
            boxes: (N, 6) [x1,y1,x2,y2,conf,class]
            input_shape: (H, W) 原始图像尺寸
        Returns:
            rescaled_boxes: 还原后的边界框
        """
        if len(boxes) == 0:
            return boxes
        
        orig_h, orig_w = input_shape
        
        # 计算缩放比例
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size
        
        # 还原坐标
        boxes[:, 0] *= scale_x  # x1
        boxes[:, 1] *= scale_y  # y1
        boxes[:, 2] *= scale_x  # x2
        boxes[:, 3] *= scale_y  # y2
        
        return boxes


def create_postprocessor(num_classes=20, input_size=416, **kwargs):
    """创建YOLOv3后处理器"""
    return YOLOv3PostProcessor(num_classes=num_classes, input_size=input_size, **kwargs)


# 便捷函数
def postprocess_yolov3(predictions, 
                      num_classes=20,
                      input_size=416,
                      conf_threshold=0.5,
                      nms_threshold=0.4,
                      input_shape=None):
    """
    便捷的后处理函数
    Args:
        predictions: 模型预测
        num_classes: 类别数量
        input_size: 输入尺寸
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
        input_shape: 原始图像尺寸
    Returns:
        检测结果
    """
    processor = YOLOv3PostProcessor(
        num_classes=num_classes,
        input_size=input_size,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )
    
    return processor(predictions, input_shape)
