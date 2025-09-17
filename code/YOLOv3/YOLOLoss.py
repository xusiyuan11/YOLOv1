"""
YOLOv3损失函数实现
支持多尺度训练和锚框匹配
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from anchors import YOLOv3AnchorSystem


class YOLOv3Loss(nn.Module):
    """
    YOLOv3损失函数
    包含坐标损失、置信度损失和分类损失
    """
    def __init__(self, 
                 num_classes=20, 
                 input_size=416,
                 lambda_coord=5.0,      # 增加坐标损失权重
                 lambda_obj=1.0, 
                 lambda_noobj=0.1,      # 减少无目标损失权重
                 lambda_class=1.0,
                 ignore_threshold=0.5):
        super(YOLOv3Loss, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.ignore_threshold = ignore_threshold
        
        # 创建锚框系统
        self.anchor_system = YOLOv3AnchorSystem(input_size)
        self.grid_sizes = [13, 26, 52]
        
        # 损失函数
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        计算YOLOv3损失
        Args:
            predictions: 模型预测 [(batch, 75, 13, 13), (batch, 75, 26, 26), (batch, 75, 52, 52)]
            targets: 真实标签 (batch_size, max_objects, 5) [x1, y1, x2, y2, class_id]
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失详情
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        
        total_loss = 0
        loss_dict = {
            'coord_loss': 0,
            'obj_loss': 0, 
            'noobj_loss': 0,
            'class_loss': 0
        }
        
        # 为每个尺度计算损失
        for scale_idx, (pred, grid_size) in enumerate(zip(predictions, self.grid_sizes)):
            # 构建该尺度的目标
            scale_targets = self._build_targets(targets, scale_idx, grid_size, batch_size, device)
            
            # 计算该尺度的损失
            scale_losses = self._compute_scale_loss(pred, scale_targets, scale_idx, device)
            
            # 累加损失
            for key in loss_dict.keys():
                loss_dict[key] += scale_losses[key]
            
            total_loss += scale_losses['total']
        
        # 更新损失字典
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def _build_targets(self, targets, scale_idx, grid_size, batch_size, device):
        """
        为指定尺度构建训练目标
        Args:
            targets: (batch_size, max_objects, 5) [x1, y1, x2, y2, class_id]
            scale_idx: 尺度索引 (0, 1, 2)
            grid_size: 网格大小
            batch_size: 批次大小
            device: 设备
        Returns:
            scale_targets: 该尺度的目标张量
        """
        num_anchors = 3
        
        # 初始化目标张量 [batch, grid, grid, anchors, 5+num_classes]
        target_shape = (batch_size, grid_size, grid_size, num_anchors, 5 + self.num_classes)
        scale_targets = torch.zeros(target_shape, device=device)
        
        # 获取当前尺度的锚框
        anchors = self.anchor_system.anchors_config[scale_idx]
        stride = self.input_size / grid_size
        
        # 检查targets的类型和结构（移除调试输出以提高性能）
        
        # 处理不同的targets格式
        if isinstance(targets, list):
            # 如果targets是列表，检查是否为YOLOv1格式 [gt, mask_pos, mask_neg]
            if len(targets) == 0:
                return torch.zeros((batch_size, grid_size, grid_size, 3, 5 + self.num_classes), device=device)
            elif len(targets) == 3:
                # YOLOv1格式：使用第一个元素（gt）和第二个元素（mask_pos）
                gt_targets, pos_mask, neg_mask = targets
                targets = gt_targets  # 使用gt作为主要目标数据
            else:
                targets = torch.stack(targets) if len(targets[0].shape) > 0 else targets[0]
        
        for batch_idx in range(batch_size):
            if batch_idx >= targets.shape[0]:
                continue  # 跳过超出范围的批次
                
            batch_targets = targets[batch_idx]
            
            # 处理batch_targets的形状
            
            # 检查batch_targets的维度并处理
            if len(batch_targets.shape) == 3:
                # 如果是3维，需要展平为2维 (假设最后一维是特征维度)
                original_shape = batch_targets.shape
                batch_targets = batch_targets.view(-1, original_shape[-1])
            elif len(batch_targets.shape) == 4:
                # 如果是4维，需要展平为2维
                original_shape = batch_targets.shape
                batch_targets = batch_targets.view(-1, original_shape[-1])
            
            # 过滤有效目标 (class_id >= 0)
            if batch_targets.shape[1] < 5:
                continue
                
            valid_mask = batch_targets[:, 4] >= 0
            
            if not valid_mask.any():
                continue
                
            valid_targets = batch_targets[valid_mask]
            
            for target in valid_targets:
                # 检查target的维度，如果是25维，需要解析YOLOv1格式
                if target.shape[0] == 25:
                    # YOLOv1格式: [tx, ty, tw, th, conf, ...20个类别概率...]
                    tx, ty, tw, th, conf = target[:5].float()
                    class_probs = target[5:].float()
                    class_id = torch.argmax(class_probs).item()
                    
                    # YOLOv1格式已经是中心点坐标
                    x_center = tx
                    y_center = ty
                    width = tw
                    height = th
                else:
                    # 标准格式: [x1, y1, x2, y2, class_id]
                    target_float = target[:5].float()
                    x1, y1, x2, y2, class_id_float = target_float
                    class_id = int(class_id_float.item())
                    
                    # 转换为中心点坐标
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                
                # 转换为网格坐标
                grid_x = x_center * grid_size
                grid_y = y_center * grid_size
                grid_w = width * grid_size
                grid_h = height * grid_size
                
                # 找到对应的网格单元
                grid_i = int(grid_x)
                grid_j = int(grid_y)
                
                if 0 <= grid_i < grid_size and 0 <= grid_j < grid_size:
                    # 计算与所有锚框的IoU，选择最佳匹配
                    best_anchor_idx = self._find_best_anchor(grid_w * stride, grid_h * stride, anchors)
                    
                    # 设置目标值
                    scale_targets[batch_idx, grid_j, grid_i, best_anchor_idx, 0] = grid_x - grid_i  # dx
                    scale_targets[batch_idx, grid_j, grid_i, best_anchor_idx, 1] = grid_y - grid_j  # dy
                    scale_targets[batch_idx, grid_j, grid_i, best_anchor_idx, 2] = torch.log(grid_w / (anchors[best_anchor_idx][0] / stride) + 1e-6)  # dw
                    scale_targets[batch_idx, grid_j, grid_i, best_anchor_idx, 3] = torch.log(grid_h / (anchors[best_anchor_idx][1] / stride) + 1e-6)  # dh
                    scale_targets[batch_idx, grid_j, grid_i, best_anchor_idx, 4] = 1.0  # objectness
                    scale_targets[batch_idx, grid_j, grid_i, best_anchor_idx, 5 + class_id] = 1.0  # class
        
        return scale_targets
    
    def _find_best_anchor(self, gt_w, gt_h, anchors):
        """
        找到与真实框最匹配的锚框
        Args:
            gt_w, gt_h: 真实框的宽高 (像素)
            anchors: 锚框列表 [(w, h), ...]
        Returns:
            best_anchor_idx: 最佳锚框索引
        """
        ious = []
        for anchor_w, anchor_h in anchors:
            # 计算IoU (简化版本，基于宽高比较)
            iou = min(gt_w / anchor_w, anchor_w / gt_w) * min(gt_h / anchor_h, anchor_h / gt_h)
            ious.append(iou)
        
        return torch.argmax(torch.tensor(ious)).item()
    
    def _compute_scale_loss(self, predictions, targets, scale_idx, device):
        """
        计算单个尺度的损失
        Args:
            predictions: (batch, 75, grid, grid)
            targets: (batch, grid, grid, 3, 5+num_classes)
            scale_idx: 尺度索引
            device: 设备
        Returns:
            losses: 损失字典
        """
        batch_size, _, pred_grid_size, _ = predictions.shape
        target_grid_size = targets.shape[1]  # 从targets获取实际网格大小
        num_anchors = 3
        
        # 如果网格大小不匹配，调整targets的大小
        if pred_grid_size != target_grid_size:
            # 重新构建targets以匹配预测的网格大小
            # 简单的处理方法：重新初始化targets张量
            new_targets = torch.zeros(
                (batch_size, pred_grid_size, pred_grid_size, num_anchors, 5 + self.num_classes),
                device=device, dtype=targets.dtype
            )
            
            # 如果预测网格更大，进行上采样；如果更小，进行下采样
            scale_factor = pred_grid_size / target_grid_size
            
            # 简单的最近邻映射
            for b in range(batch_size):
                for i in range(target_grid_size):
                    for j in range(target_grid_size):
                        new_i = min(int(i * scale_factor), pred_grid_size - 1)
                        new_j = min(int(j * scale_factor), pred_grid_size - 1)
                        new_targets[b, new_i, new_j] = targets[b, i, j]
            
            targets = new_targets
        
        # 重塑预测张量 (batch, grid, grid, anchors, 5+num_classes)
        pred = predictions.view(batch_size, num_anchors, 5 + self.num_classes, pred_grid_size, pred_grid_size)
        pred = pred.permute(0, 3, 4, 1, 2).contiguous()
        
        # 分离预测结果
        pred_xy = torch.sigmoid(pred[..., :2])  # 中心点偏移
        pred_wh = pred[..., 2:4]  # 宽高缩放
        pred_conf = torch.sigmoid(pred[..., 4])  # 置信度
        pred_cls = pred[..., 5:]  # 类别logits
        
        # 分离目标
        target_xy = targets[..., :2]
        target_wh = targets[..., 2:4]
        target_conf = targets[..., 4]
        target_cls = targets[..., 5:]
        
        # 创建掩码
        obj_mask = target_conf > 0  # 有目标的位置
        noobj_mask = target_conf == 0  # 无目标的位置
        
        # 1. 坐标损失 (只对有目标的位置计算)
        coord_loss = 0
        if obj_mask.sum() > 0:
            coord_loss_xy = self.mse_loss(pred_xy[obj_mask], target_xy[obj_mask]).sum()
            coord_loss_wh = self.mse_loss(pred_wh[obj_mask], target_wh[obj_mask]).sum()
            coord_loss = self.lambda_coord * (coord_loss_xy + coord_loss_wh)
        
        # 2. 置信度损失
        obj_loss = 0
        if obj_mask.sum() > 0:
            obj_loss = self.lambda_obj * self.bce_loss(pred_conf[obj_mask], target_conf[obj_mask]).sum()
        
        noobj_loss = 0
        if noobj_mask.sum() > 0:
            noobj_loss = self.lambda_noobj * self.bce_loss(pred_conf[noobj_mask], target_conf[noobj_mask]).sum()
        
        # 3. 分类损失 (只对有目标的位置计算)
        class_loss = 0
        if obj_mask.sum() > 0:
            # 使用sigmoid + BCE，支持多标签分类
            pred_cls_sigmoid = torch.sigmoid(pred_cls[obj_mask])
            class_loss = self.lambda_class * self.bce_loss(pred_cls_sigmoid, target_cls[obj_mask]).sum()
        
        # 总损失
        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        
        # 归一化 (按有目标的数量)
        num_obj = obj_mask.sum().float() + 1e-6
        
        return {
            'coord_loss': coord_loss / num_obj,
            'obj_loss': obj_loss / num_obj,
            'noobj_loss': noobj_loss / noobj_mask.sum().float().clamp(min=1),
            'class_loss': class_loss / num_obj,
            'total': total_loss / num_obj
        }


# 兼容旧接口的包装器
class YOLOLoss(YOLOv3Loss):
    """向后兼容的YOLO损失函数"""
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, grid_size=7, num_boxes=2, num_classes=20):
        # 转换参数到YOLOv3格式
        super().__init__(
            num_classes=num_classes,
            lambda_coord=lambda_coord,
            lambda_noobj=lambda_noobj
        )
        print("警告: 使用YOLOv3损失函数，部分YOLOv1参数已忽略")


def create_yolov3_loss(num_classes=20, input_size=416, **kwargs):
    """创建YOLOv3损失函数"""
    return YOLOv3Loss(num_classes=num_classes, input_size=input_size, **kwargs)