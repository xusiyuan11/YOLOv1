import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    
    def __init__(self, lambda_coord=10.0, lambda_noobj=0.1, grid_size=7, num_boxes=2, num_classes=20):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord  # 坐标损失权重 (增加到10.0)
        self.lambda_noobj = lambda_noobj  # 无目标置信度损失权重 (减少到0.1)
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        print(f"YOLOLoss初始化: λ_coord={lambda_coord}, λ_noobj={lambda_noobj}")  # 调试信息
        
    def forward(self, predictions, targets):

        batch_size = predictions.size(0)
        
        # 处理targets格式：如果是列表，提取第一个元素作为ground truth
        if isinstance(targets, list):
            targets = targets[0]  # 使用gt，忽略mask_pos和mask_neg
        
        # 分离预测结果
        pred_boxes = predictions[:, :, :, :self.num_boxes*5].view(
            batch_size, self.grid_size, self.grid_size, self.num_boxes, 5
        )
        pred_classes = predictions[:, :, :, self.num_boxes*5:]
        
        # 分离目标
        target_boxes = targets[:, :, :, :self.num_boxes*5].view(
            batch_size, self.grid_size, self.grid_size, self.num_boxes, 5
        )
        target_classes = targets[:, :, :, self.num_boxes*5:]
        
        # 创建掩码 - 使用第一个边界框的置信度
        obj_mask = target_boxes[:, :, :, 0, 4] > 0  # 包含目标的网格
        noobj_mask = target_boxes[:, :, :, 0, 4] == 0  # 不包含目标的网格
        
        # 1. 坐标损失 (只对包含目标的网格计算)
        coord_loss = self._compute_coordinate_loss(pred_boxes, target_boxes, obj_mask)
        
        # 2. 置信度损失
        conf_loss_obj, conf_loss_noobj = self._compute_confidence_loss(
            pred_boxes, target_boxes, obj_mask, noobj_mask
        )
        
        # 3. 分类损失 (只对包含目标的网格计算)
        class_loss = self._compute_classification_loss(pred_classes, target_classes, obj_mask)
        
        # 总损失
        total_loss = (
            self.lambda_coord * coord_loss +
            conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            class_loss
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'conf_loss_obj': conf_loss_obj,
            'conf_loss_noobj': conf_loss_noobj,
            'class_loss': class_loss,
            # 添加加权后的损失组件用于调试
            'weighted_coord_loss': self.lambda_coord * coord_loss,
            'weighted_noobj_loss': self.lambda_noobj * conf_loss_noobj
        }
        
        return total_loss, loss_dict
    
    def _compute_coordinate_loss(self, pred_boxes, target_boxes, obj_mask):
        """计算坐标损失"""
        # 选择最佳预测框 (IoU最高的)
        best_box_indices = self._find_best_boxes(pred_boxes, target_boxes, obj_mask)

        coord_loss = 0.0

        for i in range(self.num_boxes):
            box_mask = (best_box_indices == i) & obj_mask

            if box_mask.sum() > 0:
                # 正确的索引方式：先选择box维度，再应用mask
                pred_xy = pred_boxes[..., i, :2][box_mask]
                pred_wh = pred_boxes[..., i, 2:4][box_mask]
                target_xy = target_boxes[..., 0, :2][box_mask]  # 使用第一个目标框
                target_wh = target_boxes[..., 0, 2:4][box_mask]  # 使用第一个目标框

                # 坐标损失 (x, y)
                xy_loss = F.mse_loss(pred_xy, target_xy, reduction='sum')

                # 宽高损失 (w, h) - 使用平方根
                wh_loss = F.mse_loss(
                    torch.sqrt(torch.clamp(pred_wh, min=0)),
                    torch.sqrt(target_wh),
                    reduction='sum'
                )

                coord_loss += xy_loss + wh_loss

        return coord_loss
    
    def _compute_confidence_loss(self, pred_boxes, target_boxes, obj_mask, noobj_mask):
        """计算置信度损失"""
        conf_loss_obj = 0.0
        conf_loss_noobj = 0.0
        
        # 有目标的置信度损失
        best_box_indices = self._find_best_boxes(pred_boxes, target_boxes, obj_mask)
        
        for i in range(self.num_boxes):
            box_mask = (best_box_indices == i) & obj_mask
            if box_mask.sum() > 0:
                pred_conf = pred_boxes[..., i, 4][box_mask]
                target_conf = target_boxes[..., 0, 4][box_mask]  # 使用第一个目标框的置信度
                conf_loss_obj += F.mse_loss(pred_conf, target_conf, reduction='sum')
        
        # 无目标的置信度损失
        if noobj_mask.sum() > 0:
            for i in range(self.num_boxes):
                pred_conf_noobj = pred_boxes[..., i, 4][noobj_mask]
                target_conf_noobj = torch.zeros_like(pred_conf_noobj)
                conf_loss_noobj += F.mse_loss(pred_conf_noobj, target_conf_noobj, reduction='sum')
        
        return conf_loss_obj, conf_loss_noobj
        
        return conf_loss_obj, conf_loss_noobj
    
    def _compute_classification_loss(self, pred_classes, target_classes, obj_mask):
        """计算分类损失"""
        if obj_mask.sum() > 0:
            pred_cls = pred_classes[obj_mask]
            target_cls = target_classes[obj_mask]
            class_loss = F.mse_loss(pred_cls, target_cls, reduction='sum')
        else:
            class_loss = torch.tensor(0.0, device=pred_classes.device)
        return class_loss
    
    def _find_best_boxes(self, pred_boxes, target_boxes, obj_mask):
        """为每个目标找到最佳预测框"""
        batch_size, grid_size, _, num_boxes, _ = pred_boxes.shape
        best_box_indices = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.long, device=pred_boxes.device)
        
        for b in range(batch_size):
            for i in range(grid_size):
                for j in range(grid_size):
                    if obj_mask[b, i, j]:
                        target_box = target_boxes[b, i, j, 0, :4]  # 使用第一个目标框作为参考
                        max_iou = -1
                        best_box = 0
                        
                        for box_idx in range(num_boxes):
                            pred_box = pred_boxes[b, i, j, box_idx, :4]
                            iou = self._calculate_iou(pred_box, target_box)
                            
                            if iou > max_iou:
                                max_iou = iou
                                best_box = box_idx
                        
                        best_box_indices[b, i, j] = best_box
        
        return best_box_indices
    
    def _calculate_iou(self, box1, box2):
        """计算两个边界框的 IoU"""
        # 转换为 (x1, y1, x2, y2) 格式
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2
        
        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2
        
        # 计算交集
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - inter_area
        
        # 计算 IoU
        iou = inter_area / (union_area + 1e-6)
        return iou



