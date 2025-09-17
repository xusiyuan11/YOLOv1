"""
SwinYOLO: 基于Swin Transformer的简化YOLO检测器
直接在Swin后面接检测头，架构更清晰
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import create_backbone


class SwinYOLODetector(nn.Module):
    """
    基于Swin Transformer的YOLO检测器
    简化架构：Swin Backbone + 检测头
    """
    
    def __init__(self, 
                 num_classes=20, 
                 input_size=448, 
                 grid_size=7,
                 num_boxes=2):
        super(SwinYOLODetector, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        
        # Swin Transformer backbone
        self.backbone = create_backbone('swin', input_size=input_size)
        
        # 检测头：将Swin的512维特征转换为YOLO输出
        # 输出通道数: num_boxes * 5 + num_classes
        # 每个box: (x, y, w, h, confidence) = 5个值
        output_channels = num_boxes * 5 + num_classes
        
        self.detection_head = nn.Sequential(
            # 第一层：特征增强
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 第二层：降维
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 第三层：输出层
            nn.Conv2d(256, output_channels, kernel_size=1),
        )
        
        # 🔧 改进检测头初始化，特别是分类层的偏置
        self._init_detection_head()
        
        # 如果Swin输出14x14，需要调整到目标grid_size
        self.need_resize = True  # Swin输出14x14，通常需要调整
    
    def _init_detection_head(self):
        """初始化检测头，特别是分类层的偏置"""
        # 获取最后一层（输出层）
        output_layer = self.detection_head[-1]
        
        # 初始化置信度偏置为负值，让模型开始时更谨慎
        # 坐标和尺寸的偏置保持为0
        with torch.no_grad():
            # 置信度偏置设为-2，对应sigmoid后约0.12的初始置信度
            output_layer.bias[self.num_boxes*4:self.num_boxes*5].fill_(-2.0)
            
            # 分类层偏置设为小的随机值，避免完全偏向某些类别
            class_start_idx = self.num_boxes * 5
            output_layer.bias[class_start_idx:].normal_(0, 0.01)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, 448, 448]
        Returns:
            output: YOLO输出 [B, grid_size, grid_size, num_boxes*5 + num_classes]
        """
        # Swin backbone提取特征
        features = self.backbone(x)  # [B, 512, 14, 14]
        
        # 调整特征图尺寸到目标grid_size
        if self.need_resize and features.shape[-1] != self.grid_size:
            features = F.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
        
        # 检测头处理
        detection_output = self.detection_head(features)  # [B, output_channels, grid_size, grid_size]
        
        # 转换维度顺序: [B, C, H, W] -> [B, H, W, C]
        detection_output = detection_output.permute(0, 2, 3, 1)  # [B, grid_size, grid_size, output_channels]
        
        return detection_output
    
    def freeze_backbone(self):
        """冻结backbone，只训练检测头"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✅ Swin backbone已冻结，只训练检测头")
    
    def unfreeze_backbone(self):
        """解冻backbone，进行端到端微调"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✅ Swin backbone已解冻，进行端到端训练")
    
    def get_parameter_groups(self, base_lr=0.001, backbone_lr_ratio=0.1):
        """
        获取不同学习率的参数组
        backbone使用较小的学习率，检测头使用较大的学习率
        """
        backbone_params = []
        detection_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    detection_params.append(param)
        
        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * backbone_lr_ratio,
                'name': 'swin_backbone'
            })
        if detection_params:
            param_groups.append({
                'params': detection_params,
                'lr': base_lr,
                'name': 'detection_head'
            })
        
        return param_groups
    
    def load_pretrained_backbone(self, pretrained_path):
        """
        加载预训练的分类模型中的backbone权重
        """
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # 尝试不同的键名
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'backbone_state_dict' in checkpoint:
                state_dict = checkpoint['backbone_state_dict']
            else:
                state_dict = checkpoint
            
            # 提取backbone相关权重
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if 'backbone' in key:
                    # 去掉前缀
                    new_key = key.replace('model.backbone.', '').replace('backbone.', '')
                    backbone_state_dict[new_key] = value
            
            # 加载权重
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
            print(f"✅ 成功加载预训练backbone")
            print(f"   加载层数: {len(backbone_state_dict) - len(missing_keys)}")
            if missing_keys:
                print(f"   未匹配层数: {len(missing_keys)}")
                
        except Exception as e:
            print(f"❌ 加载预训练backbone失败: {e}")
            print("   将使用随机初始化继续训练")


class SwinYOLOLoss(nn.Module):
    """
    简化的YOLO损失函数，适配SwinYOLO
    """
    
    def __init__(self, 
                 num_classes=20, 
                 num_boxes=2,
                 lambda_coord=10.0,  # 增加坐标损失权重 (应用YOLOv1/v3经验)
                 lambda_noobj=0.1):  # 减少无目标损失权重 (应用优化经验)
        super(SwinYOLOLoss, self).__init__()
        
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, predictions, targets):
        """
        计算YOLO损失
        Args:
            predictions: 模型预测 [B, grid_size, grid_size, num_boxes*5 + num_classes]
            targets: 真实标签 [B, grid_size, grid_size, num_boxes*5 + num_classes]
        """
        batch_size, grid_size, _, _ = predictions.shape
        
        # 分离预测的不同部分
        # 边界框: [B, grid_size, grid_size, num_boxes*4]
        # 置信度: [B, grid_size, grid_size, num_boxes]  
        # 类别: [B, grid_size, grid_size, num_classes]
        
        pred_boxes = predictions[..., :self.num_boxes*4].view(batch_size, grid_size, grid_size, self.num_boxes, 4)
        pred_conf = predictions[..., self.num_boxes*4:self.num_boxes*5].view(batch_size, grid_size, grid_size, self.num_boxes)
        pred_classes = predictions[..., self.num_boxes*5:]
        
        target_boxes = targets[..., :self.num_boxes*4].view(batch_size, grid_size, grid_size, self.num_boxes, 4)
        target_conf = targets[..., self.num_boxes*4:self.num_boxes*5].view(batch_size, grid_size, grid_size, self.num_boxes)
        target_classes = targets[..., self.num_boxes*5:]
        
        # 计算各项损失
        coord_loss = self._coordinate_loss(pred_boxes, target_boxes, target_conf)
        conf_loss = self._confidence_loss(pred_conf, target_conf)
        class_loss = self._classification_loss(pred_classes, target_classes, target_conf)
        
        # 🔧 平衡损失权重，解决类别预测偏向问题
        lambda_conf = 1.0   # 降低置信度损失权重
        lambda_class = 2.0  # 增加分类损失权重，鼓励学习更多类别
        total_loss = (self.lambda_coord * coord_loss + 
                     lambda_conf * conf_loss + 
                     lambda_class * class_loss)
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'conf_loss': conf_loss, 
            'class_loss': class_loss
        }
    
    def _coordinate_loss(self, pred_boxes, target_boxes, target_conf):
        """坐标损失"""
        # 只计算有目标的网格的坐标损失
        mask = target_conf > 0  # [B, grid_size, grid_size, num_boxes]
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_boxes.device)
        
        # 坐标损失 (x, y, w, h)
        coord_loss = F.mse_loss(
            pred_boxes[mask], 
            target_boxes[mask], 
            reduction='sum'
        ) / mask.sum()
        
        return coord_loss
    
    def _confidence_loss(self, pred_conf, target_conf):
        """置信度损失"""
        # 有目标的置信度损失
        obj_mask = target_conf > 0
        obj_loss = F.mse_loss(pred_conf[obj_mask], target_conf[obj_mask], reduction='sum') if obj_mask.sum() > 0 else 0
        
        # 无目标的置信度损失
        noobj_mask = target_conf == 0
        noobj_loss = F.mse_loss(pred_conf[noobj_mask], target_conf[noobj_mask], reduction='sum') if noobj_mask.sum() > 0 else 0
        
        return obj_loss + self.lambda_noobj * noobj_loss
    
    def _classification_loss(self, pred_classes, target_classes, target_conf):
        """分类损失 - 增强版，解决类别预测偏向问题"""
        # 只计算有目标的网格的分类损失
        mask = target_conf.max(dim=-1)[0] > 0  # [B, grid_size, grid_size]
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred_classes.device)
        
        # 使用交叉熵损失替代MSE，更适合分类任务
        pred_classes_masked = pred_classes[mask]  # [N, num_classes]
        target_classes_masked = target_classes[mask]  # [N, num_classes]
        
        # 将one-hot编码转换为类别索引
        target_indices = torch.argmax(target_classes_masked, dim=-1)  # [N]
        
        # 使用交叉熵损失，自动处理类别平衡
        class_loss = F.cross_entropy(pred_classes_masked, target_indices, reduction='mean')
        
        return class_loss


def create_swin_yolo(num_classes=20, input_size=448, grid_size=7, num_boxes=2):
    """
    创建SwinYOLO检测器
    
    Args:
        num_classes: 类别数量
        input_size: 输入图像尺寸
        grid_size: 网格尺寸
        num_boxes: 每个网格预测的边界框数量
    
    Returns:
        model: SwinYOLO检测器
        loss_fn: 对应的损失函数
    """
    model = SwinYOLODetector(
        num_classes=num_classes,
        input_size=input_size, 
        grid_size=grid_size,
        num_boxes=num_boxes
    )
    
    loss_fn = SwinYOLOLoss(
        num_classes=num_classes,
        num_boxes=num_boxes
    )
    
    return model, loss_fn


if __name__ == "__main__":
    print("🚀 测试SwinYOLO检测器")
    
    # 创建模型
    model, loss_fn = create_swin_yolo(num_classes=20, input_size=448, grid_size=7)
    
    # 测试前向传播
    x = torch.randn(2, 3, 448, 448)
    output = model(x)
    
    print(f"✅ SwinYOLO测试成功!")
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   期望输出: [2, 7, 7, {2*5 + 20}]")
    
    # 测试参数组
    param_groups = model.get_parameter_groups()
    print(f"   参数组数量: {len(param_groups)}")
    for i, group in enumerate(param_groups):
        print(f"     组{i+1}: {group['name']}, 参数数量: {len(group['params'])}, 学习率: {group['lr']}")
    
    print(f"\n📊 模型统计:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数量: {total_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")
