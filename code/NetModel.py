import torch
import torch.nn as nn
from backbone import DarkNet, EfficientDarkNet


class ClassificationModel(nn.Module):
    """独立的分类模型，专门用于预训练backbone"""
    
    def __init__(self, backbone, num_classes: int = 80):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        # 分类头：轻量化设计，主要目的是训练backbone
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # 通过backbone提取特征（这是我们要训练的重点）
        features = self.backbone(x)  # (batch_size, 512, 7, 7)
        # 分类预测（提供监督信号）
        output = self.classifier(features)
        return output
    
    def save_trained_backbone(self, save_path: str):
        """保存训练好的backbone权重"""
        backbone_state = {
            'backbone_state_dict': self.backbone.state_dict(),
            'backbone_class': self.backbone.__class__.__name__,
            'input_size': getattr(self.backbone, 'input_size', 448)
        }
        torch.save(backbone_state, save_path)
        print(f"预训练backbone已保存: {save_path}")


class DetectionModel(nn.Module):
    """检测模型，使用预训练好的backbone"""
    
    def __init__(self, class_num: int = 20, input_size: int = 448, 
                 use_efficient_backbone: bool = True, pretrained_backbone_path: str = None):
        super(DetectionModel, self).__init__()
        self.class_num = class_num
        self.input_size = input_size
        self.feature_size = input_size // 64  # 7x7 for 448x448 input
        
        # 创建backbone
        if use_efficient_backbone:
            self.backbone = EfficientDarkNet(input_size=input_size)
        else:
            self.backbone = DarkNet(input_size=input_size)
        
        # 如果提供了预训练路径，加载权重
        if pretrained_backbone_path:
            self.load_pretrained_backbone(pretrained_backbone_path)
        
        # 检测头：轻量化设计
        self.detection_head = nn.Sequential(
            # 先降维减少参数
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # 直接输出到目标通道数，避免大的全连接层
            nn.Conv2d(128, 5 * 2 + self.class_num, kernel_size=1, stride=1, padding=0),
        )
        
        # 移除大的全连接层，使用全卷积设计
        self.fc_detection = None  # 不再使用全连接层
        
        # 输出尺寸
        self.output_size = 5 * 2 + self.class_num  # 2个边界框 + 类别数
        
    def forward(self, x):
        # 使用预训练的backbone提取特征
        features = self.backbone(x)  # (batch_size, 512, 7, 7)
        
        # 通过轻量化检测头直接输出（全卷积设计）
        detection_output = self.detection_head(features)  # (batch_size, 30, 7, 7)
        
        # 重新排列输出维度：(batch, channels, H, W) → (batch, H, W, channels)
        detection_output = detection_output.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 30)
        
        return detection_output
    
    def load_pretrained_backbone(self, pretrained_path: str):
        """加载分类器训练好的backbone权重"""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            backbone_state_dict = checkpoint['backbone_state_dict']
            
            # 加载权重到backbone
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=True)
            
            print(f"成功加载预训练backbone: {pretrained_path}")
            
        except Exception as e:
            print(f"加载预训练backbone失败: {e}")
            print("将使用随机初始化")
    
    def freeze_backbone(self):
        """冻结backbone，只训练检测头"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone已冻结")
    
    def unfreeze_backbone(self):
        """解冻backbone，允许端到端微调"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone已解冻")
    
    def get_parameter_groups(self, base_lr: float, backbone_lr_ratio: float = 0.1):
        """获取不同学习率的参数组"""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        detection_params = []
        
        for param in self.detection_head.parameters():
            if param.requires_grad:
                detection_params.append(param)
        for param in self.fc_detection.parameters():
            if param.requires_grad:
                detection_params.append(param)
        
        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params, 
                'lr': base_lr * backbone_lr_ratio, 
                'name': 'backbone'
            })
        if detection_params:
            param_groups.append({
                'params': detection_params, 
                'lr': base_lr, 
                'name': 'detection_head'
            })
        
        return param_groups


class YOLOv1(nn.Module):
    """YOLO工厂类：根据训练模式创建分类模型或检测模型"""
    
    def __init__(self, class_num: int = 20, grid_size: int = 7, training_mode: str = 'detection', 
                 input_size: int = 448, use_efficient_backbone: bool = True, pretrained_backbone_path: str = None):
        super(YOLOv1, self).__init__()
        self.class_num = class_num
        self.grid_size = grid_size
        self.input_size = input_size
        self.training_mode = training_mode
        self.use_efficient_backbone = use_efficient_backbone
        
        if training_mode == 'classification':
            # 创建backbone
            if use_efficient_backbone:
                backbone = EfficientDarkNet(input_size=input_size)
            else:
                backbone = DarkNet(input_size=input_size)
            
            # 创建分类模型
            self.model = ClassificationModel(backbone, num_classes=class_num)
            
        elif training_mode == 'detection':
            # 创建检测模型
            self.model = DetectionModel(
                class_num=class_num,
                input_size=input_size,
                use_efficient_backbone=use_efficient_backbone,
                pretrained_backbone_path=pretrained_backbone_path
            )
        
        else:
            raise ValueError(f"不支持的训练模式: {training_mode}")
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def set_training_mode(self, mode: str):
        """设置训练模式"""
        self.training_mode = mode
        
        if hasattr(self.model, 'freeze_backbone'):
            if mode == 'detection_frozen':
                self.model.freeze_backbone()
            elif mode == 'detection_finetune':
                self.model.unfreeze_backbone()
    
    def save_backbone_weights(self, save_path: str):
        """保存backbone权重"""
        if isinstance(self.model, ClassificationModel):
            self.model.save_trained_backbone(save_path)
        else:
            print("当前不是分类模型，无法保存backbone权重")
    
    def load_pretrained_backbone(self, pretrained_path: str):
        """加载预训练backbone"""
        if isinstance(self.model, DetectionModel):
            self.model.load_pretrained_backbone(pretrained_path)
        else:
            print("当前不是检测模型，无需加载backbone权重")
    
    def freeze_backbone(self):
        """冻结backbone"""
        if hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
    
    def unfreeze_backbone(self):
        """解冻backbone"""
        if hasattr(self.model, 'unfreeze_backbone'):
            self.model.unfreeze_backbone()
    
    def get_backbone_learning_rates(self, base_lr: float, backbone_lr_ratio: float = 0.1):
        """获取分组学习率"""
        if hasattr(self.model, 'get_parameter_groups'):
            return self.model.get_parameter_groups(base_lr, backbone_lr_ratio)
        else:
            # 分类模型使用统一学习率
            return [{'params': self.parameters(), 'lr': base_lr, 'name': 'all'}]


def create_classification_model(num_classes: int = 80, input_size: int = 448, 
                              use_efficient_backbone: bool = True):
    """创建分类模型用于预训练backbone"""
    return YOLOv1(
        class_num=num_classes,
        training_mode='classification',
        input_size=input_size,
        use_efficient_backbone=use_efficient_backbone
    )


def create_detection_model(class_num: int = 20, input_size: int = 448,
                          use_efficient_backbone: bool = True, 
                          pretrained_backbone_path: str = None):
    """创建检测模型"""
    return YOLOv1(
        class_num=class_num,
        training_mode='detection',
        input_size=input_size,
        use_efficient_backbone=use_efficient_backbone,
        pretrained_backbone_path=pretrained_backbone_path
    )
    
def create_yolo_model(class_num: int = 20, grid_size: int = 7, training_mode: str = 'detection'):
    """创建 YOLO 模型（兼容旧接口）"""
    return YOLOv1(class_num=class_num, grid_size=grid_size, training_mode=training_mode)


def test_two_stage_training():
    """测试两阶段训练流程"""
    print("测试分离式两阶段训练流程")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 阶段1：创建分类模型训练backbone
    print("\n阶段1：分类预训练")
    classification_model = create_classification_model(num_classes=80, input_size=448)
    classification_model = classification_model.to(device)
    
    # 检查参数
    total_params = sum(p.numel() for p in classification_model.parameters())
    trainable_params = sum(p.numel() for p in classification_model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 测试前向传播
    x = torch.randn(2, 3, 448, 448).to(device)
    with torch.no_grad():
        cls_output = classification_model(x)
    print(f"分类输出: {cls_output.shape}")
    
    # 模拟保存backbone
    import tempfile
    import os
    temp_dir = tempfile.mkdtemp()
    backbone_path = os.path.join(temp_dir, 'trained_backbone.pth')
    classification_model.save_backbone_weights(backbone_path)
    
    # 阶段2：创建检测模型使用预训练backbone
    print("\n阶段2：检测微调")
    detection_model = create_detection_model(
        class_num=20, 
        input_size=448,
        pretrained_backbone_path=backbone_path
    )
    detection_model = detection_model.to(device)
    
    # 检查参数
    total_params = sum(p.numel() for p in detection_model.parameters())
    trainable_params = sum(p.numel() for p in detection_model.parameters() if p.requires_grad)
    print(f"参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 测试前向传播
    with torch.no_grad():
        det_output = detection_model(x)
    print(f"检测输出: {det_output.shape}")
    
    # 测试冻结backbone
    print("\n训练策略测试:")
    detection_model.freeze_backbone()
    frozen_trainable = sum(p.numel() for p in detection_model.parameters() if p.requires_grad)
    print(f"冻结backbone - 可训练参数: {frozen_trainable:,}")
    
    # 测试解冻backbone
    detection_model.unfreeze_backbone()
    unfrozen_trainable = sum(p.numel() for p in detection_model.parameters() if p.requires_grad)
    print(f"解冻backbone - 可训练参数: {unfrozen_trainable:,}")
    
    # 清理临时文件
    os.remove(backbone_path)
    os.rmdir(temp_dir)
    
    print("\n两阶段训练流程测试完成！")


def test_yolo_model():
    """测试 YOLO 模型（兼容性测试）"""
    print("\n兼容性测试:")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_yolo_model(class_num=20).to(device)
    
    print(f"设备: {device}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试输入
    x = torch.randn(2, 3, 448, 448).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
    
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    expected_shape = (2, 7, 7, 30)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    print("兼容性测试通过！")


if __name__ == "__main__":
    test_two_stage_training()
    test_yolo_model()
