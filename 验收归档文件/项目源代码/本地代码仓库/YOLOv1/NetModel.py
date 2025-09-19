
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
        """保存训练好的backbone权重，确保键值完全匹配"""
        # 直接获取backbone的state_dict，这样就不会有任何前缀
        backbone_state_dict = self.backbone.state_dict()
        
        # 确保没有任何前缀（如果有的话，去掉它们）
        clean_backbone_state_dict = {}
        for key, value in backbone_state_dict.items():
            # 去掉可能的前缀
            clean_key = key
            if key.startswith('model.backbone.'):
                clean_key = key.replace('model.backbone.', '')
            elif key.startswith('backbone.'):
                clean_key = key.replace('backbone.', '')
            clean_backbone_state_dict[clean_key] = value
        
        # 创建一个临时的检测模型backbone来验证键值格式一致
        temp_backbone = self.backbone.__class__(input_size=self.backbone.input_size)
        temp_state_dict = temp_backbone.state_dict()
        
        # 验证所有键值都匹配
        missing_keys = set(temp_state_dict.keys()) - set(clean_backbone_state_dict.keys())
        unexpected_keys = set(clean_backbone_state_dict.keys()) - set(temp_state_dict.keys())
        
        if missing_keys:
            print(f"警告：缺失的层 {missing_keys}")
        if unexpected_keys:
            print(f"警告：多余的层 {unexpected_keys}")
        
        # 验证形状匹配
        shape_mismatch = []
        for key in temp_state_dict.keys():
            if key in clean_backbone_state_dict:
                if clean_backbone_state_dict[key].shape != temp_state_dict[key].shape:
                    shape_mismatch.append(f"{key}: {clean_backbone_state_dict[key].shape} vs {temp_state_dict[key].shape}")
        
        if shape_mismatch:
            print(f"警告：形状不匹配的层: {shape_mismatch}")
        
        backbone_state = {
            'backbone_state_dict': clean_backbone_state_dict,
            'backbone_class': self.backbone.__class__.__name__,
            'input_size': getattr(self.backbone, 'input_size', 448)
        }
        torch.save(backbone_state, save_path)
        print(f"预训练backbone已保存: {save_path}")
        print(f"保存的层数: {len(clean_backbone_state_dict)}")
        print(f"示例键值: {list(clean_backbone_state_dict.keys())[:5]}")


class DetectionModel(nn.Module):
    """检测模型，使用预训练好的backbone"""
    
    def __init__(self, class_num: int = 20, input_size: int = 448, 
                 grid_size: int = 7, use_efficient_backbone: bool = True, 
                 pretrained_backbone_path: str = None):
        super(DetectionModel, self).__init__()
        self.class_num = class_num
        self.input_size = input_size
        self.grid_size = grid_size
        self.feature_size = grid_size  # 使用传入的grid_size而不是计算值
        
        # 创建backbone
        if use_efficient_backbone:
            self.backbone = EfficientDarkNet(input_size=input_size)
        else:
            self.backbone = DarkNet(input_size=input_size)
        
        # 如果提供了预训练路径，尝试加载权重
        if pretrained_backbone_path:
            try:
                self.load_pretrained_backbone(pretrained_backbone_path)
            except Exception as e:
                print(f"加载预训练backbone失败: {e}")
                print("将使用随机初始化的backbone继续训练")
        else:
            print("backbone将通过后续权重加载初始化")
        
        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(128, 5 * 2 + self.class_num, kernel_size=1, stride=1, padding=0),
        )
        
        self.fc_detection = None 
        
        self.output_size = 5 * 2 + self.class_num  
        
    def forward(self, x):
        # 使用预训练的backbone提取特征
        features = self.backbone(x)  # (batch_size, 512, 7, 7)
        
        # 通过轻量化检测头直接输出（全卷积设计）
        detection_output = self.detection_head(features)  # (batch_size, 30, 7, 7)
        
        # 重新排列输出维度：(batch, channels, H, W) → (batch, H, W, channels)
        detection_output = detection_output.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 30)
        
        return detection_output
    
    def load_pretrained_backbone(self, pretrained_path: str):
        """加载分类器训练好的backbone权重，智能处理不同的键值前缀格式"""
        import os
        
        # 检查文件是否存在
        if not os.path.exists(pretrained_path):
            print(f"预训练backbone文件不存在: {pretrained_path}")
            print("将使用随机初始化的backbone")
            return
            
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            
            # 尝试获取backbone权重，可能的键名
            backbone_state_dict = None
            if 'backbone_state_dict' in checkpoint:
                backbone_state_dict = checkpoint['backbone_state_dict']
                print("找到backbone_state_dict键")
            elif 'model_state_dict' in checkpoint:
                # 如果是完整模型权重，尝试提取backbone部分
                full_state_dict = checkpoint['model_state_dict']
                backbone_state_dict = {}
                
                # 提取backbone相关的权重
                for key, value in full_state_dict.items():
                    if 'backbone' in key:
                        # 去掉各种可能的前缀
                        clean_key = key
                        if key.startswith('model.backbone.'):
                            clean_key = key.replace('model.backbone.', '')
                        elif key.startswith('backbone.'):
                            clean_key = key.replace('backbone.', '')
                        elif key.startswith('model.'):
                            clean_key = key.replace('model.', '')
                        backbone_state_dict[clean_key] = value
                print(f"从完整模型权重中提取backbone部分，共{len(backbone_state_dict)}个层")
            else:
                # 如果没有特定键，假设整个checkpoint就是backbone权重
                backbone_state_dict = checkpoint
                print("将整个checkpoint作为backbone权重")
            
            if not backbone_state_dict:
                print("无法从checkpoint中提取backbone权重")
                return
            
            print(f"正在加载预训练backbone: {pretrained_path}")
            print(f"权重文件包含 {len(backbone_state_dict)} 个层")
            print(f"示例键值: {list(backbone_state_dict.keys())[:5]}")
            
            # 获取当前backbone期望的键值
            current_backbone_dict = self.backbone.state_dict()
            print(f"当前backbone期望 {len(current_backbone_dict)} 个层")
            print(f"期望的键值示例: {list(current_backbone_dict.keys())[:5]}")
            
            # 智能匹配和加载
            matched_state_dict = {}
            for expected_key in current_backbone_dict.keys():
                found = False
                
                # 尝试直接匹配
                if expected_key in backbone_state_dict:
                    if backbone_state_dict[expected_key].shape == current_backbone_dict[expected_key].shape:
                        matched_state_dict[expected_key] = backbone_state_dict[expected_key]
                        found = True
                
                # 如果直接匹配失败，尝试添加各种前缀
                if not found:
                    for prefix in ['model.backbone.', 'backbone.', 'model.']:
                        prefixed_key = prefix + expected_key
                        if prefixed_key in backbone_state_dict:
                            if backbone_state_dict[prefixed_key].shape == current_backbone_dict[expected_key].shape:
                                matched_state_dict[expected_key] = backbone_state_dict[prefixed_key]
                                found = True
                                break
                
                if not found:
                    print(f"警告：无法找到匹配的权重: {expected_key}")
            
            print(f"成功匹配 {len(matched_state_dict)}/{len(current_backbone_dict)} 个层")
            
            # 加载匹配的权重
            missing_keys, unexpected_keys = self.backbone.load_state_dict(matched_state_dict, strict=False)
            
            if len(matched_state_dict) > 0:
                print(f"成功加载预训练backbone!")
                print(f"加载的层数: {len(matched_state_dict)}")
                if missing_keys:
                    print(f"未加载的层 (将使用随机初始化): {missing_keys}")
            else:
                print("警告：没有匹配到任何权重，使用随机初始化")
            
        except Exception as e:
            print(f"加载预训练backbone失败: {e}")
            print("将使用随机初始化的backbone继续训练")
    
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
        
        # 移除对fc_detection的引用，因为它被设置为None
        # for param in self.fc_detection.parameters():
        #     if param.requires_grad:
        #         detection_params.append(param)
        
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
                grid_size=grid_size,
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
                          grid_size: int = 7,
                          use_efficient_backbone: bool = True, 
                          pretrained_backbone_path: str = None):
    """创建检测模型"""
    return YOLOv1(
        class_num=class_num,
        grid_size=grid_size,
        training_mode='detection',
        input_size=input_size,
        use_efficient_backbone=use_efficient_backbone,
        pretrained_backbone_path=pretrained_backbone_path
    )
    
def create_yolo_model(class_num: int = 20, grid_size: int = 7, training_mode: str = 'detection'):
    """创建 YOLO 模型（兼容旧接口）"""
    return YOLOv1(class_num=class_num, grid_size=grid_size, training_mode=training_mode)
