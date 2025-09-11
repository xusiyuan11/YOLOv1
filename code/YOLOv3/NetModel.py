"""
YOLOv3网络模型实现
包含特征金字塔网络(FPN)和多尺度检测头
"""
import torch
import torch.nn as nn
from backbone import DarkNet53


class ConvBlock(nn.Module):
    """基础卷积块：Conv + BN + LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super(ConvBlock, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=False)
    
    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))


class YOLOv3DetectionHead(nn.Module):
    """YOLOv3检测头"""
    def __init__(self, in_channels, num_classes=20):
        super(YOLOv3DetectionHead, self).__init__()
        self.num_classes = num_classes
        # 每个锚框预测: 5(x,y,w,h,conf) + num_classes
        self.num_anchors = 3
        self.output_channels = self.num_anchors * (5 + num_classes)
        
        # 检测头网络
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels, in_channels * 2, 3),
            ConvBlock(in_channels * 2, in_channels, 1),
            ConvBlock(in_channels, in_channels * 2, 3),
            ConvBlock(in_channels * 2, in_channels, 1),
            ConvBlock(in_channels, in_channels * 2, 3),
        )
        
        # 最终预测层
        self.final_conv = nn.Conv2d(in_channels * 2, self.output_channels, 1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        prediction = self.final_conv(x)
        return prediction


class YOLOv3FPN(nn.Module):
    """
    YOLOv3特征金字塔网络
    实现多尺度特征融合和检测
    """
    def __init__(self, num_classes=20):
        super(YOLOv3FPN, self).__init__()
        self.num_classes = num_classes
        self.backbone = DarkNet53()
        
        # 13x13尺度的处理网络
        self.conv_set_13 = nn.Sequential(
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
            ConvBlock(512, 1024, 3),
            ConvBlock(1024, 512, 1),
        )
        
        # 13x13检测头
        self.detection_13 = YOLOv3DetectionHead(512, num_classes)
        
        # 13x13 -> 26x26的上采样路径
        self.conv_13_to_26 = ConvBlock(512, 256, 1)
        self.upsample_13_to_26 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 26x26尺度的处理网络 (256 + 512 = 768输入)
        self.conv_set_26 = nn.Sequential(
            ConvBlock(768, 256, 1),  # 768 = 256(上采样) + 512(skip connection)
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
            ConvBlock(256, 512, 3),
            ConvBlock(512, 256, 1),
        )
        
        # 26x26检测头
        self.detection_26 = YOLOv3DetectionHead(256, num_classes)
        
        # 26x26 -> 52x52的上采样路径
        self.conv_26_to_52 = ConvBlock(256, 128, 1)
        self.upsample_26_to_52 = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 52x52尺度的处理网络 (128 + 256 = 384输入)
        self.conv_set_52 = nn.Sequential(
            ConvBlock(384, 128, 1),  # 384 = 128(上采样) + 256(skip connection)
            ConvBlock(128, 256, 3),
            ConvBlock(256, 128, 1),
            ConvBlock(128, 256, 3),
            ConvBlock(256, 128, 1),
        )
        
        # 52x52检测头
        self.detection_52 = YOLOv3DetectionHead(128, num_classes)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入图像 (batch_size, 3, 416, 416)
        Returns:
            三个尺度的预测结果 (pred_13, pred_26, pred_52)
        """
        # 通过DarkNet-53获取多尺度特征
        feat_52, feat_26, feat_13 = self.backbone(x)
        
        # === 13x13尺度处理 ===
        x_13 = self.conv_set_13(feat_13)  # (batch, 512, 13, 13)
        pred_13 = self.detection_13(x_13)  # (batch, 75, 13, 13) for 20 classes
        
        # === 13x13 -> 26x26 特征融合 ===
        # 上采样13x13特征
        x_13_up = self.conv_13_to_26(x_13)  # (batch, 256, 13, 13)
        x_13_up = self.upsample_13_to_26(x_13_up)  # (batch, 256, 26, 26)
        
        # 与26x26特征融合
        x_26 = torch.cat([x_13_up, feat_26], dim=1)  # (batch, 768, 26, 26)
        x_26 = self.conv_set_26(x_26)  # (batch, 256, 26, 26)
        pred_26 = self.detection_26(x_26)  # (batch, 75, 26, 26)
        
        # === 26x26 -> 52x52 特征融合 ===
        # 上采样26x26特征
        x_26_up = self.conv_26_to_52(x_26)  # (batch, 128, 26, 26)
        x_26_up = self.upsample_26_to_52(x_26_up)  # (batch, 128, 52, 52)
        
        # 与52x52特征融合
        x_52 = torch.cat([x_26_up, feat_52], dim=1)  # (batch, 384, 52, 52)
        x_52 = self.conv_set_52(x_52)  # (batch, 128, 52, 52)
        pred_52 = self.detection_52(x_52)  # (batch, 75, 52, 52)
        
        return pred_13, pred_26, pred_52


class YOLOv3Model(nn.Module):
    """
    完整的YOLOv3模型
    包含预定义锚框信息
    """
    def __init__(self, num_classes=20, input_size=416):
        super(YOLOv3Model, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        
        # YOLOv3预定义锚框 (w, h) - 相对于416x416
        self.anchors = [
            # 13x13尺度 - 大目标
            [(116, 90), (156, 198), (373, 326)],
            # 26x26尺度 - 中等目标  
            [(30, 61), (62, 45), (59, 119)],
            # 52x52尺度 - 小目标
            [(10, 13), (16, 30), (33, 23)]
        ]
        
        # 主网络
        self.fpn = YOLOv3FPN(num_classes)
    
    def forward(self, x):
        """前向传播"""
        return self.fpn(x)
    
    def get_anchors(self, scale_idx):
        """获取指定尺度的锚框"""
        return self.anchors[scale_idx]


# 创建模型的工厂函数
def create_yolov3_model(num_classes=20, input_size=416):
    """
    创建YOLOv3模型
    Args:
        num_classes: 类别数量
        input_size: 输入图像尺寸
    Returns:
        YOLOv3模型
    """
    return YOLOv3Model(num_classes=num_classes, input_size=input_size)


# 创建分类模型
def create_classification_model(num_classes=1000, input_size=416, use_efficient_backbone=True):
    """
    创建用于分类训练的模型
    Args:
        num_classes: 分类类别数
        input_size: 输入图像尺寸
        use_efficient_backbone: 是否使用高效骨干网络
    Returns:
        分类模型
    """
    from backbone import create_backbone
    import torch.nn as nn
    
    # 创建骨干网络
    backbone = create_backbone('darknet53', input_size)
    
    # 创建分类头
    class ClassificationModel(nn.Module):
        def __init__(self, backbone, num_classes):
            super(ClassificationModel, self).__init__()
            self.backbone = backbone
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(1024, num_classes)  # DarkNet53输出1024维特征
            
        def forward(self, x):
            # 通过骨干网络
            _, _, features = self.backbone(x)  # 取最后一层特征
            # 全局平均池化
            pooled = self.global_avg_pool(features)
            pooled = pooled.view(pooled.size(0), -1)
            # 分类
            output = self.classifier(pooled)
            return output
    
    return ClassificationModel(backbone, num_classes)


# 保持向后兼容的接口
def create_detection_model(class_num=20, input_size=416, **kwargs):
    """创建检测模型（兼容旧接口）"""
    return create_yolov3_model(num_classes=class_num, input_size=input_size)