"""
YOLOv3 Backbone Networks
包含DarkNet-53等骨干网络实现
"""
import torch
import torch.nn as nn


class DarkNetResidualBlock(nn.Module):
    """DarkNet-53的残差块"""
    def __init__(self, in_channels):
        super(DarkNetResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=False)
    
    def forward(self, x):
        residual = x
        
        out = self.leaky_relu(self.bn1(self.conv1(x)))
        out = self.leaky_relu(self.bn2(self.conv2(out)))
        
        out += residual
        return out


class DarkNet53(nn.Module):
    """
    DarkNet-53骨干网络，用于YOLOv3
    输入: (batch_size, 3, 416, 416)
    输出多个尺度的特征图，用于特征金字塔网络
    """
    def __init__(self, input_size=416):
        super(DarkNet53, self).__init__()
        self.input_size = input_size
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=False)
        
        # 下采样+残差块组合
        # 第1组: 32 -> 64, 416x416 -> 208x208
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.res_block1 = self._make_layer(64, 1)
        
        # 第2组: 64 -> 128, 208x208 -> 104x104
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.res_block2 = self._make_layer(128, 2)
        
        # 第3组: 128 -> 256, 104x104 -> 52x52 (用于52x52输出)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.res_block3 = self._make_layer(256, 8)
        
        # 第4组: 256 -> 512, 52x52 -> 26x26 (用于26x26输出)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.res_block4 = self._make_layer(512, 8)
        
        # 第5组: 512 -> 1024, 26x26 -> 13x13 (用于13x13输出)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(1024)
        self.res_block5 = self._make_layer(1024, 4)
        
        self._initialize_weights()
    
    def _make_layer(self, channels, num_blocks):
        """创建残差块层"""
        layers = []
        for _ in range(num_blocks):
            layers.append(DarkNetResidualBlock(channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播，返回多个尺度的特征图
        返回: (feat_52, feat_26, feat_13)
        """
        # 初始层: 416x416 -> 416x416
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        
        # 第1组: 416x416 -> 208x208
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.res_block1(x)
        
        # 第2组: 208x208 -> 104x104
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.res_block2(x)
        
        # 第3组: 104x104 -> 52x52 (保存用于FPN)
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.res_block3(x)
        feat_52 = x  # 256通道, 52x52, 用于小目标检测
        
        # 第4组: 52x52 -> 26x26 (保存用于FPN)
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.res_block4(x)
        feat_26 = x  # 512通道, 26x26, 用于中等目标检测
        
        # 第5组: 26x26 -> 13x13
        x = self.leaky_relu(self.bn6(self.conv6(x)))
        x = self.res_block5(x)
        feat_13 = x  # 1024通道, 13x13, 用于大目标检测
        
        return feat_52, feat_26, feat_13


# 为了保持兼容性，保留一个简化的接口
def create_backbone(backbone_type='darknet53', input_size=416):
    """
    创建骨干网络
    Args:
        backbone_type: 骨干网络类型 ('darknet53')
        input_size: 输入图像尺寸
    Returns:
        骨干网络模型
    """
    if backbone_type == 'darknet53':
        return DarkNet53(input_size=input_size)
    else:
        raise ValueError(f"不支持的骨干网络类型: {backbone_type}")