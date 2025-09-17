import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，大幅减少参数量和计算量"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class EfficientBlock(nn.Module):
    """高效残差块，结合了深度可分离卷积和残差连接"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(EfficientBlock, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        
        self.dw_conv = DepthwiseSeparableConv(out_channels // 2, out_channels // 2, 3, stride, 1)
        
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接
        if not self.use_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        
        # 主分支
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dw_conv(out)
        out = self.bn2(self.conv2(out))
        
        # 残差连接
        if self.use_residual:
            out += residual
        else:
            out += self.shortcut(residual)
        
        return self.relu(out)


class EfficientDarkNet(nn.Module):
    """优化的高效DarkNet骨干网络"""
    def __init__(self, input_size=448):
        super(EfficientDarkNet, self).__init__()
        self.input_size = input_size
        
        # Stem层 - 快速下采样 448->112
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 448->224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 224->112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # 阶段1: 112->56
        self.stage1 = nn.Sequential(
            EfficientBlock(64, 128, stride=2),    # 下采样
            EfficientBlock(128, 128, stride=1),   # 特征提取
        )
        
        # 阶段2: 56->28
        self.stage2 = nn.Sequential(
            EfficientBlock(128, 256, stride=2),   # 下采样
            EfficientBlock(256, 256, stride=1),   # 特征提取
            EfficientBlock(256, 256, stride=1),   # 额外特征提取
        )
        
        # 阶段3: 28->14
        self.stage3 = nn.Sequential(
            EfficientBlock(256, 512, stride=2),   # 下采样
            EfficientBlock(512, 512, stride=1),   # 特征提取
        )
        
        # 阶段4: 14->7 (最终输出)
        self.stage4 = nn.Sequential(
            EfficientBlock(512, 512, stride=2),   # 下采样到7x7
            EfficientBlock(512, 512, stride=1),   # 最终特征提取
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem: 448->112
        x = self.stem(x)
        
        # 阶段1: 112->56
        x = self.stage1(x)
        
        # 阶段2: 56->28  
        x = self.stage2(x)
        
        # 阶段3: 28->14
        x = self.stage3(x)
        
        # 阶段4: 14->7
        x = self.stage4(x)
        
        return x


# 保持原有DarkNet作为备用
class DarkNet(nn.Module):
    """原始DarkNet（保持兼容性）"""
    def __init__(self, input_size=448):
        super(DarkNet, self).__init__()
        self.input_size = input_size
        
        # 第一阶段: 448->224 (调整回448输入)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 移除第一个maxpool，用stride=2的conv代替，减少信息损失
        
        # 第二阶段: 224->112
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 减少通道数，增加stride
        self.bn2 = nn.BatchNorm2d(128)
        
        # 第三阶段: 112->56 (减少层数，保持效率)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # stride=2代替maxpool
        self.bn6 = nn.BatchNorm2d(256)
        
        # 第四阶段: 56->28 (简化中间层)
        self.conv7 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # stride=2代替maxpool
        self.bn12 = nn.BatchNorm2d(512)
        
        # 第五阶段: 28->14 (最终特征提取)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.bn15 = nn.BatchNorm2d(256)
        self.conv16 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn17 = nn.BatchNorm2d(512)
        self.conv18 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)  # 最终下采样到7x7
        self.bn18 = nn.BatchNorm2d(512)
        
        # 最后两层保持不变
        self.conv19 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn19 = nn.BatchNorm2d(512)
        self.conv20 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn20 = nn.BatchNorm2d(512)
    def forward(self, x):
        # 第一阶段: 448->224
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 第二阶段: 224->112
        x = self.relu(self.bn2(self.conv2(x)))
        
        # 第三阶段: 112->56
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        
        # 第四阶段: 56->28
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
        x = self.relu(self.bn9(self.conv9(x)))
        x = self.relu(self.bn10(self.conv10(x)))
        x = self.relu(self.bn11(self.conv11(x)))
        x = self.relu(self.bn12(self.conv12(x)))
        
        # 第五阶段: 28->14
        x = self.relu(self.bn13(self.conv13(x)))
        x = self.relu(self.bn14(self.conv14(x)))
        x = self.relu(self.bn15(self.conv15(x)))
        x = self.relu(self.bn16(self.conv16(x)))
        x = self.relu(self.bn17(self.conv17(x)))
        x = self.relu(self.bn18(self.conv18(x)))
        
        # 最终特征
        x = self.relu(self.bn19(self.conv19(x)))
        x = self.relu(self.bn20(self.conv20(x)))

        return x
