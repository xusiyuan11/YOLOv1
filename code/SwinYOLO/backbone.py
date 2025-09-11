import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import os

try:
    import timm
    from timm.models.swin_transformer import SwinTransformer
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Swin Transformer backbone will not be available.")



class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone，支持448x448输入
    基于Microsoft的Swin Transformer，通过位置编码插值支持更大的输入尺寸
    """
    def __init__(self, 
                 input_size=448, 
                 pretrained_path=r"C:\Users\asus\.cache\huggingface\hub\models--microsoft--swin-large-patch4-window7-224\snapshots\d433db83a1c10a34c365fc4928186c8fb8c642dd",
                 model_name='swin_large_patch4_window7_224'):
        super(SwinTransformerBackbone, self).__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Swin Transformer backbone. Install with: pip install timm")
        
        self.input_size = input_size
        self.pretrained_path = pretrained_path
        
        # 创建Swin Transformer模型
        self.swin = timm.create_model(
            model_name,
            pretrained=False,  # 我们会手动加载预训练权重
            num_classes=0,     # 移除分类头
            global_pool='',    # 移除全局池化
            img_size=input_size  # 设置输入尺寸
        )
        
        # 获取特征维度 - 需要动态获取，因为可能不准确
        self.feature_dim = None  # 将在第一次前向传播时确定
        
        # 输出适配层将在第一次前向传播时创建
        self.output_adapter = None
        
        # 加载预训练权重
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path):
        """
        加载预训练权重并适配到448x448输入尺寸
        """
        try:
            # 如果是本地路径
            if os.path.exists(pretrained_path):
                checkpoint_files = list(Path(pretrained_path).glob("*.bin")) + \
                                 list(Path(pretrained_path).glob("*.pth")) + \
                                 list(Path(pretrained_path).glob("*.safetensors"))
                
                if checkpoint_files:
                    checkpoint_file = checkpoint_files[0]
                    print(f"Loading pretrained weights from: {checkpoint_file}")
                    
                    # 尝试不同的加载方式
                    try:
                        if str(checkpoint_file).endswith('.safetensors'):
                            # 需要安装 safetensors 库
                            try:
                                from safetensors.torch import load_file
                                state_dict = load_file(checkpoint_file)
                            except ImportError:
                                print("safetensors not available, trying torch.load")
                                state_dict = torch.load(checkpoint_file, map_location='cpu')
                        else:
                            state_dict = torch.load(checkpoint_file, map_location='cpu')
                        
                        # 适配位置编码到新的输入尺寸
                        state_dict = self.adapt_position_embeddings(state_dict)
                        
                        # 加载权重
                        missing_keys, unexpected_keys = self.swin.load_state_dict(state_dict, strict=False)
                        print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                        
                    except Exception as e:
                        print(f"Error loading checkpoint: {e}")
                        print("Continuing with random initialization...")
                else:
                    print(f"No checkpoint files found in {pretrained_path}")
            else:
                print(f"Pretrained path {pretrained_path} does not exist")
                
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Continuing with random initialization...")
    
    def adapt_position_embeddings(self, state_dict):
        """
        适配位置编码从224x224到448x448
        """
        # Swin Transformer的位置编码通常在各个stage中
        adapted_state_dict = {}
        
        for key, value in state_dict.items():
            # 简化处理，主要针对绝对位置编码
            if 'absolute_pos_embed' in key and value.dim() == 2:
                # 计算原始和目标的网格大小
                total_tokens = value.shape[1]
                
                # 检查是否有class token (通常第一个token)
                if total_tokens > 1:
                    # 假设没有class token，直接计算网格大小
                    old_size = int(total_tokens ** 0.5)
                    new_size = self.input_size // 32  # Swin的默认下采样率
                    
                    if old_size * old_size == total_tokens and old_size != new_size:
                        print(f"Interpolating position embeddings from {old_size}x{old_size} to {new_size}x{new_size}")
                        
                        try:
                            # 重塑为网格形状
                            embed_dim = value.shape[-1]
                            pos_embed = value.reshape(1, old_size, old_size, embed_dim).permute(0, 3, 1, 2)
                            
                            # 插值到新尺寸
                            pos_embed = F.interpolate(pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
                            
                            # 重新展平
                            value = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
                            
                        except Exception as e:
                            print(f"Failed to interpolate {key}: {e}, keeping original")
                
                elif 'relative_position_bias_table' in key:
                    # 相对位置偏置表的处理比较复杂，暂时保持原样
                    print(f"Skipping relative position bias adaptation for {key}")
            
            adapted_state_dict[key] = value
        
        return adapted_state_dict
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [B, 3, 448, 448]
        Returns:
            output: 特征张量 [B, 512, 14, 14] (适配YOLO需要的尺寸)
        """
        # 通过Swin Transformer提取特征
        features = self.swin.forward_features(x)
        
        # 检查Swin输出的形状，可能是 [B, C, H, W] 或 [B, H*W, C]
        if len(features.shape) == 4:  # [B, C, H, W] 格式
            B, C, H, W = features.shape
        elif len(features.shape) == 3:  # [B, H*W, C] 格式  
            B, HW, C = features.shape
            H = W = int(HW ** 0.5)  # 假设H=W
            # 重塑为标准的卷积特征图格式
            features = features.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        else:
            raise ValueError(f"Unexpected Swin output shape: {features.shape}")
        
        # 动态创建输出适配层（仅在第一次调用时）
        if self.output_adapter is None:
            self.feature_dim = features.shape[1]  # 获取真实的特征维度
            print(f"Creating output adapter with input dim: {self.feature_dim}")
            self.output_adapter = nn.Sequential(
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ).to(features.device)  # 确保在正确的设备上
        
        # 如果需要，调整到目标尺寸 (14x14 for YOLO)
        if features.shape[-1] != 14 or features.shape[-2] != 14:
            features = F.adaptive_avg_pool2d(features, (14, 14))
        
        # 通过输出适配层
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
        adapted_features = self.output_adapter(features_flat)  # [B*H*W, 512]
        adapted_features = adapted_features.reshape(B, H, W, 512).permute(0, 3, 1, 2)  # [B, 512, H, W]
        
        return adapted_features


# 创建backbone的工厂函数
def create_backbone(backbone_type='swin', input_size=448, **kwargs):
    """
    创建backbone网络的工厂函数
    Args:
        backbone_type: 目前只支持 'swin' (Swin Transformer Large)
        input_size: 输入图像尺寸
        **kwargs: 其他参数
    """
    if backbone_type == 'swin':
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for Swin Transformer. Install with: pip install timm")
        return SwinTransformerBackbone(input_size=input_size, **kwargs)
    else:
        raise ValueError(f"Unsupported backbone type: {backbone_type}. Currently only 'swin' is supported.")



if __name__ == "__main__":
    # 默认测试Swin Transformer backbone
    if TIMM_AVAILABLE:
        print("测试Swin Transformer Backbone (默认)...")
        try:
            swin_model = create_backbone()  # 默认使用swin
            x = torch.randn(1, 3, 448, 448)
            y = swin_model(x)
            print(f"Swin输出形状: {y.shape}")  # 应该是 [1, 512, 14, 14]
            
            print(f"\nSwin总参数量: {sum(p.numel() for p in swin_model.parameters()):,}")
            print(f"Swin可训练参数量: {sum(p.numel() for p in swin_model.parameters() if p.requires_grad):,}")
            print("✅ Swin Transformer backbone 测试成功!")
        except Exception as e:
            print(f"❌ Swin Transformer测试失败: {e}")
    else:
        print("❌ timm未安装，无法使用Swin Transformer")
        print("请运行: pip install timm")
