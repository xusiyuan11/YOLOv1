# Swin Transformer Backbone for YOLO-Enhanced

## 概述

本项目成功将 **Swin Transformer** 集成为 YOLO-Enhanced 的 backbone，支持从 224x224 预训练权重适配到 448x448 输入尺寸。

## 功能特点

### ✅ 已实现功能
- **448x448 输入支持**: 通过位置编码插值技术，将224x224预训练模型适配到448x448
- **预训练权重加载**: 自动加载并适配 HuggingFace 缓存的预训练权重
- **灵活的模型格式支持**: 支持 `.bin`, `.pth`, `.safetensors` 格式
- **YOLO兼容输出**: 输出 `[B, 512, 14, 14]` 特征图，兼容现有YOLO架构
- **内存优化**: 智能的特征适配层，减少内存占用

### 🔧 技术细节
- **位置编码插值**: 使用双三次插值(bicubic)适配位置编码
- **输出适配层**: 将 Swin 的 `[B, H*W, C]` 输出转换为标准的 `[B, C, H, W]` 格式
- **错误处理**: 完善的异常处理和回退机制

## 安装要求

```bash
pip install timm  # 用于 Swin Transformer
pip install safetensors  # 可选，用于加载 .safetensors 格式权重
```

## 使用方法

### 1. 基本使用

```python
from backbone import create_backbone

# 创建 Swin Transformer backbone
model = create_backbone(
    backbone_type='swin',
    input_size=448,
    pretrained_path=r"C:\Users\asus\.cache\huggingface\hub\models--microsoft--swin-large-patch4-window7-224\snapshots\d433db83a1c10a34c365fc4928186c8fb8c642dd"
)

# 前向传播
import torch
x = torch.randn(1, 3, 448, 448)
features = model(x)  # 输出: [1, 512, 14, 14]
```

### 2. 在 YOLO 训练中使用

修改训练配置文件，将 backbone 类型设置为 'swin':

```python
# 训练配置示例
config = {
    'backbone_type': 'swin',
    'input_size': 448,
    'pretrained_path': r"C:\Users\asus\.cache\huggingface\hub\models--microsoft--swin-large-patch4-window7-224\snapshots\d433db83a1c10a34c365fc4928186c8fb8c642dd",
    # 其他配置...
}
```

### 3. 测试脚本

```bash
cd code/YOLO-enhanced
python test_swin_backbone.py
```

## 预训练权重路径

你的预训练权重位于:
```
C:\Users\asus\.cache\huggingface\hub\models--microsoft--swin-large-patch4-window7-224\snapshots\d433db83a1c10a34c365fc4928186c8fb8c642dd
```

系统会自动搜索以下格式的权重文件:
- `*.bin` (HuggingFace 格式)
- `*.pth` (PyTorch 格式)  
- `*.safetensors` (SafeTensors 格式)

## 模型架构

```
输入: [B, 3, 448, 448]
    ↓
Swin Transformer
    ↓
特征提取: [B, H*W, C]
    ↓
重塑为: [B, C, H, W]
    ↓
自适应池化到: [B, C, 14, 14]
    ↓
输出适配层: [B, 512, 14, 14]
```

## 性能对比

| Backbone | 参数量 | 推理时间 | 内存占用 | 精度 |
|----------|--------|----------|----------|------|
| EfficientBackbone | ~1M | ~10ms | ~4MB | 基线 |
| DarkNet | ~15M | ~15ms | ~60MB | 中等 |
| **Swin Transformer** | **~197M** | **~50ms** | **~788MB** | **最高** |

*注: 以上数据为448x448输入的估算值，实际性能可能因硬件而异*

## 优势与劣势

### ✅ 优势
- **更强的特征表示能力**: Transformer架构天然适合捕获长距离依赖
- **预训练优势**: 利用大规模ImageNet预训练，提升泛化能力
- **注意力机制**: 自适应地关注重要区域，提高检测精度
- **多尺度特征**: Swin的分层结构提供丰富的多尺度特征

### ⚠️ 劣势  
- **计算开销大**: 参数量和计算量远超传统CNN backbone
- **内存需求高**: 需要更多GPU内存
- **推理速度慢**: 适合精度优先的场景，不适合实时应用

## 适用场景

### 🎯 推荐使用
- **离线检测任务**: 对速度要求不高，但对精度要求很高
- **小目标检测**: Transformer的全局建模能力有助于小目标检测
- **复杂场景**: 目标密集、遮挡严重的复杂场景
- **研究实验**: 探索Transformer在目标检测中的潜力

### ❌不推荐使用
- **实时检测**: 移动端、嵌入式设备等资源受限环境
- **大批量推理**: 服务器端大规模并行推理
- **简单场景**: 目标明显、背景简单的场景

## 故障排除

### 1. timm 未安装
```
ImportError: timm is required for Swin Transformer
```
**解决**: `pip install timm`

### 2. 预训练权重路径错误
```
No checkpoint files found in /path/to/weights
```
**解决**: 检查路径是否正确，确保包含 `.bin`, `.pth` 或 `.safetensors` 文件

### 3. 内存不足
```
CUDA out of memory
```
**解决**: 减小 batch size 或使用 CPU 推理

### 4. 位置编码适配失败
```
Error adapting position embeddings
```
**解决**: 检查预训练权重是否为标准的 Swin-Large-224 格式

## 未来改进

- [ ] 支持更多 Swin 变体 (Swin-B, Swin-S, Swin-T)
- [ ] 实现知识蒸馏，将 Swin 知识迁移到轻量级模型
- [ ] 支持动态输入尺寸
- [ ] 优化推理速度 (TensorRT, ONNX)
- [ ] 添加量化支持

## 参考文献

1. [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
2. [timm: PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)
3. [HuggingFace Transformers](https://huggingface.co/microsoft/swin-large-patch4-window7-224)

---

**作者**: YOLO-Enhanced Team  
**更新时间**: 2025-09-11  
**版本**: v1.0
