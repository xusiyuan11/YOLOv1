# SwinYOLO: 基于Swin Transformer的YOLO检测器

本项目创新性地将Swin Transformer作为骨干网络与YOLO检测头结合，实现了高精度的目标检测算法。相比传统CNN骨干网络，SwinYOLO在保持实时性的同时显著提升了检测精度。

## 快速开始

### 环境要求

```bash
torch>=1.10.0
torchvision>=0.11.0
timm>=0.6.0              # Swin Transformer实现
opencv-python>=4.5.0
numpy>=1.19.0
tqdm
matplotlib
einops                   # 张量操作库
```

### 安装依赖

```bash
pip install torch torchvision timm einops opencv-python tqdm matplotlib
```

### 数据准备

```bash
# VOC2012数据集结构
data/VOC2012/VOCdevkit/VOC2012/
├── JPEGImages/          # 图像文件
├── Annotations/         # XML标注文件
└── ImageSets/Main/      # 数据集划分文件
```

## 使用方法

### 1. 端到端训练（推荐）

```bash
# 使用预训练Swin Transformer骨干网络
python train_swin_yolo.py

# 自定义配置训练
python train_swin_yolo.py --batch_size 8 --learning_rate 0.001 --epochs 100
```

### 2. 分阶段训练

```bash
# 第一阶段：冻结backbone，只训练检测头
python train_swin_yolo.py --freeze_backbone --epochs 20

# 第二阶段：解冻backbone，端到端微调
python train_swin_yolo.py --unfreeze_backbone --learning_rate 0.0001 --epochs 50
```

### 3. 模型评估和推理

```bash
# 模型评估
python evaluation.py --model_path ./checkpoints/swin_yolo/best_model.pth

# 单张图像推理
python inference.py --image ./test.jpg --model_path ./checkpoints/swin_yolo/best_model.pth

# 批量推理
python batch_inference.py --input_dir ./images/ --output_dir ./results/
```

### 4. 可视化分析

```bash
# 训练过程可视化
python visualization.py --log_dir ./checkpoints/swin_yolo/

# 注意力可视化
python visualize_attention.py --model_path ./checkpoints/swin_yolo/best_model.pth --image ./test.jpg
```

## 代码结构

```
SwinYOLO/
├── SwinYOLO.py              # 主模型定义
├── backbone.py              # Swin Transformer骨干网络
├── train_swin_yolo.py       # 训练脚本
├── evaluation.py            # 评估和mAP计算
├── dataset.py               # 数据加载和增强
├── Utils.py                 # 工具函数
├── visualization.py         # 可视化工具
├── inference.py             # 推理脚本
└── checkpoints/             # 模型检查点
```

## 模型架构

### 1. Swin Transformer骨干网络

```python
# Swin-Tiny配置
swin_config = {
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "patch_size": 4
}
```

### 2. 检测头设计

```python
# 检测头架构
detection_head = nn.Sequential(
    nn.Conv2d(512, 512, 3, padding=1),    # 特征增强
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.1),
    nn.Conv2d(512, 256, 3, padding=1),    # 降维
    nn.BatchNorm2d(256), 
    nn.LeakyReLU(0.1),
    nn.Conv2d(256, 30, 1)                 # 输出层: 2*5+20
)
```

### 3. 损失函数优化

```python
loss_weights = {
    "coord_loss": 1.0,      # 降低坐标损失权重
    "conf_loss": 1.0,       # 置信度损失
    "class_loss": 2.0       # 增强分类学习
}
```

## 核心创新

### 1. Hierarchical Feature Extraction

- **多尺度窗口**: 7×7窗口实现局部自注意力
- **Shifted Windows**: 跨窗口信息交互
- **Patch Merging**: 自然的下采样过程
- **全局感受野**: Transformer的长距离依赖建模

### 2. 自适应特征融合

- **Position Embedding**: 保留空间位置信息
- **Layer Normalization**: 稳定训练过程
- **Residual Connections**: 深层网络优化
- **Feature Pyramid**: 多层特征利用

### 3. 训练策略优化

- **渐进式解冻**: 先检测头后骨干网络
- **差分学习率**: 骨干网络用更小学习率
- **数据增强**: Mixup, CutMix, Mosaic
- **正则化**: DropPath, Label Smoothing

## 配置选项

### 模型配置

```json
{
  "model_config": {
    "swin_type": "swin_tiny_patch4_window7_224",
    "pretrained": true,
    "input_size": 448,
    "grid_size": 7,
    "num_boxes": 2,
    "num_classes": 20,
    "freeze_backbone": false
  }
}
```

### 训练配置

```json
{
  "training_config": {
    "batch_size": 8,
    "learning_rate": 0.001,
    "backbone_lr_ratio": 0.1,
    "epochs": 100,
    "warmup_epochs": 5,
    "weight_decay": 0.0001,
    "gradient_clip": 1.0
  }
}
```

### 数据增强配置

```json
{
  "augmentation_config": {
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "mosaic_prob": 0.5,
    "hsv_gain": [0.5, 0.5, 0.5],
    "rotation_degree": 10
  }
}
```

## 相关资源

- [Swin Transformer原文](https://arxiv.org/abs/2103.14030)
- [YOLO系列论文](https://pjreddie.com/darknet/yolo/)
- [Timm库文档](https://huggingface.co/docs/timm/)
- [Vision Transformer资源](https://github.com/lucidrains/vit-pytorch)

## 创新点总结

1. **首次将Swin Transformer用于YOLO架构**
2. **设计了高效的检测头适配方案**
3. **优化了损失函数权重平衡**
4. **实现了完整的训练和推理流程**
5. **提供了丰富的可视化和分析工具**


