# YOLOv1 复现实现

本项目实现了完整的YOLOv1目标检测算法，包含两阶段训练策略、高效的EfficientDarkNet骨干网络以及完整的训练和推理流程。

## 快速开始

### 环境要求

```bash
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.5.0
numpy>=1.19.0
tqdm
matplotlib
```

### 数据准备

```bash
# VOC2012数据集结构
data/VOC2012/VOCdevkit/VOC2012/
├── JPEGImages/          # 图像文件
├── Annotations/         # XML标注文件
└── ImageSets/Main/      # 数据集划分文件

# COCO数据集结构（可选，用于backbone预训练）
data/COCO/
├── train2017/           # 训练图像
├── val2017/             # 验证图像
└── annotations/         # JSON标注文件
```

## 使用方法

### 1. 单阶段检测训练（推荐）

```bash
# 直接训练完整的检测模型
python Train_Detection.py

# 自定义参数
python Train_Detection.py --batch_size 16 --learning_rate 0.001 --epochs 100
```

### 2. 两阶段训练（原始YOLO方法）

```bash
# 第一阶段：backbone分类预训练
python Train_Classification.py

# 第二阶段：检测微调
python Train_Detection.py --pretrained_backbone ./checkpoints/classification/best_classification_model.pth

# 完整两阶段流程
python Train_Complete.py
```

### 3. 模型测试和推理

```bash
# 测试模型性能
python Test.py --model_path ./checkpoints/detection/best_detection_model.pth

# 可视化检测结果
python result_visualisation.py --model_path ./checkpoints/detection/best_detection_model.pth --image_dir ./test_images/
```

## 代码结构

```
YOLOv1/
├── backbone.py              # EfficientDarkNet骨干网络
├── NetModel.py              # YOLO检测网络
├── YOLOLoss.py              # YOLO损失函数
├── dataset.py               # 数据加载和预处理
├── Utils.py                 # 工具函数（mAP计算等）
├── Train_Detection.py       # 检测训练脚本
├── Train_Classification.py  # 分类预训练脚本
├── Train_Complete.py        # 完整两阶段训练
├── Test.py                  # 模型测试
├── visualization.py         # 可视化工具
└── two_stage_config.json    # 训练配置文件
```

## 核心配置

### 模型配置

```json
{
  "model_config": {
    "grid_size": 7,           # 网格大小 7×7
    "num_boxes": 2,           # 每个网格预测的边界框数量
    "num_classes": 20,        # VOC类别数
    "input_size": 448,        # 输入图像尺寸
    "use_efficient_backbone": true  # 使用高效骨干网络
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 100,
    "lambda_coord": 5.0,      # 坐标损失权重
    "lambda_noobj": 0.5       # 无目标置信度损失权重
  }
}
```

### 数据配置

```json
{
  "data_config": {
    "voc2012_jpeg_dir": "../../data/VOC2012/VOCdevkit/VOC2012/JPEGImages",
    "voc2012_anno_dir": "../../data/VOC2012/VOCdevkit/VOC2012/Annotations",
    "class_file": "./voc_classes.txt",
    "grid_size": 7,
    "input_size": 448
  }
}
```

## 关键特性

### 1. 高效骨干网络

- **EfficientDarkNet**: 结合深度可分离卷积和残差连接
- **参数优化**: 相比原版DarkNet减少50%参数量
- **多尺度训练**: 支持不同输入尺寸

### 2. 改进的损失函数

- **坐标损失**: MSE损失，lambda_coord=5.0
- **置信度损失**: 有目标和无目标分别处理
- **分类损失**: 多类别交叉熵损失
- **梯度裁剪**: 防止梯度爆炸

### 3. 数据增强

- **随机裁剪和缩放**
- **颜色抖动**
- **水平翻转**

### 调试模式

```bash
# 开启详细日志
python Train_Detection.py --debug --verbose

# 可视化训练数据
python dataset.py  # 直接运行查看数据加载
```
