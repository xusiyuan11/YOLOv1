# YOLOv3 目标检测项目

基于PyTorch实现的YOLOv3目标检测系统，支持训练、测试和可视化功能。

## 项目特性

- **YOLOv3网络架构**: 基于DarkNet-53骨干网络的特征金字塔结构
- **多尺度检测**: 13x13、26x26、52x52三个检测尺度
- **高效训练**: 支持GPU加速训练和混合精度
- **灵活数据集**: 支持VOC和COCO数据格式
- **完整流程**: 从训练到测试的完整检测流程
- **结果可视化**: 丰富的检测结果可视化工具

## 项目结构

```
YOLOv3/
├── backbone.py              # DarkNet-53骨干网络
├── NetModel.py              # YOLOv3主网络模型
├── YOLOLoss.py              # YOLOv3损失函数
├── postprocess.py           # 后处理模块(NMS、解码)
├── anchors.py               # 锚框系统
├── Train_Detection.py       # 检测训练器
├── Test.py                  # 测试和评估
├── dataset.py               # 数据集加载器
├── Utils.py                 # 工具函数集合
├── OPT.py                   # 优化器管理
├── result_visualisation.py  # 结果可视化
├── compatibility_fixes.py   # 兼容性修复
└── main.py                  # 主程序入口
```

## 环境要求

```bash
Python >= 3.10
PyTorch >= 2.0
CUDA >= 11.8 (可选，GPU加速)
```

### 依赖包安装

```bash
conda activate python3.10
pip install torch torchvision opencv-python numpy tqdm matplotlib
```

## 核心组件

### 1. 网络架构 (`NetModel.py`)

- **DarkNet-53骨干网络**: 53层卷积神经网络
- **特征金字塔网络(FPN)**: 多尺度特征融合
- **检测头**: 三个尺度的检测输出
- **参数量**: 75,399,361个可训练参数

```python
from NetModel import create_yolov3_model

# 创建YOLOv3模型
model = create_yolov3_model(num_classes=20, input_size=416)
```

### 2. 损失函数 (`YOLOLoss.py`)

- **位置损失**: 边界框回归损失
- **置信度损失**: 目标性预测损失
- **分类损失**: 类别预测损失
- **多尺度损失**: 三个检测尺度的联合损失

```python
from YOLOLoss import YOLOv3Loss

criterion = YOLOv3Loss(num_classes=20, input_size=416)
```

### 3. 后处理 (`postprocess.py`)

- **预测解码**: 将网络输出转换为检测框
- **非极大值抑制(NMS)**: 去除重复检测
- **多尺度融合**: 合并三个尺度的检测结果
- **坐标转换**: 相对坐标转绝对坐标

```python
from postprocess import postprocess_yolov3

detections = postprocess_yolov3(
    predictions,
    conf_threshold=0.5,
    nms_threshold=0.4
)
```

### 4. 锚框系统 (`anchors.py`)

- **预定义锚框**: 针对不同尺度优化的锚框
- **多尺度锚框**: 每个检测尺度3个锚框
- **自适应锚框**: 支持数据集自适应锚框生成

```python
# YOLOv3预定义锚框 (w, h)
anchors = [
    [(116, 90), (156, 198), (373, 326)],  # 13x13尺度 - 大目标
    [(30, 61), (62, 45), (59, 119)],      # 26x26尺度 - 中等目标  
    [(10, 13), (16, 30), (33, 23)]        # 52x52尺度 - 小目标
]
```

## 使用指南

### 1. 快速开始

```bash
# 系统演示
python main.py demo

# 训练模型
python main.py train

# 测试模型
python main.py test

# 结果可视化
python main.py visualize
```

### 2. 训练模型

```python
from Train_Detection import DetectionTrainer
from Utils import load_hyperparameters

# 加载配置
hyperparams = load_hyperparameters()

# 创建训练器
trainer = DetectionTrainer(hyperparams)

# 开始训练
trainer.train()
```

### 3. 模型测试

```python
from Test import YOLOv3Tester
from NetModel import create_yolov3_model

# 创建模型和测试器
model = create_yolov3_model(num_classes=20)
tester = YOLOv3Tester(model)

# 单张图像检测
detections = tester.detect_single_image("image.jpg", "output.jpg")

# 批量图像检测
results = tester.batch_detect_images("images/", "outputs/")
```

### 4. 自定义配置

```python
from Utils import load_hyperparameters, save_hyperparameters

# 加载默认配置
config = load_hyperparameters()

# 修改配置
config.update({
    'input_size': 416,
    'batch_size': 16,
    'learning_rate': 0.001,
    'num_classes': 80,  # COCO数据集
})

# 保存配置
save_hyperparameters(config, 'config.json')
```

## 数据集支持

### VOC格式数据集

```python
from dataset import VOC_Detection_Set

dataset = VOC_Detection_Set(
    voc2012_jpeg_dir="./data/JPEGImages",
    voc2012_anno_dir="./data/Annotations", 
    class_file="./voc_classes.txt",
    input_size=416
)
```

### COCO格式数据集

```python
from dataset import COCO_Detection_Set

dataset = COCO_Detection_Set(
    imgs_path="./data/images",
    coco_json="./data/annotations.json",
    input_size=416
)
```

## 模型性能

### 网络规格

- **输入尺寸**: 416×416×3
- **输出尺度**: 13×13、26×26、52×52
- **锚框数量**: 每个位置3个锚框
- **检测类别**: 20类(VOC) / 80类(COCO)
- **推理速度**: ~30 FPS (GTX 1080Ti)

### 检测精度

- **mAP@0.5**: 根据训练数据集而定
- **小目标检测**: 52×52尺度专门处理
- **大目标检测**: 13×13尺度专门处理
- **中等目标**: 26×26尺度专门处理

## 高级功能

### 1. 混合精度训练

```python
# 启用自动混合精度
hyperparams['use_amp'] = True
```

### 2. 模型微调

```python
# 加载预训练权重
model.load_state_dict(torch.load('pretrained.pth'))

# 冻结骨干网络
for param in model.fpn.backbone.parameters():
    param.requires_grad = False
```

### 3. 数据增强

```python
# 支持多种数据增强
transforms = [
    'random_flip',
    'random_crop', 
    'color_jitter',
    'mixup',
    'mosaic'
]
```

### 4. 模型剪枝

```python
# 支持通道剪枝和结构化剪枝
from Utils import prune_model

pruned_model = prune_model(model, prune_ratio=0.3)
```

## 结果可视化

### 检测结果可视化

```python
from result_visualisation import ResultVisualizer

visualizer = ResultVisualizer()
vis_image = visualizer.visualize_detections(image, detections)
```

### 训练过程可视化

```python
# 支持训练损失、验证精度曲线
visualizer.plot_training_curves(train_losses, val_maps)
```

### 检测统计分析

```python
# 生成检测结果统计报告
report = visualizer.generate_detection_report(detections)
```

## 兼容性特性

- **自动格式转换**: 支持YOLOv1数据格式自动转换
- **向后兼容**: 保持与原有接口的兼容性
- **跨平台支持**: Windows、Linux、macOS
- **多GPU训练**: 支持DataParallel和DistributedDataParallel

## 配置参数

### 模型参数

```python
model_config = {
    'num_classes': 20,      # 检测类别数
    'input_size': 416,      # 输入图像尺寸
    'grid_sizes': [13, 26, 52],  # 检测网格尺寸
    'num_anchors': 3        # 每个位置锚框数
}
```

### 训练参数

```python
train_config = {
    'learning_rate': 0.001,  # 学习率
    'batch_size': 16,        # 批大小
    'epochs': 100,           # 训练轮数
    'weight_decay': 0.0005,  # 权重衰减
    'momentum': 0.9          # 动量
}
```

### 后处理参数

```python
postprocess_config = {
    'conf_threshold': 0.5,   # 置信度阈值
    'nms_threshold': 0.4,    # NMS IoU阈值  
    'max_detections': 100    # 最大检测数量
}
```

## 快速示例

```python
import torch
from NetModel import create_yolov3_model
from postprocess import postprocess_yolov3
from Utils import load_hyperparameters

# 1. 创建模型
model = create_yolov3_model(num_classes=20, input_size=416)

# 2. 加载权重
model.load_state_dict(torch.load('best_model.pth'))

# 3. 推理
with torch.no_grad():
    predictions = model(input_tensor)

# 4. 后处理
detections = postprocess_yolov3(predictions, conf_threshold=0.5)

print(f"检测到 {len(detections[0])} 个目标")
```

**YOLOv3 - 高性能实时目标检测系统**
