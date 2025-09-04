# YOLO复现项目

基于PyTorch实现的YOLOv1目标检测算法复现，支持两阶段训练：COCO分类预训练 + VOC检测微调。

## 项目特色

### 两阶段训练策略
1. **阶段1**: 使用COCO数据集中的分割目标进行分类预训练，训练整个卷积网络
2. **阶段2**: 使用VOC2007+2012数据集进行目标检测微调

### 项目结构

```
code/
├── main.py                   # 主程序入口（兼容性保留）
├── Train_Complete.py         # 新：完整两阶段训练控制器
├── Train_Classification.py  #  新：阶段1分类预训练
├── Train_Detection.py       #  新：阶段2检测训练
├── Train.py                  # 原始两阶段训练脚本（保留）
├── two_stage_config.json     # 两阶段训练配置文件
├── backbone.py               # DarkNet骨干网络（含轻量化版本）
├── NetModel.py              #  优化：YOLO网络模型（分离式架构）
├── YOLOLoss.py              # YOLO损失函数
├── Utils.py                 # 工具函数集合
├── OPT.py                   # 优化器管理
├── dataset.py               # 数据集处理（包含COCO分割分类数据集）
├── Test.py                  # 测试模块
├── result_visualisation.py  # 结果可视化
├── voc_classes.txt          # VOC数据集类别文件
├── ui_main.py               # 用户界面
```

## 核心模块说明

### 分离式训练系统

#### Train_Complete.py - 训练流程控制器
- **交互式训练**: 提供三种训练模式选择
- **完整流程**: 自动执行两阶段训练
- **单阶段调试**: 支持单独训练某个阶段
- **依赖检查**: 自动检查文件依赖关系

#### Train_Classification.py - 阶段1分类预训练
- **BackboneClassificationTrainer类**: 专门的分类训练器
- **数据处理**: COCO分割目标分类数据集
- **模型输出**: 训练好的backbone权重 (230万参数)
- **验证机制**: 训练集/验证集分离，实时监控精度
- **断点续训**: 支持从checkpoint恢复训练

#### Train_Detection.py - 阶段2检测训练  
- **DetectionTrainer类**: 专门的检测训练器
- **权重加载**: 自动加载预训练backbone
- **训练策略**: 冻结+解冻的渐进式训练
- **参数分组**: backbone和检测头使用不同学习率
- **模型输出**: 完整检测模型 (348万参数，减少94.5%)

###  优化的网络架构

#### NetModel.py - 分离式模型设计
- **ClassificationModel**: 阶段1分类模型
  - 输入: (batch, 3, 448, 448)
  - 输出: (batch, num_classes) 分类概率
  - 参数量: 230万 (仅backbone)
- **DetectionModel**: 阶段2检测模型  
  - 输入: (batch, 3, 448, 448)
  - 输出: (batch, 7, 7, 30) YOLO格式
  - 参数量: 348万 (backbone + 轻量检测头)
- **权重迁移**: 无缝的backbone权重迁移
- **训练模式**: 支持backbone冻结/解冻切换

#### backbone.py - 轻量化骨干网络
- **EfficientDarkNet**: 优化版DarkNet
  - 使用深度可分离卷积
  - 参数量减少60-70%
  - 推理速度提升2-3倍
- **原始DarkNet**: 保留经典实现
- **性能对比**: 自带benchmark工具

### 📊 原始训练系统（兼容保留）

#### Train.py - 两阶段训练核心
- **TwoStageYOLOTrainer类**: 完整的两阶段训练管理器
- **阶段1**: COCO分割目标分类预训练
- **阶段2**: VOC目标检测微调
- **配置文件支持**: 通过JSON配置文件灵活设置参数

## 使用方法

###  快速开始

```bash
# 方式1: 完整两阶段训练
python Train_Complete.py
# 选择 1 - 完整两阶段训练

# 方式2: 分离式训练（适合调试）
python Train_Classification.py  # 阶段1：分类预训练
python Train_Detection.py       # 阶段2：检测训练
# 选择 2 - 仅分类预训练
# 选择 3 - 仅检测训练


# 兼容性：原始训练方式
python main.py train

# 测试和可视化
python main.py test --checkpoint model.pth
python main.py visualize
python main.py demo
```

```

### 数据准备

确保你的数据目录结构如下：
data/
├── COCO/
│   ├── train2017/          # COCO训练图像
│   ├── val2017/            # COCO验证图像
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
└── VOC2012/
    └── VOCdevkit/
        ├── VOC2007/
        │   ├── JPEGImages/
        │   └── Annotations/
        ├── VOC2012/
        │   ├── JPEGImages/
        │   └── Annotations/

```

### ⚙️ 配置文件

项目使用 `two_stage_config.json` 配置文件来管理训练参数：

```json
{
  "stage1_coco_classification": {
    "description": "阶段1: COCO分割目标分类预训练",
    "coco_train_images": "../data/COCO/train2017",
    "coco_train_annotations": "../data/COCO/annotations/instances_train2017.json",
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 0.001,
    "min_object_area": 1000,
    "max_objects_per_image": 10
  },
  "stage2_voc_detection": {
    "description": "阶段2: VOC目标检测微调",
    "voc2007_train_images": "../data/VOC2012/VOCdevkit/VOC2007/JPEGImages",
    "voc2007_train_annotations": "../data/VOC2012/VOCdevkit/VOC2007/Annotations",
    "voc2012_train_images": "../data/VOC2012/VOCdevkit/VOC2012/JPEGImages",
    "voc2012_train_annotations": "../data/VOC2012/VOCdevkit/VOC2012/Annotations",
    "epochs": 80,
    "batch_size": 8,
    "learning_rate": 0.0001
  },
  "model_config": {
    "input_size": 448,
    "grid_size": 64,
    "num_classes_coco": 80,
    "num_classes_voc": 20,
    "lambda_coord": 5.0,
    "lambda_noobj": 0.5
  }
}
```



## 评估指标

- **阶段1**: 分类准确率 (Accuracy)
- **阶段2**: 
  - **mAP**: 平均精度均值
  - **IoU**: 交并比
  - **检测损失**: 综合检测损失值

## 系统要求

### 💻 硬件要求
- **GPU**: NVIDIA GPU (推荐GTX 4060或更高)
- **内存**: 8GB+ RAM
- **存储**: 50GB+ 可用空间 (数据集+模型)

### 依赖库
```bash
torch >= 1.8.0
torchvision >= 0.9.0
opencv-python >= 4.5.0
numpy >= 1.19.0
tqdm >= 4.60.0
pycocotools >= 2.0.2
```

### 安装说明
```bash
# 1. 克隆项目
git clone <repository-url>
cd YOLO复现

# 2. 安装依赖
pip install torch torchvision opencv-python numpy tqdm pycocotools

# 3. 准备数据集
# 下载COCO和VOC数据集到data/目录

# 4. 开始训练（推荐新式分离训练）
python Train_Complete.py
# 或者：python Train_Classification.py

# 5. 兼容性训练
python main.py train
```

## 使用建议

### 🆕 新手推荐流程
1. **首次使用**: `python Train_Complete.py` → 选择1（完整训练）
2. **调试分类**: `python Train_Complete.py` → 选择2（仅分类）  
3. **调试检测**: `python Train_Complete.py` → 选择3（仅检测）
4. **高级调试**: 直接修改 `Train_Classification.py` 和 `Train_Detection.py`

### 参数调优建议
- **显存不足**: 减小batch_size（分类16→8，检测8→4）
- **训练加速**: 使用EfficientDarkNet（默认开启）
- **精度优化**: 增加训练轮数，调整学习率
- **调试便利**: 使用分离式训练，单独优化各阶段

### 输出文件管理
```
checkpoints/
├── classification/           # 分类训练输出
│   ├── best_classification_model.pth
│   ├── trained_backbone.pth  # ← 用于阶段2
│   └── classification_checkpoint_*.pth
└── detection/               # 检测训练输出
    ├── best_detection_model.pth  # ← 最终模型
    └── detection_checkpoint_*.pth
```

## 注意事项

### 训练前准备
1. **数据集路径**: 确保配置文件中的数据集路径正确
2. **GPU内存**: 根据GPU显存调整batch_size (推荐: 16GB显存用batch_size=16)
3. **磁盘空间**: 确保有足够空间存储模型检查点
4. **权限检查**: 确保程序有读写数据目录和输出目录的权限

### 性能优化
- **批量大小**: 根据GPU显存调整，显存不足时减小batch_size
- **学习率**: 阶段1使用0.001，阶段2使用0.0001
- **数据加载**: 使用多进程加载数据 (num_workers=4)
- **检查点保存**: 每20个epoch自动保存，防止训练中断

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 解决方案：减小batch_size
   # 在配置文件中修改batch_size为更小值
   ```

2. **数据集加载失败**
   ```bash
   # 检查数据路径是否正确
   # 检查图像和标注文件是否对应
   # 确保COCO annotations格式正确
   ```

3. **训练不收敛**
   ```bash
   # 检查学习率设置
   # 确保数据增强适度
   # 检查标注质量
   ```

4. **导入模块错误**
   ```bash
   # 确保所有依赖库已安装
   pip install -r requirements.txt
   ```



### 开发规范
- 代码风格：遵循PEP8规范
- 注释：中文注释，详细说明函数功能
- 测试：添加必要的测试用例
- 文档：更新相关文档


## 🙏 致谢

感谢以下开源项目和数据集：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [COCO Dataset](https://cocodataset.org/) - 目标检测数据集  
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) - 目标检测数据集
- [YOLOv1 Paper](https://arxiv.org/abs/1506.02640) - 原始论文

---
