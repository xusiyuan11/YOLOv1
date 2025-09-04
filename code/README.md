# YOLO复现项目

基于PyTorch实现的YOLOv1目标检测算法复现，支持两阶段训练：COCO分类预训练 + VOC检测微调。

## 项目特色

### 两阶段训练策略
1. **阶段1**: 使用COCO数据集中的分割目标进行分类预训练，训练整个卷积网络
2. **阶段2**: 使用VOC2007+2012数据集进行目标检测微调

### 项目结构

```
code/
├── main.py                   # 主程序入口
├── Train_Complete.py         # 完整两阶段训练控制器
├── Train_Classification.py  #  阶段1分类预训练
├── Train_Detection.py       #  阶段2检测训练
├── two_stage_config.json     # 两阶段训练配置文件
├── backbone.py               # DarkNet骨干网络（含轻量化版本）
├── NetModel.py              # YOLO网络模型
├── YOLOLoss.py              # YOLO损失函数
├── Utils.py                 # 工具函数集合
├── OPT.py                   # 优化器管理
├── dataset.py               # 数据集处理（包含COCO分割分类数据集）
├── Test.py                  # 测试模块
├── result_visualisation.py  # 结果可视化
├── voc_classes.txt          # VOC数据集类别文件
├── ui_main.py               # 用户界面
```

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


# 测试和可视化
python main.py test --checkpoint model.pth
python main.py visualize
python main.py demo
```

```

### 数据准备

确保你的数据目录结构如下：
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
        └── VOC2012/
            ├── JPEGImages/
            └── Annotations/
```

###  配置文件

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
    "voc_data_path": "../data/VOC2012",
    "use_voc2007": true,
    "use_voc2012": true,
    "train_val_split": 0.8,
    "class_file": "./voc_classes.txt",
    "epochs": 20,
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
git clone 
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


## 致谢

感谢以下开源项目和数据集：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [COCO Dataset](https://cocodataset.org/) - 目标检测数据集  
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) - 目标检测数据集
- [YOLOv1 Paper](https://arxiv.org/abs/1506.02640) - 原始论文

---



