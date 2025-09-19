# YOLO代码复现框架 - 项目源代码

## 代码概览

本文件夹包含了完整的YOLO算法复现代码实现，涵盖了从经典YOLOv1到现代SwinYOLO的多个版本。每个版本都提供了完整的训练、测试和评估功能。

## 代码结构

### 本地代码仓库

```
项目源代码/
├── 本地代码仓库/
│   ├── YOLOv1/           # YOLO第一版实现
│   ├── YOLOv3/           # YOLO第三版实现
│   ├── SwinYOLO/         # 基于Swin Transformer的YOLO
│   └── ui/               # 图形用户界面模块
```

## 各版本详细说明

### YOLOv1 - 经典目标检测算法

**核心文件结构：**

```
YOLOv1/
├── NetModel.py           # 网络模型定义
├── backbone.py           # 骨干网络实现
├── dataset.py           # 数据集处理
├── YOLOLoss.py          # 损失函数实现
├── Train_Classification.py  # 分类训练脚本
├── Train_Detection.py   # 检测训练脚本
├── Train_Complete.py    # 完整训练流程
├── Test.py              # 测试脚本
├── main.py              # 主程序入口
├── Utils.py             # 工具函数
├── visualization.py     # 可视化工具
├── result_visualisation.py  # 结果可视化
├── OPT.py               # 优化器配置
├── voc_classes.txt      # VOC数据集类别
└── checkpoints/         # 模型权重
```

**主要特性：**

- 经典的7×7网格检测方案
- 端到端训练架构
- 完整的分类+检测训练流程
- VOC数据集支持

### YOLOv3 - 改进版目标检测

**核心文件结构：**

```
YOLOv3/
├── NetModel.py           # 网络模型定义
├── backbone.py           # DarkNet-53骨干网络
├── anchors.py           # 锚框处理
├── postprocess.py       # 后处理模块
├── YOLOLoss.py          # 改进的损失函数
├── compatibility_fixes.py  # 兼容性修复
├── Train_Classification.py  # 分类训练
├── Train_Detection.py   # 检测训练
├── Train_Complete.py    # 完整训练
├── Test.py              # 测试评估
├── dataset.py           # 数据集处理
├── Visualization.py     # 可视化工具
├── result_visualisation.py  # 结果展示
├── main.py              # 主程序
├── Utils.py             # 辅助工具
├── OPT.py               # 优化配置
├── voc_classes.txt      # 类别定义
├── checkpoints/         # 模型权重

```

**主要特性：**

- 多尺度特征金字塔检测
- 先验锚框机制
- 改进的损失函数设计
- 更强的特征提取能力
- 完善的后处理流程

### SwinYOLO - 现代化Transformer架构

**核心文件结构：**

```
SwinYOLO/
├── SwinYOLO.py          # 主模型实现
├── backbone.py          # Swin Transformer骨干网络
├── train_swin_yolo.py   # 训练脚本
├── dataset.py           # 数据处理
├── evaluation.py        # 模型评估
├── visualization.py     # 可视化工具
├── Utils.py             # 工具函数
├── voc_classes.txt      # 类别定义
└── checkpoints/         # 模型权重
```

**主要特性：**

- 集成Swin Transformer架构
- 先进的注意力机制
- 现代化的训练策略
- 优化的特征融合方法

### UI模块 - 图形用户界面

**核心文件结构：**

```
ui/
├── main.py              # 主启动程序
├── aiui.py              # UI界面设置类
├── aiwindow.py          # 主窗口逻辑
├── aicamera.py          # 摄像头处理模块
├── aiapplication.py     # 应用程序逻辑
└── __pycache__/         # Python缓存文件
```

**主要特性：**

- 基于PyQt5的现代化图形界面
- 支持实时摄像头目标检测
- 集成多个YOLO模型选择
- 可视化检测结果展示
- 用户友好的参数调节界面
- 支持图片和视频文件检测

## 环境配置

### 基础依赖

```bash
# Python环境要求
Python >= 3.7

# 核心依赖包
torch>=1.10.0
torchvision>=0.11.0
timm>=0.6.0            
opencv-python>=4.5.0
numpy>=1.19.0
tqdm
einops  
matplotlib >= 3.3.0
PIL >= 8.0.0

# UI界面依赖
PyQt5 >= 5.15.0
```

### 数据集准备

- **支持格式**：VOC数据集格式
- **类别文件**：各版本都包含 `voc_classes.txt`
- **数据路径**：请根据代码中的路径配置调整

## 快速开始

### YOLOv1 使用示例

```bash
cd YOLOv1/

# 完整训练（分类+检测）
python Train_Complete.py

# 单独分类训练
python Train_Classification.py

# 单独检测训练
python Train_Detection.py

# 模型测试
python Test.py

# 主程序运行
python main.py
```

### YOLOv3 使用示例

```bash
cd YOLOv3/

# 完整训练流程
python Train_Complete.py

# 分阶段训练
python Train_Classification.py  # 先训练分类
python Train_Detection.py       # 再训练检测

# 测试评估
python Test.py

# 可视化结果
python Visualization.py
```

### SwinYOLO 使用示例

```bash
cd SwinYOLO/

# 训练SwinYOLO模型
python train_swin_yolo.py

# 模型评估
python evaluation.py

# 结果可视化
python visualization.py
```

### UI界面使用示例

```bash
cd ui/

# 启动图形界面程序
python main.py

# 程序功能：
# - 选择YOLO模型版本（YOLOv1/YOLOv3/SwinYOLO）
# - 实时摄像头检测
# - 图片文件检测
# - 视频文件检测
# - 参数调节和结果可视化
```

## 相关资源

- **项目文档**：`../项目文档/` 查看详细设计文档
- **技术论文**：参考各YOLO版本的原始论文
- **在线教程**：建议配合相关技术博客学习
