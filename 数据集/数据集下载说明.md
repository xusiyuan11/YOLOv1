# 数据集下载说明

##  概述

本YOLO复现项目支持VOC和COCO数据集进行目标检测训练。
由于数据集过大不便上传到库，本文档将详细介绍如何下载和配置这些数据集。

##  所需数据集

### VOC2007数据集
- **数据集名称**: PASCAL Visual Object Classes Challenge 2007 (VOC2007)
- **用途**: 目标检测训练数据补充
- **类别数**: 20个类别
- **格式**: 图像 + XML标注文件

### VOC2012数据集
- **数据集名称**: PASCAL Visual Object Classes Challenge 2012 (VOC2012)
- **用途**: 目标检测训练和验证
- **类别数**: 20个类别
- **格式**: 图像 + XML标注文件

### COCO2017数据集
- **数据集名称**: Microsoft Common Objects in Context 2017 (COCO2017)
- **用途**: 大规模目标检测训练和验证
- **类别数**: 80个类别
- **格式**: 图像 + JSON标注文件
- **优势**: 数据量大、标注质量高、类别丰富

> **说明**: 
> - VOC数据集：通常将VOC2007+VOC2012的训练/验证集合并作为完整训练集，使用VOC2007的测试集进行最终评估
> - COCO数据集：使用train2017作为训练集，val2017作为验证集，test2017作为测试集

## 下载方式

### 方式一：官方下载（推荐）

#### VOC2007数据集

##### 1. VOC2007 训练/验证数据
```bash
# 下载地址
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

# 文件信息
文件名: VOCtrainval_06-Nov-2007.tar
大小: ~460MB
包含: 训练集 + 验证集图像和标注
```

##### 2. VOC2007 测试数据
```bash
# 下载地址
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# 文件信息
文件名: VOCtest_06-Nov-2007.tar
大小: ~430MB
包含: 测试集图像和标注
```

#### VOC2012数据集

##### 1. VOC2012 训练/验证数据
```bash
# 下载地址
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# 文件信息
文件名: VOCtrainval_11-May-2012.tar
大小: ~2GB
包含: 训练集 + 验证集图像和标注
```

##### 2. VOC2012 测试数据（可选）
```bash
# 下载地址
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtest_11-May-2012.tar

# 文件信息
文件名: VOCtest_11-May-2012.tar  
大小: ~1.9GB
包含: 测试集图像（无标注）
```

#### COCO2017数据集

##### 1. COCO2017 训练集图像
```bash
# 下载地址
http://images.cocodataset.org/zips/train2017.zip

# 文件信息
文件名: train2017.zip
大小: ~19GB
包含: 118,287张训练图像
```

##### 2. COCO2017 验证集图像
```bash
# 下载地址
http://images.cocodataset.org/zips/val2017.zip

# 文件信息
文件名: val2017.zip
大小: ~1GB
包含: 5,000张验证图像
```

##### 3. COCO2017 测试集图像（可选）
```bash
# 下载地址
http://images.cocodataset.org/zips/test2017.zip

# 文件信息
文件名: test2017.zip
大小: ~6GB
包含: 40,670张测试图像
```

##### 4. COCO2017 标注文件
```bash
# 下载地址
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# 文件信息
文件名: annotations_trainval2017.zip
大小: ~241MB
包含: 训练集和验证集的标注文件
```


## 目录结构

下载并解压后，确保目录结构如下：

```
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
        └── VOC2012/
            ├── JPEGImages/
            └── Annotations/
```

##  安装步骤

### 步骤1：创建数据目录
```bash
mkdir -p data/VOC2012
cd data/VOC2012
```

### 步骤2：下载数据集
```bash
# 使用wget下载（Linux/Mac）
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# 或使用curl下载
curl -O http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

### 步骤3：解压文件
```bash
tar -xvf VOCtrainval_11-May-2012.tar
```

### 步骤4：验证目录结构
```bash
ls -la VOCdevkit/VOC2012/
```

应该看到：`Annotations`, `ImageSets`, `JPEGImages`, `SegmentationClass`, `SegmentationObject`

## ⚙️ 配置文件

确保项目配置文件指向正确的数据路径：

### two_stage_config.json
```json
{
  "voc_config": {
    "voc2012_jpeg_dir": "../data/VOC2012/VOCdevkit/VOC2012/JPEGImages",
    "voc2012_anno_dir": "../data/VOC2012/VOCdevkit/VOC2012/Annotations",
    "class_file": "./voc_classes.txt",
    "batch_size": 8,
    "num_classes": 20
  }
}
```

### 训练脚本中的配置
```python
voc_config = {
    'voc2012_jpeg_dir': '../data/VOC2012/VOCdevkit/VOC2012/JPEGImages',
    'voc2012_anno_dir': '../data/VOC2012/VOCdevkit/VOC2012/Annotations',
    'batch_size': 8,
    'num_classes': 20
}
```

##  VOC2012类别

数据集包含以下20个类别：

| 编号 | 类别名称 | 中文名称 |
|------|----------|----------|
| 0 | aeroplane | 飞机 |
| 1 | bicycle | 自行车 |
| 2 | bird | 鸟类 |
| 3 | boat | 船只 |
| 4 | bottle | 瓶子 |
| 5 | bus | 公交车 |
| 6 | car | 汽车 |
| 7 | cat | 猫 |
| 8 | chair | 椅子 |
| 9 | cow | 牛 |
| 10 | diningtable | 餐桌 |
| 11 | dog | 狗 |
| 12 | horse | 马 |
| 13 | motorbike | 摩托车 |
| 14 | person | 人 |
| 15 | pottedplant | 盆栽植物 |
| 16 | sheep | 羊 |
| 17 | sofa | 沙发 |
| 18 | train | 火车 |
| 19 | tvmonitor | 电视/显示器 |

##  数据集统计

- **训练集**: 5,717张图像
- **验证集**: 5,823张图像  
- **测试集**: 10,991张图像（无标注）
- **总计**: 22,531张图像
- **平均每张图像目标数**: 2.4个
- **图像尺寸**: 变化（最小约200x150，最大约500x400）

##  验证安装

运行以下Python代码验证数据集是否正确安装：

```python
import os
from dataset import VOC_Detection_Set

# 验证路径
jpeg_dir = '../data/VOC2012/VOCdevkit/VOC2012/JPEGImages'
anno_dir = '../data/VOC2012/VOCdevkit/VOC2012/Annotations'

print(f"图像目录存在: {os.path.exists(jpeg_dir)}")
print(f"标注目录存在: {os.path.exists(anno_dir)}")

if os.path.exists(jpeg_dir) and os.path.exists(anno_dir):
    print(f"图像数量: {len(os.listdir(jpeg_dir))}")
    print(f"标注数量: {len(os.listdir(anno_dir))}")
    
    # 测试数据加载
    dataset = VOC_Detection_Set(
        voc2012_jpeg_dir=jpeg_dir,
        voc2012_anno_dir=anno_dir,
        class_file='./voc_classes.txt',
        input_size=448,
        grid_size=7
    )
    print(f"数据集大小: {len(dataset)}")
    print("✅ 数据集配置正确！")
else:
    print("❌ 数据集路径不正确，请检查下载和解压")
```

##  常见问题

### Q1: 下载速度慢怎么办？
**A**: 使用国内镜像或百度网盘下载，或使用下载工具（如迅雷）。

### Q2: 解压后目录结构不对？
**A**: 确保解压到正确位置，应该有`VOCdevkit/VOC2012`这样的嵌套结构。

### Q3: 路径配置错误？
**A**: 检查配置文件中的路径是否与实际目录结构匹配，注意相对路径和绝对路径。

### Q4: 内存不足？
**A**: 可以修改`batch_size`为更小的值，如4或2。

### Q5: 权限问题？
**A**: 确保对数据目录有读写权限。


## 支持

如果在数据集下载和配置过程中遇到问题，请：

1. 检查网络连接
2. 验证下载文件的完整性
3. 确认目录权限
4. 参考错误日志

---

**注意**: 请确保遵守VOC数据集的使用许可协议，仅用于学术研究和教育目的。


