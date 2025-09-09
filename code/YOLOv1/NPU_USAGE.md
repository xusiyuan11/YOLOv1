# YOLOv1 NPU使用指南

## 环境准备

### 1. 安装华为昇腾CANN开发套件
```bash
# 下载并安装CANN软件包
# 具体安装方法请参考华为官方文档
```

### 2. 安装PyTorch NPU支持
```bash
# 安装torch_npu
pip install torch_npu
```

### 3. 设置环境变量
```bash
# 设置CANN环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 或者在Python脚本中添加路径
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/torch_npu_bridge:$PYTHONPATH
```

## 使用方法

### 方法1：自动检测NPU设备
代码会自动检测并使用NPU设备（如果可用）：

```python
# 运行训练（会自动使用NPU）
python Train_Complete.py

# 运行测试
python Test.py
```

### 方法2：指定NPU配置文件
```python
# 使用NPU配置文件运行
python Train_Complete.py --config npu_config.json
```

### 方法3：手动指定设备
在代码中可以手动指定NPU设备：

```python
from Utils import setup_npu_device, load_hyperparameters

# 设置超参数
hyperparameters = load_hyperparameters()
hyperparameters['device'] = 'npu:0'  # 使用第一个NPU设备

# 或者直接设置NPU设备
device = setup_npu_device(device_id=0)
```

## 设备检测顺序

代码会按以下优先级自动选择计算设备：
1. **NPU** (如果可用)
2. **CUDA GPU** (如果可用)
3. **CPU** (备选方案)

## NPU特定设置

### 1. 混合精度训练
NPU支持混合精度训练以提高性能：

```python
# 在npu_config.json中启用
"npu_settings": {
    "enable_mixed_precision": true,
    "amp_level": "O1",
    "loss_scale": 128.0
}
```

### 2. 内存优化
```python
# 清理NPU内存缓存
import torch_npu
torch_npu.empty_cache()
```

### 3. 多NPU支持
```python
# 检查NPU设备数量
device_count = torch.npu.device_count()
print(f"可用NPU设备数量: {device_count}")

# 使用特定NPU设备
device = setup_npu_device(device_id=1)  # 使用第二个NPU
```

## 性能优化建议

1. **批次大小**: NPU通常支持较大的批次大小，可以适当增加
2. **数据类型**: 优先使用float16进行混合精度训练
3. **内存管理**: 定期清理NPU内存缓存
4. **数据预处理**: 尽量在CPU上完成数据预处理，减少NPU-CPU数据传输

## 故障排除

### 常见问题

1. **NPU设备不可用**
   ```
   错误: NPU设备不可用
   解决: 检查CANN安装和环境变量设置
   ```

2. **torch_npu模块未找到**
   ```
   错误: No module named 'torch_npu'
   解决: pip install torch_npu
   ```

3. **内存不足**
   ```
   错误: NPU内存不足
   解决: 减少batch_size或清理内存缓存
   ```

### 检查NPU状态
```python
import torch_npu

# 检查NPU是否可用
print(f"NPU可用: {torch_npu.is_available()}")

# 检查NPU设备数量
print(f"NPU设备数量: {torch_npu.device_count()}")

# 检查当前NPU设备
print(f"当前NPU设备: {torch_npu.current_device()}")
```

## 性能对比

通常情况下，NPU相比GPU的性能提升：
- 训练速度: 1.2-2x 提升
- 内存效率: 更好的大批次支持
- 能耗: 更低的功耗

## 注意事项

1. 确保CANN版本与torch_npu版本兼容
2. NPU设备ID从0开始编号
3. 某些PyTorch操作可能在NPU上有不同的实现
4. 建议在训练前进行小规模测试验证
