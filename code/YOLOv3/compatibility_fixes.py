"""
YOLOv3兼容性修复
提供数据格式转换和接口适配函数
"""
import torch
import numpy as np
from typing import List, Dict, Tuple


def convert_yolov1_targets_to_yolov3(yolov1_targets, max_objects=50):
    """
    将YOLOv1数据集格式转换为YOLOv3格式
    Args:
        yolov1_targets: [gt, mask_pos, mask_neg] 格式
        max_objects: 最大目标数量
    Returns:
        yolov3_targets: (max_objects, 5) 格式 [x1, y1, x2, y2, class_id]
    """
    if isinstance(yolov1_targets, list) and len(yolov1_targets) == 3:
        gt, mask_pos, mask_neg = yolov1_targets
    else:
        # 如果已经是简单格式，直接返回
        return yolov1_targets
    
    # 初始化YOLOv3格式目标
    batch_size = gt.shape[0] if len(gt.shape) > 2 else 1
    if batch_size == 1 and len(gt.shape) == 2:
        gt = gt.unsqueeze(0)
        mask_pos = mask_pos.unsqueeze(0)
    
    yolov3_targets = []
    
    for b in range(batch_size):
        targets = torch.full((max_objects, 5), -1, dtype=torch.float32)
        
        # 从网格编码中提取边界框
        gt_batch = gt[b] if batch_size > 1 else gt
        mask_batch = mask_pos[b] if batch_size > 1 else mask_pos
        
        # 找到有目标的网格
        if len(mask_batch.shape) == 3:
            mask_batch = mask_batch.squeeze(-1)
        
        obj_indices = torch.nonzero(mask_batch, as_tuple=False)
        
        obj_count = 0
        for idx in obj_indices:
            if obj_count >= max_objects:
                break
                
            i, j = idx[0].item(), idx[1].item()
            
            # 提取边界框信息（假设格式为 [tx, ty, tw, th, conf, ...classes...]）
            cell_data = gt_batch[i, j]
            
            # 解码坐标（简化版本）
            grid_size = gt_batch.shape[0]  # 假设是方形网格
            
            tx, ty, tw, th = cell_data[0:4]
            
            # 转换为绝对坐标 (0-1范围)
            cx = (j + tx) / grid_size
            cy = (i + ty) / grid_size
            w = tw
            h = th
            
            # 转换为边界框格式
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            
            # 获取类别ID（假设在第5位之后）
            if len(cell_data) > 5:
                class_probs = cell_data[5:25]  # 假设20个类别
                class_id = torch.argmax(class_probs).item()
            else:
                class_id = 0
            
            targets[obj_count] = torch.tensor([x1, y1, x2, y2, class_id])
            obj_count += 1
        
        yolov3_targets.append(targets)
    
    if len(yolov3_targets) == 1:
        return yolov3_targets[0]
    
    return torch.stack(yolov3_targets)


def create_yolov3_compatible_dataloader(original_dataloader):
    """
    创建YOLOv3兼容的数据加载器包装器
    """
    class YOLOv3DataLoaderWrapper:
        def __init__(self, original_loader):
            self.original_loader = original_loader
            self.dataset = original_loader.dataset
        
        def __iter__(self):
            for batch in self.original_loader:
                if len(batch) == 2:
                    images, targets = batch
                    # 转换目标格式
                    if isinstance(targets, list) and len(targets) > 0:
                        if isinstance(targets[0], list) and len(targets[0]) == 3:
                            # 批次中每个样本都是 [gt, mask_pos, mask_neg] 格式
                            converted_targets = []
                            for target in targets:
                                converted = convert_yolov1_targets_to_yolov3(target)
                                converted_targets.append(converted)
                            targets = torch.stack(converted_targets)
                        elif torch.is_tensor(targets) and len(targets.shape) > 2:
                            # 整个批次是一个张量
                            targets = convert_yolov1_targets_to_yolov3(targets)
                    
                    yield images, targets
                else:
                    yield batch
        
        def __len__(self):
            return len(self.original_loader)
    
    return YOLOv3DataLoaderWrapper(original_dataloader)


def update_hyperparameters_for_yolov3(hyperparams):
    """
    更新超参数以适配YOLOv3
    """
    yolov3_updates = {
        'input_size': 416,  # YOLOv3标准输入尺寸
        'grid_sizes': [13, 26, 52],  # 多尺度网格
        'num_anchors': 3,  # 每个位置3个锚框
        'anchor_scales': [
            [(116, 90), (156, 198), (373, 326)],  # 13x13
            [(30, 61), (62, 45), (59, 119)],      # 26x26  
            [(10, 13), (16, 30), (33, 23)]        # 52x52
        ]
    }
    
    # 更新现有超参数
    hyperparams.update(yolov3_updates)
    
    # 调整相关参数
    if 'grid_size' in hyperparams:
        # YOLOv1的grid_size参数在YOLOv3中不再使用
        hyperparams['legacy_grid_size'] = hyperparams.pop('grid_size')
    
    return hyperparams


def yolov3_decode_wrapper(predictions, **kwargs):
    """
    YOLOv3解码函数的包装器，替代Utils.decode_yolo_output
    """
    try:
        from postprocess import postprocess_yolov3
        
        # 提取参数
        conf_threshold = kwargs.get('conf_threshold', 0.5)
        nms_threshold = kwargs.get('nms_threshold', 0.4)
        input_shape = kwargs.get('input_shape', None)
        
        # 使用YOLOv3后处理
        detections = postprocess_yolov3(
            predictions,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_shape=input_shape
        )
        
        # 转换为Utils.decode_yolo_output的返回格式
        formatted_detections = []
        for detection in detections:
            if len(detection) > 0:
                formatted_detection = {
                    'boxes': detection[:, :4].numpy(),
                    'scores': detection[:, 4].numpy(),
                    'class_ids': detection[:, 5].numpy().astype(int)
                }
            else:
                formatted_detection = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'class_ids': np.array([])
                }
            formatted_detections.append(formatted_detection)
        
        return formatted_detections
        
    except ImportError:
        print("警告: 无法导入YOLOv3后处理模块，使用原始Utils函数")
        from Utils import decode_yolo_output
        return decode_yolo_output(predictions, **kwargs)


# 便捷的猴子补丁函数
def apply_yolov3_compatibility_patches():
    """
    应用YOLOv3兼容性补丁
    这个函数会修改Utils模块以支持YOLOv3
    """
    try:
        import Utils
        
        # 保存原始函数
        Utils._original_decode_yolo_output = Utils.decode_yolo_output
        
        # 替换为YOLOv3兼容版本
        Utils.decode_yolo_output = yolov3_decode_wrapper
        
        print("✅ YOLOv3兼容性补丁已应用")
        
    except ImportError:
        print("⚠️ 无法导入Utils模块，跳过兼容性补丁")


def restore_original_functions():
    """
    恢复原始函数（如果需要的话）
    """
    try:
        import Utils
        if hasattr(Utils, '_original_decode_yolo_output'):
            Utils.decode_yolo_output = Utils._original_decode_yolo_output
            delattr(Utils, '_original_decode_yolo_output')
            print("✅ 已恢复原始函数")
    except:
        pass


# 使用示例
if __name__ == "__main__":
    print("YOLOv3兼容性修复工具")
    print("=" * 40)
    
    # 应用兼容性补丁
    apply_yolov3_compatibility_patches()
    
    # 测试超参数更新
    from Utils import load_hyperparameters
    hyperparams = load_hyperparameters()
    updated_hyperparams = update_hyperparameters_for_yolov3(hyperparams)
    
    print("更新后的超参数:")
    for key, value in updated_hyperparams.items():
        if key.startswith(('input_size', 'grid_sizes', 'num_anchors')):
            print(f"  {key}: {value}")
    
    print("\n兼容性修复完成！现在可以使用YOLOv3了。")

