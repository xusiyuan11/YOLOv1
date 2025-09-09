import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional
import os
import json

# NPU支持 - PyTorch版本
try:
    import torch_npu
    NPU_AVAILABLE = True
    print("PyTorch NPU支持已加载")
except ImportError:
    NPU_AVAILABLE = False
    print("PyTorch NPU支持未安装，将使用CPU/GPU")


def get_best_device():
    """
    自动检测并返回最佳可用设备
    优先级: NPU > CUDA > CPU
    """
    if NPU_AVAILABLE and torch.npu.is_available():
        device_count = torch.npu.device_count()
        if device_count > 0:
            print(f"检测到 {device_count} 个NPU设备，使用 npu:0")
            return 'npu:0'
    
    if torch.cuda.is_available():
        print("检测到CUDA设备，使用 cuda:0")
        return 'cuda:0'
    
    print("使用CPU设备")
    return 'cpu'


def setup_npu_device(device_id=0):
    """
    设置NPU设备
    Args:
        device_id: NPU设备ID，默认为0
    Returns:
        torch.device: NPU设备对象
    """
    if not NPU_AVAILABLE:
        raise RuntimeError("NPU支持未安装，请安装torch_npu")
    
    if not torch.npu.is_available():
        raise RuntimeError("NPU设备不可用")
    
    device = torch.device(f'npu:{device_id}')
    torch.npu.set_device(device_id)
    
    print(f"已设置NPU设备: {device}")
    try:
        print(f"NPU设备名称: {torch.npu.get_device_name(device_id)}")
    except:
        print("无法获取NPU设备名称")
    
    return device


def load_hyperparameters(config_path: str = None) -> Dict:
    default_hyperparameters = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 100,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'input_size': 448,
        'grid_size': 64,
        'num_classes': 20,
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        'class_smooth_value': 0.01,
        'conf_threshold': 0.1,
        'nms_threshold': 0.5,
        'device': get_best_device()
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
        default_hyperparameters.update(custom_config)
    
    return default_hyperparameters


def save_hyperparameters(hyperparameters: Dict, save_path: str):
    """保存超参数配置"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparameters, f, indent=4, ensure_ascii=False)


def calculate_iou(box1: List[float], box2: List[float]) -> float:

    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 计算并集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> List[int]:

    if len(boxes) == 0:
        return []
    
    # 按置信度排序
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        # 选择置信度最高的框
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # 计算与其他框的 IoU
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = np.array([calculate_iou(current_box, box) for box in other_boxes])
        
        # 移除 IoU 大于阈值的框
        indices = indices[1:][ious <= threshold]
    
    return keep


def decode_yolo_output(predictions: torch.Tensor, 
                      conf_threshold: float = 0.1,
                      nms_threshold: float = 0.5,
                      input_size: int = 448,
                      grid_size: int = 7,
                      num_classes: int = 20) -> List[Dict]:
    batch_size = predictions.size(0)
    detections = []
    
    for b in range(batch_size):
        pred = predictions[b]  # (grid_size, grid_size, 30)
        
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 提取预测信息
                cell_pred = pred[i, j]
                
                # 两个边界框
                for box_idx in range(2):
                    start_idx = box_idx * 5
                    
                    # 边界框参数
                    x = (cell_pred[start_idx] + j) / grid_size
                    y = (cell_pred[start_idx + 1] + i) / grid_size
                    w = cell_pred[start_idx + 2]
                    h = cell_pred[start_idx + 3]
                    conf = torch.sigmoid(cell_pred[start_idx + 4])
                    
                    # 类别概率
                    class_probs = torch.softmax(cell_pred[10:], dim=0)
                    class_conf, class_id = torch.max(class_probs, dim=0)
                    
                    # 最终置信度
                    final_conf = conf * class_conf
                    
                    if final_conf > conf_threshold:
                        # 转换为像素坐标
                        x1 = (x - w/2) * input_size
                        y1 = (y - h/2) * input_size
                        x2 = (x + w/2) * input_size
                        y2 = (y + h/2) * input_size
                        
                        boxes.append([x1, y1, x2, y2])
                        scores.append(final_conf.item())
                        class_ids.append(class_id.item())
        
        # 应用 NMS
        if len(boxes) > 0:
            boxes = np.array(boxes)
            scores = np.array(scores)
            class_ids = np.array(class_ids)
            
            keep_indices = non_max_suppression(boxes, scores, nms_threshold)
            
            detection = {
                'boxes': boxes[keep_indices],
                'scores': scores[keep_indices],
                'class_ids': class_ids[keep_indices]
            }
        else:
            detection = {
                'boxes': np.array([]),
                'scores': np.array([]),
                'class_ids': np.array([])
            }
        
        detections.append(detection)
    
    return detections


def visualize_detections(image: np.ndarray, 
                        detection: Dict, 
                        class_names: List[str] = None,
                        conf_threshold: float = 0.5) -> np.ndarray:
    vis_image = image.copy()
    
    boxes = detection['boxes']
    scores = detection['scores']
    class_ids = detection['class_ids']
    
    if class_names is None:
        class_names = [f'class_{i}' for i in range(20)]
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score >= conf_threshold:
            x1, y1, x2, y2 = box.astype(int)
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f'{class_names[int(class_id)]}: {score:.2f}'
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return vis_image


def save_checkpoint(checkpoint_dict: dict, filepath: str):
    """保存模型检查点"""
    torch.save(checkpoint_dict, filepath)
    print(f"Checkpoint saved to {filepath}")


def save_simple_checkpoint(model: nn.Module, 
                          optimizer: torch.optim.Optimizer, 
                          epoch: int, 
                          loss: float, 
                          filepath: str):
    """保存简单模型检查点（向后兼容）"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    save_checkpoint(checkpoint, filepath)


def load_checkpoint(filepath: str):
    """加载模型检查点，返回完整的checkpoint字典"""
    checkpoint = torch.load(filepath, map_location='cpu')
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def load_simple_checkpoint(model: nn.Module, 
                          optimizer: torch.optim.Optimizer, 
                          filepath: str) -> Tuple[int, float]:
    """加载简单模型检查点（向后兼容）"""
    checkpoint = load_checkpoint(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint.get('loss', 0.0)
    print(f"Model loaded, epoch: {epoch}, loss: {loss:.4f}")
    return epoch, loss


def calculate_map(predictions: List[Dict], 
                 targets: List[Dict], 
                 num_classes: int = 20,
                 iou_threshold: float = 0.5) -> Dict:
    aps = []
    
    for class_id in range(num_classes):
        # 收集该类别的所有预测和真实标签
        pred_boxes = []
        pred_scores = []
        true_boxes = []
        
        for pred, target in zip(predictions, targets):
            # 预测框
            if len(pred['boxes']) > 0:
                class_mask = pred['class_ids'] == class_id
                pred_boxes.extend(pred['boxes'][class_mask])
                pred_scores.extend(pred['scores'][class_mask])
            
            # 真实框
            if len(target['boxes']) > 0:
                class_mask = target['class_ids'] == class_id
                true_boxes.extend(target['boxes'][class_mask])
        
        if len(true_boxes) == 0:
            continue
        
        # 计算 AP
        ap = calculate_ap(pred_boxes, pred_scores, true_boxes, iou_threshold)
        aps.append(ap)
    
    map_score = np.mean(aps) if aps else 0.0
    
    return {
        'mAP': map_score,
        'APs': aps,
        'num_classes': len(aps)
    }


def calculate_ap(pred_boxes: List, 
                pred_scores: List, 
                true_boxes: List, 
                iou_threshold: float = 0.5) -> float:
    """计算单个类别的 AP"""
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0.0
    
    # 按置信度排序
    indices = np.argsort(pred_scores)[::-1]
    pred_boxes = np.array(pred_boxes)[indices]
    pred_scores = np.array(pred_scores)[indices]
    true_boxes = np.array(true_boxes)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched = [False] * len(true_boxes)
    
    for i, pred_box in enumerate(pred_boxes):
        max_iou = 0
        max_idx = -1
        
        for j, true_box in enumerate(true_boxes):
            if not matched[j]:
                iou = calculate_iou(pred_box, true_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
        
        if max_iou >= iou_threshold:
            tp[i] = 1
            matched[max_idx] = True
        else:
            fp[i] = 1
    
    # 计算 precision 和 recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / len(true_boxes)
    
    # 计算 AP
    ap = 0
    for r in np.arange(0, 1.1, 0.1):
        p_max = np.max(precision[recall >= r]) if np.any(recall >= r) else 0
        ap += p_max / 11
    
    return ap


def create_class_names(dataset_type: str = 'voc') -> List[str]:
    """创建类别名称列表"""
    if dataset_type.lower() == 'voc':
        return [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    elif dataset_type.lower() == 'coco':
        # COCO 80类的简化版本（前20类）
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow'
        ]
    else:
        return [f'class_{i}' for i in range(20)]

