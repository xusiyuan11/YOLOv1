import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Optional
import os
import json


def load_hyperparameters(config_path: str = None) -> Dict:
    default_hyperparameters = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 100,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'input_size': 416,  # YOLOv3标准输入尺寸
        'grid_size': 64,    # 保留向后兼容性
        'grid_sizes': [13, 26, 52],  # YOLOv3多尺度网格
        'num_classes': 20,
        'num_anchors': 3,   # YOLOv3每个位置3个锚框
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        'class_smooth_value': 0.01,
        'conf_threshold': 0.5,  # YOLOv3推荐阈值
        'nms_threshold': 0.4,   # YOLOv3推荐阈值
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
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
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
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
                if np.any(class_mask):  # 确保有匹配的类别
                    selected_boxes = pred['boxes'][class_mask]
                    selected_scores = pred['scores'][class_mask]
                    
                    # 转换为列表以便extend
                    if selected_boxes.ndim == 1:
                        # 单个检测框
                        pred_boxes.append(selected_boxes)
                        pred_scores.append(selected_scores.item() if hasattr(selected_scores, 'item') else selected_scores)
                    else:
                        # 多个检测框
                        pred_boxes.extend(selected_boxes.tolist())
                        pred_scores.extend(selected_scores.tolist())
            
            # 真实框
            if len(target['boxes']) > 0:
                class_mask = target['class_ids'] == class_id
                if np.any(class_mask):  # 确保有匹配的类别
                    selected_true_boxes = target['boxes'][class_mask]
                    
                    # 转换为列表以便extend
                    if selected_true_boxes.ndim == 1:
                        # 单个真实框
                        true_boxes.append(selected_true_boxes)
                    else:
                        # 多个真实框
                        true_boxes.extend(selected_true_boxes.tolist())
        
        if len(true_boxes) == 0:
            continue
        
        # 计算 AP
        ap = calculate_ap(pred_boxes, pred_scores, true_boxes, iou_threshold)
        aps.append(ap)
    
    map_score = np.mean(aps) if aps else 0.0
    
    if debug:
        print(f"\n📊 mAP计算结果:")
        print(f"   有效类别数: {len(aps)}")
        print(f"   mAP: {map_score:.4f}")
        if len(aps) > 0:
            print(f"   最高AP: {max(aps):.4f}")
            print(f"   最低AP: {min(aps):.4f}")
            print(f"   前5类AP: {[round(ap, 4) for ap in aps[:5]]}")
    
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


def test_map_calculation():
    """测试mAP计算功能"""
    print("🧪 开始测试mAP计算功能...")
    
    # 创建虚拟的预测和真实数据
    predictions = []
    targets = []
    
    # 样本1: 正确预测
    predictions.append({
        'boxes': np.array([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]),
        'scores': np.array([0.9, 0.8]),
        'class_ids': np.array([0, 1])
    })
    targets.append({
        'boxes': np.array([[0.1, 0.1, 0.5, 0.5], [0.6, 0.6, 0.9, 0.9]]),
        'class_ids': np.array([0, 1])
    })
    
    # 样本2: 部分正确预测
    predictions.append({
        'boxes': np.array([[0.2, 0.2, 0.4, 0.4]]),
        'scores': np.array([0.7]),
        'class_ids': np.array([0])
    })
    targets.append({
        'boxes': np.array([[0.1, 0.1, 0.3, 0.3], [0.7, 0.7, 0.9, 0.9]]),
        'class_ids': np.array([0, 2])
    })
    
    # 样本3: 错误预测
    predictions.append({
        'boxes': np.array([[0.8, 0.8, 0.95, 0.95]]),
        'scores': np.array([0.6]),
        'class_ids': np.array([3])
    })
    targets.append({
        'boxes': np.array([[0.1, 0.1, 0.2, 0.2]]),
        'class_ids': np.array([1])
    })
    
    # 样本4: 空预测
    predictions.append({
        'boxes': np.array([]),
        'scores': np.array([]),
        'class_ids': np.array([])
    })
    targets.append({
        'boxes': np.array([[0.3, 0.3, 0.6, 0.6]]),
        'class_ids': np.array([2])
    })
    
    print(f"✅ 创建了{len(predictions)}个测试样本")
    
    # 测试基本mAP计算
    print("\n📊 测试1: 基本mAP计算")
    result = calculate_map(predictions, targets, num_classes=5, debug=True)
    
    # 测试单个样本
    print("\n📊 测试2: 单个样本测试")
    single_pred = [predictions[0]]
    single_target = [targets[0]]
    result2 = calculate_map(single_pred, single_target, num_classes=5, debug=True)
    
    # 测试边界情况 - 全空预测
    print("\n📊 测试3: 全空预测测试")
    empty_pred = [{'boxes': np.array([]), 'scores': np.array([]), 'class_ids': np.array([])}]
    empty_target = [{'boxes': np.array([[0.1, 0.1, 0.2, 0.2]]), 'class_ids': np.array([0])}]
    result3 = calculate_map(empty_pred, empty_target, num_classes=5, debug=True)
    
    # 测试边界情况 - 全空目标
    print("\n📊 测试4: 全空目标测试")
    pred_no_target = [{'boxes': np.array([[0.1, 0.1, 0.2, 0.2]]), 'scores': np.array([0.9]), 'class_ids': np.array([0])}]
    no_target = [{'boxes': np.array([]), 'class_ids': np.array([])}]
    result4 = calculate_map(pred_no_target, no_target, num_classes=5, debug=True)
    
    print("\n🎯 测试总结:")
    print(f"   测试1 mAP: {result['mAP']:.4f}")
    print(f"   测试2 mAP: {result2['mAP']:.4f}")
    print(f"   测试3 mAP: {result3['mAP']:.4f}")
    print(f"   测试4 mAP: {result4['mAP']:.4f}")
    
    print("\n✅ mAP计算功能测试完成!")


if __name__ == "__main__":
    test_map_calculation()


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

