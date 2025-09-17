"""
SwinYOLO评估模块
包含mAP、IoU、NMS等目标检测评估指标和后处理函数
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    计算两个边界框的IoU
    
    Args:
        box1: [N, 4] (x1, y1, x2, y2)
        box2: [M, 4] (x1, y1, x2, y2)
    
    Returns:
        iou: [N, M] IoU矩阵
    """
    # 计算交集区域
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    # 计算交集面积
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # 计算各自面积
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # 计算并集面积
    union_area = area1[:, None] + area2 - inter_area
    
    # 计算IoU
    iou = inter_area / (union_area + 1e-6)
    return iou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    非极大值抑制(NMS)
    
    Args:
        boxes: [N, 4] 边界框坐标 (x1, y1, x2, y2)
        scores: [N] 置信度分数
        iou_threshold: IoU阈值
    
    Returns:
        keep: 保留的边界框索引
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # 按置信度排序
    _, order = scores.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        # 选择置信度最高的框
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # 计算与其他框的IoU
        iou = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        
        # 保留IoU小于阈值的框
        mask = iou <= iou_threshold
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def decode_predictions(predictions: torch.Tensor, 
                      grid_size: int = 7, 
                      num_boxes: int = 2, 
                      num_classes: int = 20,
                      conf_threshold: float = 0.01,
                      input_size: int = 448) -> List[Dict]:
    """
    解码SwinYOLO预测结果
    
    Args:
        predictions: [B, grid_size, grid_size, num_boxes*5 + num_classes]
        grid_size: 网格大小
        num_boxes: 每个网格的边界框数量
        num_classes: 类别数量
        conf_threshold: 置信度阈值
        input_size: 输入图像尺寸
    
    Returns:
        批次中每个图像的检测结果列表
    """
    batch_size = predictions.size(0)
    cell_size = input_size / grid_size
    
    batch_detections = []
    
    for b in range(batch_size):
        pred = predictions[b]  # [grid_size, grid_size, num_boxes*5 + num_classes]
        
        # 分离预测的不同部分
        boxes = pred[..., :num_boxes*4].view(grid_size, grid_size, num_boxes, 4)
        confs = torch.sigmoid(pred[..., num_boxes*4:num_boxes*5]).view(grid_size, grid_size, num_boxes)
        class_probs = torch.sigmoid(pred[..., num_boxes*5:])  # [grid_size, grid_size, num_classes]
        
        detections = []
        
        # 🚀 向量化优化 - 避免三重循环和频繁的.item()调用
        # 🔍 调试信息：显示置信度分布
        if b == 0:  # 只在第一个batch显示一次
            conf_values = confs.flatten()
            max_conf = torch.max(conf_values).item()
            mean_conf = torch.mean(conf_values).item()
            conf_above_001 = (conf_values > 0.001).sum().item()
            conf_above_01 = (conf_values > 0.01).sum().item()
            conf_above_1 = (conf_values > 0.1).sum().item()
            
            if max_conf < 0.1:  # 如果最大置信度都很低，输出警告
                print(f"   🔍 置信度统计: max={max_conf:.4f}, mean={mean_conf:.4f}")
                print(f"   🔍 置信度分布: >0.001({conf_above_001}), >0.01({conf_above_01}), >0.1({conf_above_1})")
        
        # 找到所有满足置信度阈值的框
        conf_mask = confs >= conf_threshold  # [grid_size, grid_size, num_boxes]
        
        if conf_mask.any():
            # 获取满足条件的索引
            valid_indices = torch.where(conf_mask)
            i_indices, j_indices, k_indices = valid_indices
            
            # 批量处理所有有效框
            valid_boxes = boxes[i_indices, j_indices, k_indices]  # [N, 4]
            valid_confs = confs[i_indices, j_indices, k_indices]  # [N]
            
            # 批量计算坐标（YOLO标准格式）
            center_x = (j_indices.float() + valid_boxes[:, 0]) / grid_size  # 归一化到[0,1]
            center_y = (i_indices.float() + valid_boxes[:, 1]) / grid_size  # 归一化到[0,1]
            width = valid_boxes[:, 2]  # 已经是归一化值
            height = valid_boxes[:, 3]  # 已经是归一化值
            
            # 转换为(x1, y1, x2, y2)格式
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            
            # 限制在归一化范围内[0,1]
            x1 = torch.clamp(x1, 0, 1)
            y1 = torch.clamp(y1, 0, 1)
            x2 = torch.clamp(x2, 0, 1)
            y2 = torch.clamp(y2, 0, 1)
            
            # 批量处理类别概率
            valid_class_probs = class_probs[i_indices, j_indices]  # [N, num_classes]
            class_scores = valid_class_probs * valid_confs.unsqueeze(1)  # [N, num_classes]
            max_class_scores, class_indices = torch.max(class_scores, dim=1)  # [N]
            
            # 筛选满足阈值的检测结果
            score_mask = max_class_scores > conf_threshold
            if score_mask.any():
                # 只在最后进行一次CPU转换，大幅减少GPU-CPU传输
                final_boxes = torch.stack([x1, y1, x2, y2], dim=1)[score_mask].cpu().numpy()
                final_scores = max_class_scores[score_mask].cpu().numpy()
                final_classes = class_indices[score_mask].cpu().numpy()
                
                # 批量添加检测结果
                for i in range(len(final_boxes)):
                    detections.append({
                        'bbox': final_boxes[i].tolist(),
                        'score': float(final_scores[i]),
                        'class_id': int(final_classes[i])
                    })
        
        batch_detections.append(detections)
    
    return batch_detections


def apply_nms_to_detections(detections: List[Dict], 
                           iou_threshold: float = 0.5) -> List[Dict]:
    """
    对检测结果应用NMS
    
    Args:
        detections: 检测结果列表
        iou_threshold: IoU阈值
    
    Returns:
        经过NMS处理的检测结果
    """
    if not detections:
        return []
    
    # 按类别分组
    class_detections = defaultdict(list)
    for det in detections:
        class_detections[det['class_id']].append(det)
    
    nms_detections = []
    
    # 对每个类别分别应用NMS
    for class_id, class_dets in class_detections.items():
        if not class_dets:
            continue
        
        boxes = torch.tensor([det['bbox'] for det in class_dets])
        scores = torch.tensor([det['score'] for det in class_dets])
        
        keep_indices = nms(boxes, scores, iou_threshold)
        
        for idx in keep_indices:
            nms_detections.append(class_dets[idx])
    
    return nms_detections


def compute_ap(detections: List[Dict], 
               ground_truths: List[Dict], 
               class_id: int,
               iou_threshold: float = 0.5) -> float:
    """
    计算单个类别的Average Precision
    
    Args:
        detections: 检测结果
        ground_truths: 真实标注
        class_id: 类别ID
        iou_threshold: IoU阈值
    
    Returns:
        AP值
    """
    # 筛选特定类别的检测结果和真实标注
    class_dets = [det for det in detections if det['class_id'] == class_id]
    class_gts = [gt for gt in ground_truths if gt['class_id'] == class_id]
    
    if not class_gts:
        return 0.0
    
    if not class_dets:
        return 0.0
    
    # 按置信度排序
    class_dets.sort(key=lambda x: x['score'], reverse=True)
    
    # 计算TP和FP
    tp = []
    fp = []
    gt_matched = [False] * len(class_gts)
    
    for det in class_dets:
        det_box = torch.tensor(det['bbox']).unsqueeze(0)
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(class_gts):
            if gt_matched[gt_idx]:
                continue
            
            gt_box = torch.tensor(gt['bbox']).unsqueeze(0)
            iou = box_iou(det_box, gt_box)[0, 0].item()
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # 计算precision和recall
    tp = np.array(tp)
    fp = np.array(fp)
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(class_gts)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # 计算AP (11点插值法)
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if mask.any():
            ap += np.max(precisions[mask])
    ap /= 11
    
    return ap


def compute_map(all_detections: List[List[Dict]], 
                all_ground_truths: List[List[Dict]], 
                num_classes: int = 20,
                iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    计算mAP
    
    Args:
        all_detections: 所有图像的检测结果
        all_ground_truths: 所有图像的真实标注
        num_classes: 类别数量
        iou_threshold: IoU阈值
    
    Returns:
        包含各类别AP和mAP的字典
    """
    # 合并所有检测结果和真实标注
    combined_detections = []
    combined_ground_truths = []
    
    for img_dets, img_gts in zip(all_detections, all_ground_truths):
        combined_detections.extend(img_dets)
        combined_ground_truths.extend(img_gts)
    
    # 计算每个类别的AP
    aps = []
    class_aps = {}
    
    for class_id in range(num_classes):
        ap = compute_ap(combined_detections, combined_ground_truths, class_id, iou_threshold)
        aps.append(ap)
        class_aps[f'class_{class_id}'] = ap
    
    # 计算mAP
    map_score = np.mean(aps)
    
    result = {
        'mAP': map_score,
        **class_aps
    }
    
    return result


def evaluate_model(model, dataloader, device, num_classes=20, conf_threshold=0.01, iou_threshold=0.5):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        dataloader: 验证数据加载器
        device: 设备
        num_classes: 类别数量
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
    
    Returns:
        评估结果字典
    """
    model.eval()
    
    all_detections = []
    all_ground_truths = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            
            # 模型预测
            predictions = model(images)
            
            # 解码预测结果
            batch_detections = decode_predictions(
                predictions, 
                conf_threshold=conf_threshold
            )
            
            # 对每个检测结果应用NMS
            for detections in batch_detections:
                nms_detections = apply_nms_to_detections(detections, iou_threshold)
                all_detections.append(nms_detections)
            
            # 处理真实标注（SwinYOLO数据格式）
            # targets是列表：[[gt1, mask_pos1, mask_neg1], [gt2, mask_pos2, mask_neg2], ...]
            # 我们需要提取所有的gt：[gt1, gt2, gt3, ...]
            try:
                if isinstance(targets, list) and len(targets) > 0:
                    gt_list = []
                    for sample_targets in targets:
                        if isinstance(sample_targets, list) and len(sample_targets) >= 1:
                            gt_list.append(sample_targets[0])  # 提取每个样本的gt
                        else:
                            print(f"Warning: 样本targets格式不正确，跳过")
                            continue
                    
                    if len(gt_list) > 0:
                        # 堆叠所有gt tensor
                        gt_targets = torch.stack(gt_list)
                    else:
                        print(f"Warning: 没有有效的gt数据，跳过此批次的mAP计算")
                        continue
                else:
                    # 如果targets不是列表格式，直接使用
                    gt_targets = targets
            except Exception as e:
                print(f"Warning: 处理gt_targets时出错({e})，跳过此批次的mAP计算")
                continue
            
            # 解码真实标注为检测格式
            batch_ground_truths = decode_ground_truths(gt_targets)
            all_ground_truths.extend(batch_ground_truths)
    
    # 🔍 统计预测和真实标签的类别分布 (用于调试)
    from collections import Counter
    pred_class_counts = Counter()
    gt_class_counts = Counter()
    
    for detections in all_detections:
        for det in detections:
            pred_class_counts[det['class_id']] += 1
    
    for ground_truths in all_ground_truths:
        for gt in ground_truths:
            gt_class_counts[gt['class_id']] += 1
    
    print(f"   预测类别统计: {len(pred_class_counts)}/20个类别被预测")
    print(f"   真实类别统计: {len(gt_class_counts)}/20个类别存在")
    
    if len(pred_class_counts) < 5:
        print(f"   ⚠️  预测类别过少! 只预测了{len(pred_class_counts)}个类别")
        print(f"   预测分布: {dict(pred_class_counts.most_common(5))}")
    
    if len(gt_class_counts) < 5:
        print(f"   ⚠️  真实类别过少! 只有{len(gt_class_counts)}个类别")
        print(f"   真实分布: {dict(gt_class_counts.most_common(5))}")
    
    # 计算mAP
    map_results = compute_map(all_detections, all_ground_truths, num_classes, iou_threshold)
    
    return map_results


def decode_ground_truths(gt_targets, 
                        grid_size: int = 7, 
                        num_boxes: int = 2, 
                        num_classes: int = 20,
                        input_size: int = 448) -> List[List[Dict]]:
    """
    解码真实标注为评估格式
    
    Args:
        gt_targets: [B, grid_size, grid_size, num_boxes*5 + num_classes] 或 list
        grid_size: 网格大小
        num_boxes: 每个网格的边界框数量
        num_classes: 类别数量
        input_size: 输入图像尺寸
    
    Returns:
        批次中每个图像的真实标注列表
    """
    # 处理不同的输入格式
    if isinstance(gt_targets, list):
        if len(gt_targets) == 0:
            return []
        # 如果是列表，转换为tensor
        if isinstance(gt_targets[0], torch.Tensor):
            gt_targets = torch.stack(gt_targets)
        else:
            # 如果列表中的元素不是tensor，返回空结果
            return [[] for _ in range(len(gt_targets))]
    
    if not isinstance(gt_targets, torch.Tensor):
        return []
        
    batch_size = gt_targets.size(0)
    cell_size = input_size / grid_size
    
    batch_ground_truths = []
    
    for b in range(batch_size):
        gt = gt_targets[b]  # [grid_size, grid_size, num_boxes*5 + num_classes]
        
        # 分离不同部分
        boxes = gt[..., :num_boxes*4].view(grid_size, grid_size, num_boxes, 4)
        confs = gt[..., num_boxes*4:num_boxes*5].view(grid_size, grid_size, num_boxes)
        class_probs = gt[..., num_boxes*5:]  # [grid_size, grid_size, num_classes]
        
        ground_truths = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 检查是否有目标
                if confs[i, j].max().item() > 0:
                    # 找到置信度最高的边界框
                    best_box_idx = torch.argmax(confs[i, j]).item()
                    
                    # 解码边界框坐标（YOLO标准格式）
                    x = (j + boxes[i, j, best_box_idx, 0].item()) / grid_size  # 归一化到[0,1]
                    y = (i + boxes[i, j, best_box_idx, 1].item()) / grid_size  # 归一化到[0,1]
                    w = boxes[i, j, best_box_idx, 2].item()  # 已经是归一化值
                    h = boxes[i, j, best_box_idx, 3].item()  # 已经是归一化值
                    
                    # 转换为(x1, y1, x2, y2)格式
                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2
                    
                    # 限制在归一化范围内[0,1]
                    x1 = max(0, min(x1, 1))
                    y1 = max(0, min(y1, 1))
                    x2 = max(0, min(x2, 1))
                    y2 = max(0, min(y2, 1))
                    
                    # 获取类别ID
                    class_id = torch.argmax(class_probs[i, j]).item()
                    
                    ground_truths.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id
                    })
        
        batch_ground_truths.append(ground_truths)
    
    return batch_ground_truths
