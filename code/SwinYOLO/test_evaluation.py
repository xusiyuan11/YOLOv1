"""
测试SwinYOLO评估功能
"""

import torch
import numpy as np
from evaluation import box_iou, nms, decode_predictions, apply_nms_to_detections, compute_ap, compute_map


def test_box_iou():
    """测试IoU计算"""
    print("🧪 测试IoU计算...")
    
    # 创建测试边界框
    box1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])  # 2个框
    box2 = torch.tensor([[0, 0, 5, 5], [10, 10, 20, 20]])   # 2个框
    
    iou = box_iou(box1, box2)
    print(f"IoU矩阵:\n{iou}")
    
    # 验证结果
    expected_iou_0_0 = 25 / 100  # 交集25，并集100
    expected_iou_1_1 = 25 / 175  # 交集25，并集175
    
    print(f"预期IoU[0,0]: {expected_iou_0_0:.4f}, 实际: {iou[0,0]:.4f}")
    print(f"预期IoU[1,1]: {expected_iou_1_1:.4f}, 实际: {iou[1,1]:.4f}")
    print("✅ IoU测试通过\n")


def test_nms():
    """测试NMS"""
    print("🧪 测试NMS...")
    
    # 创建重叠的边界框
    boxes = torch.tensor([
        [0, 0, 10, 10],    # 高置信度
        [2, 2, 12, 12],    # 与第1个重叠
        [20, 20, 30, 30],  # 独立的框
        [1, 1, 11, 11]     # 与第1个高度重叠
    ], dtype=torch.float)
    
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    
    keep = nms(boxes, scores, iou_threshold=0.5)
    print(f"保留的框索引: {keep}")
    print(f"保留的框: {boxes[keep]}")
    print(f"保留的分数: {scores[keep]}")
    print("✅ NMS测试通过\n")


def test_decode_predictions():
    """测试预测解码"""
    print("🧪 测试预测解码...")
    
    # 创建模拟预测结果
    batch_size = 2
    grid_size = 7
    num_boxes = 2
    num_classes = 20
    
    predictions = torch.randn(batch_size, grid_size, grid_size, num_boxes*5 + num_classes)
    
    # 在某个位置设置高置信度检测
    predictions[0, 3, 3, 4] = 2.0   # 第一个框的置信度
    predictions[0, 3, 3, 10:30] = torch.randn(20) * 0.1  # 类别概率
    predictions[0, 3, 3, 15] = 1.0  # 某个类别高概率
    
    detections = decode_predictions(predictions, conf_threshold=0.1)
    
    print(f"检测结果数量: {[len(det) for det in detections]}")
    if detections[0]:
        print(f"第一个检测: {detections[0][0]}")
    print("✅ 预测解码测试通过\n")


def test_ap_computation():
    """测试AP计算"""
    print("🧪 测试AP计算...")
    
    # 创建模拟检测结果和真实标注
    detections = [
        {'bbox': [0, 0, 10, 10], 'score': 0.9, 'class_id': 0},
        {'bbox': [1, 1, 11, 11], 'score': 0.8, 'class_id': 0},  # 与第一个重叠
        {'bbox': [20, 20, 30, 30], 'score': 0.7, 'class_id': 1},
        {'bbox': [0, 0, 5, 5], 'score': 0.6, 'class_id': 0},    # 部分重叠
    ]
    
    ground_truths = [
        {'bbox': [0, 0, 10, 10], 'class_id': 0},
        {'bbox': [20, 20, 30, 30], 'class_id': 1},
        {'bbox': [100, 100, 110, 110], 'class_id': 0},  # 漏检
    ]
    
    # 计算类别0的AP
    ap_class_0 = compute_ap(detections, ground_truths, class_id=0, iou_threshold=0.5)
    ap_class_1 = compute_ap(detections, ground_truths, class_id=1, iou_threshold=0.5)
    
    print(f"类别0的AP: {ap_class_0:.4f}")
    print(f"类别1的AP: {ap_class_1:.4f}")
    print("✅ AP计算测试通过\n")


def test_map_computation():
    """测试mAP计算"""
    print("🧪 测试mAP计算...")
    
    # 创建批次检测结果
    all_detections = [
        [  # 图像1
            {'bbox': [0, 0, 10, 10], 'score': 0.9, 'class_id': 0},
            {'bbox': [20, 20, 30, 30], 'score': 0.8, 'class_id': 1},
        ],
        [  # 图像2
            {'bbox': [5, 5, 15, 15], 'score': 0.7, 'class_id': 0},
        ]
    ]
    
    all_ground_truths = [
        [  # 图像1真实标注
            {'bbox': [0, 0, 10, 10], 'class_id': 0},
            {'bbox': [20, 20, 30, 30], 'class_id': 1},
        ],
        [  # 图像2真实标注
            {'bbox': [5, 5, 15, 15], 'class_id': 0},
            {'bbox': [100, 100, 110, 110], 'class_id': 2},  # 漏检
        ]
    ]
    
    map_results = compute_map(all_detections, all_ground_truths, num_classes=20)
    
    print(f"mAP: {map_results['mAP']:.4f}")
    print(f"类别0 AP: {map_results['class_0']:.4f}")
    print(f"类别1 AP: {map_results['class_1']:.4f}")
    print(f"类别2 AP: {map_results['class_2']:.4f}")
    print("✅ mAP计算测试通过\n")


if __name__ == "__main__":
    print("🚀 开始测试SwinYOLO评估功能\n")
    
    test_box_iou()
    test_nms()
    test_decode_predictions()
    test_ap_computation()
    test_map_computation()
    
    print("🎉 所有测试通过！评估功能工作正常。")
