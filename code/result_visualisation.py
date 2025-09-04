import torch
import numpy as np
import cv2
import os
from typing import Dict, List, Tuple, Optional
import json

from NetModel import YOLOv1
from Test import YOLOTester
from Utils import create_class_names, load_hyperparameters


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self, 
                 model: YOLOv1 = None,
                 hyperparameters: Dict = None,
                 class_names: List[str] = None):
        if hyperparameters is None:
            hyperparameters = load_hyperparameters()
        self.hyperparameters = hyperparameters
        
        if class_names is None:
            self.class_names = create_class_names('voc')
        else:
            self.class_names = class_names
        
        # 颜色映射
        self.colors = self._generate_colors(len(self.class_names))
        
        if model is not None:
            self.tester = YOLOTester(model, hyperparameters, class_names)
        else:
            self.tester = None
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """生成类别颜色"""
        colors = []
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def visualize_detections_advanced(self, 
                                    image: np.ndarray,
                                    detections: Dict,
                                    conf_threshold: float = 0.5,
                                    show_labels: bool = True,
                                    show_confidence: bool = True,
                                    line_thickness: int = 2,
                                    font_scale: float = 0.5) -> np.ndarray:
        vis_image = image.copy()
        
        boxes = detections['boxes']
        scores = detections['scores']
        class_ids = detections['class_ids']
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score >= conf_threshold:
                x1, y1, x2, y2 = box.astype(int)
                class_id = int(class_id)
                
                # 获取颜色
                color = self.colors[class_id % len(self.colors)]
                
                # 绘制边界框
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, line_thickness)
                
                # 绘制标签和置信度
                if show_labels or show_confidence:
                    label_parts = []
                    if show_labels:
                        label_parts.append(self.class_names[class_id])
                    if show_confidence:
                        label_parts.append(f'{score:.2f}')
                    
                    label = ': '.join(label_parts)
                    
                    # 计算文本大小
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness
                    )
                    
                    # 绘制文本背景
                    cv2.rectangle(
                        vis_image, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        color, -1
                    )
                    
                    # 绘制文本
                    cv2.putText(
                        vis_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), line_thickness
                    )
        
        return vis_image
    
    def create_detection_summary(self, detections: Dict, conf_threshold: float = 0.5) -> Dict:
        if len(detections['boxes']) == 0:
            return {
                'total_detections': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0
            }
        
        # 过滤低置信度检测
        valid_indices = detections['scores'] >= conf_threshold
        valid_scores = detections['scores'][valid_indices]
        valid_class_ids = detections['class_ids'][valid_indices]
        
        # 统计类别数量
        class_counts = {}
        for class_id in valid_class_ids:
            class_name = self.class_names[int(class_id)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary = {
            'total_detections': len(valid_scores),
            'class_counts': class_counts,
            'avg_confidence': float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0,
            'max_confidence': float(np.max(valid_scores)) if len(valid_scores) > 0 else 0.0,
            'min_confidence': float(np.min(valid_scores)) if len(valid_scores) > 0 else 0.0
        }
        
        return summary
    
    def visualize_with_summary(self, 
                              image: np.ndarray,
                              detections: Dict,
                              conf_threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
        vis_image = self.visualize_detections_advanced(image, detections, conf_threshold)
        summary = self.create_detection_summary(detections, conf_threshold)
        
        # 在图像上添加摘要信息
        y_offset = 30
        cv2.putText(
            vis_image, f"Total Detections: {summary['total_detections']}", 
            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        y_offset += 30
        for class_name, count in summary['class_counts'].items():
            cv2.putText(
                vis_image, f"{class_name}: {count}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            y_offset += 25
        
        return vis_image, summary
    
    def compare_detections(self, 
                          image: np.ndarray,
                          detections1: Dict,
                          detections2: Dict,
                          titles: List[str] = None) -> np.ndarray:
        if titles is None:
            titles = ['Detection 1', 'Detection 2']
        
        # 创建两个可视化图像
        vis1 = self.visualize_detections_advanced(image, detections1)
        vis2 = self.visualize_detections_advanced(image, detections2)
        
        # 添加标题
        cv2.putText(vis1, titles[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis2, titles[1], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 水平拼接
        comparison_image = np.hstack([vis1, vis2])
        
        return comparison_image
    
    def create_confidence_distribution_image(self, 
                                           detections: Dict,
                                           image_size: Tuple[int, int] = (400, 300)) -> np.ndarray:
        width, height = image_size
        dist_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(detections['scores']) == 0:
            cv2.putText(dist_image, "No Detections", (width//4, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return dist_image
        
        # 计算置信度分布
        scores = detections['scores']
        bins = np.linspace(0, 1, 11)  # 10个区间
        hist, _ = np.histogram(scores, bins)
        
        # 绘制直方图
        bar_width = width // len(hist)
        max_count = max(hist) if max(hist) > 0 else 1
        
        for i, count in enumerate(hist):
            bar_height = int((count / max_count) * (height - 50))
            x1 = i * bar_width
            x2 = (i + 1) * bar_width
            y1 = height - 30
            y2 = y1 - bar_height
            
            cv2.rectangle(dist_image, (x1, y1), (x2, y2), (0, 255, 0), -1)
            cv2.rectangle(dist_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # 添加标签
            cv2.putText(dist_image, f'{bins[i]:.1f}', (x1, height-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # 添加标题
        cv2.putText(dist_image, "Confidence Distribution", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return dist_image
    
    def save_detection_results(self, 
                              detections_data: List[Dict],
                              output_dir: str,
                              format_type: str = 'json'):
        os.makedirs(output_dir, exist_ok=True)
        
        if format_type == 'json':
            # 转换numpy数组为列表
            json_data = []
            for item in detections_data:
                json_item = {
                    'image_file': item['image_file'],
                    'detections': {
                        'boxes': item['detections']['boxes'].tolist() if len(item['detections']['boxes']) > 0 else [],
                        'scores': item['detections']['scores'].tolist() if len(item['detections']['scores']) > 0 else [],
                        'class_ids': item['detections']['class_ids'].tolist() if len(item['detections']['class_ids']) > 0 else []
                    }
                }
                json_data.append(json_item)
            
            output_path = os.path.join(output_dir, 'detections.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'txt':
            # YOLO格式的文本文件
            for item in detections_data:
                base_name = os.path.splitext(item['image_file'])[0]
                txt_file = os.path.join(output_dir, f'{base_name}.txt')
                
                with open(txt_file, 'w') as f:
                    detections = item['detections']
                    for box, score, class_id in zip(
                        detections['boxes'], detections['scores'], detections['class_ids']
                    ):
                        x1, y1, x2, y2 = box
                        f.write(f"{int(class_id)} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
        
        print(f"Detection results saved to {output_dir} in {format_type} format")
    
    def create_detection_report(self, 
                               detections_data: List[Dict],
                               output_path: str):
        total_images = len(detections_data)
        total_detections = sum(len(item['detections']['boxes']) for item in detections_data)
        
        # 统计类别
        class_stats = {}
        confidence_scores = []
        
        for item in detections_data:
            detections = item['detections']
            for score, class_id in zip(detections['scores'], detections['class_ids']):
                class_name = self.class_names[int(class_id)]
                class_stats[class_name] = class_stats.get(class_name, 0) + 1
                confidence_scores.append(score)
        
        # 生成报告
        report = {
            'summary': {
                'total_images': total_images,
                'total_detections': total_detections,
                'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
                'unique_classes_detected': len(class_stats)
            },
            'class_statistics': class_stats,
            'confidence_statistics': {
                'mean': float(np.mean(confidence_scores)) if confidence_scores else 0,
                'std': float(np.std(confidence_scores)) if confidence_scores else 0,
                'min': float(np.min(confidence_scores)) if confidence_scores else 0,
                'max': float(np.max(confidence_scores)) if confidence_scores else 0
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Detection report saved to {output_path}")
        return report


def visualize_training_progress(train_losses: List[float], 
                               val_losses: List[float] = None,
                               val_maps: List[float] = None,
                               save_path: str = None) -> np.ndarray:
    # 创建简单的文本报告（由于没有matplotlib）
    width, height = 800, 600
    progress_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加标题
    cv2.putText(progress_image, "Training Progress", (width//2-100, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示统计信息
    y_offset = 80
    
    if train_losses:
        cv2.putText(progress_image, f"Final Train Loss: {train_losses[-1]:.4f}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 40
        
        cv2.putText(progress_image, f"Best Train Loss: {min(train_losses):.4f}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 40
    
    if val_losses:
        cv2.putText(progress_image, f"Final Val Loss: {val_losses[-1]:.4f}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 40
        
        cv2.putText(progress_image, f"Best Val Loss: {min(val_losses):.4f}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 40
    
    if val_maps:
        cv2.putText(progress_image, f"Final mAP: {val_maps[-1]:.4f}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += 40
        
        cv2.putText(progress_image, f"Best mAP: {max(val_maps):.4f}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        y_offset += 40
    
    cv2.putText(progress_image, f"Total Epochs: {len(train_losses)}", 
               (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if save_path:
        cv2.imwrite(save_path, progress_image)
        print(f"Training progress saved to {save_path}")
    
    return progress_image


def main():
    """主可视化函数"""
    print("YOLO Result Visualization Tool")
    print("1. Visualize single image detection")
    print("2. Create detection report")
    print("3. Visualize training progress")
    
    choice = input("Choose option (1-3): ")
    
    if choice == '1':
        # 单图像可视化
        image_path = input("Enter image path: ")
        if os.path.exists(image_path):
            # 创建简单的可视化器
            visualizer = ResultVisualizer()
            
            # 读取图像
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 创建虚拟检测结果用于演示
            dummy_detections = {
                'boxes': np.array([[100, 100, 200, 200], [300, 150, 400, 250]]),
                'scores': np.array([0.9, 0.7]),
                'class_ids': np.array([0, 1])
            }
            
            vis_image, summary = visualizer.visualize_with_summary(image, dummy_detections)
            
            output_path = './visualization_output.jpg'
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, vis_image_bgr)
            
            print(f"Visualization saved to {output_path}")
            print("Detection Summary:", summary)
        else:
            print("Image not found!")
    
    elif choice == '2':
        print("Detection report functionality would be implemented here")
    
    elif choice == '3':
        # 训练进度可视化
        dummy_train_losses = [2.5, 2.2, 1.8, 1.5, 1.2, 1.0, 0.8]
        dummy_val_losses = [2.8, 2.4, 2.0, 1.7, 1.4, 1.1, 0.9]
        dummy_val_maps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65]
        
        progress_image = visualize_training_progress(
            dummy_train_losses, dummy_val_losses, dummy_val_maps,
            save_path='./training_progress.jpg'
        )
        print("Training progress visualization created")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
