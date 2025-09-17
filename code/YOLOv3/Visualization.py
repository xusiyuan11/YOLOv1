"""
YOLOv3 训练过程可视化模块
专为YOLOv3多尺度检测优化
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 尝试导入seaborn，如果失败则继续
try:
    import seaborn as sns
except ImportError:
    sns = None

# 使用英文界面避免中文字体问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class YOLOv3TrainingVisualizer:
    """YOLOv3训练过程可视化器"""
    
    def __init__(self, save_dir: str = './training_visualizations'):
        """
        初始化训练可视化器
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # 使用默认样式
        
        # 颜色方案
        self.colors = {
            'train': '#2E86AB',
            'val': '#A23B72',
            'loss': '#F18F01',
            'acc': '#C73E1D',
            'map': '#7209B7',
            'bbox': '#FF6B6B',
            'objectness': '#4ECDC4',
            'classification': '#45B7D1',
            'iou': '#FF6B35',
            'top1': '#2ECC71',
            'map50': '#E74C3C',
            'map75': '#9B59B6'
        }
        
        print(f"YOLOv3训练可视化器初始化完成，保存目录: {save_dir}")
    
    def plot_loss_curves(self, 
                        train_losses: List[float],
                        val_losses: List[float],
                        train_bbox_losses: Optional[List[float]] = None,
                        train_obj_losses: Optional[List[float]] = None,
                        train_cls_losses: Optional[List[float]] = None,
                        title: str = "YOLOv3 Loss Curves",
                        save_path: str = None):
        """
        绘制YOLOv3损失曲线（包括总损失和分项损失）
        """
        epochs = range(1, len(train_losses) + 1)
        
        # 确定子图布局
        has_detailed_losses = train_bbox_losses and train_obj_losses and train_cls_losses
        if has_detailed_losses:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # 总损失曲线
        ax1.plot(epochs, train_losses, 'o-', color=self.colors['train'], 
                label='Train Loss', linewidth=2, markersize=3)
        ax1.plot(epochs, val_losses, 's-', color=self.colors['val'], 
                label='Val Loss', linewidth=2, markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Loss')
        ax1.set_title('Total Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if has_detailed_losses:
            # 边界框损失
            ax2.plot(epochs[:len(train_bbox_losses)], train_bbox_losses, 'o-', 
                    color=self.colors['bbox'], label='BBox Loss', linewidth=2, markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('BBox Loss')
            ax2.set_title('Bounding Box Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 目标性损失
            ax3.plot(epochs[:len(train_obj_losses)], train_obj_losses, 'o-', 
                    color=self.colors['objectness'], label='Objectness Loss', linewidth=2, markersize=3)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Objectness Loss')
            ax3.set_title('Objectness Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 分类损失
            ax4.plot(epochs[:len(train_cls_losses)], train_cls_losses, 'o-', 
                    color=self.colors['classification'], label='Classification Loss', linewidth=2, markersize=3)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Classification Loss')
            ax4.set_title('Classification Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"损失曲线已保存: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'loss_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_map_curves(self,
                       train_maps: List[float],
                       val_maps: List[float],
                       map_50: Optional[List[float]] = None,
                       map_75: Optional[List[float]] = None,
                       title: str = "YOLOv3 mAP Progress",
                       save_path: str = None):
        """
        绘制mAP进度曲线
        """
        epochs = range(1, len(train_maps) + 1)
        
        if map_50 and map_75:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # 基本mAP曲线
        ax1.plot(epochs, train_maps, 'o-', color=self.colors['train'], 
                label='Train mAP', linewidth=2, markersize=3)
        ax1.plot(epochs, val_maps, 's-', color=self.colors['val'], 
                label='Val mAP', linewidth=2, markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP')
        ax1.set_title('Mean Average Precision')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if map_50 and map_75 and len(epochs) > 1:
            # mAP@0.5 和 mAP@0.75
            ax2.plot(epochs[:len(map_50)], map_50, 'o-', 
                    color='#FF9F43', label='mAP@0.5', linewidth=2, markersize=3)
            ax2.plot(epochs[:len(map_75)], map_75, 's-', 
                    color='#6C5CE7', label='mAP@0.75', linewidth=2, markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mAP')
            ax2.set_title('mAP at Different IoU Thresholds')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"mAP曲线已保存: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'map_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_learning_rate_schedule(self,
                                   learning_rates: List[float],
                                   title: str = "Learning Rate Schedule",
                                   save_path: str = None):
        """
        绘制学习率调度曲线
        """
        epochs = range(1, len(learning_rates) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, learning_rates, 'o-', color=self.colors['loss'], 
                linewidth=2, markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数刻度
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.save_dir, 'learning_rate.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_iou_metrics(self,
                        train_ious: List[float],
                        val_ious: List[float],
                        iou_50: Optional[List[float]] = None,
                        iou_75: Optional[List[float]] = None,
                        title: str = "YOLOv3 IoU Metrics",
                        save_path: str = None):
        """
        绘制IoU指标曲线
        Args:
            train_ious: 训练集平均IoU
            val_ious: 验证集平均IoU
            iou_50: IoU@0.5指标
            iou_75: IoU@0.75指标
            title: 图表标题
            save_path: 保存路径
        """
        epochs = range(1, len(train_ious) + 1)
        
        if iou_50 and iou_75:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # 基本IoU曲线
        ax1.plot(epochs, train_ious, 'o-', color=self.colors['train'], 
                label='Train IoU', linewidth=2, markersize=3)
        ax1.plot(epochs, val_ious, 's-', color=self.colors['val'], 
                label='Val IoU', linewidth=2, markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average IoU')
        ax1.set_title('Average IoU Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        if iou_50 and iou_75:
            # IoU@不同阈值
            ax2.plot(epochs[:len(iou_50)], iou_50, 'o-', 
                    color=self.colors['map50'], label='IoU@0.5', linewidth=2, markersize=3)
            ax2.plot(epochs[:len(iou_75)], iou_75, 's-', 
                    color=self.colors['map75'], label='IoU@0.75', linewidth=2, markersize=3)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('IoU')
            ax2.set_title('IoU at Different Thresholds')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"IoU曲线已保存: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'iou_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_accuracy_metrics(self,
                             train_top1: List[float],
                             val_top1: List[float],
                             train_top5: Optional[List[float]] = None,
                             val_top5: Optional[List[float]] = None,
                             title: str = "YOLOv3 Classification Accuracy",
                             save_path: str = None):
        """
        绘制分类精度曲线
        Args:
            train_top1: 训练集Top-1精度
            val_top1: 验证集Top-1精度
            train_top5: 训练集Top-5精度（可选）
            val_top5: 验证集Top-5精度（可选）
            title: 图表标题
            save_path: 保存路径
        """
        epochs = range(1, len(train_top1) + 1)
        
        plt.figure(figsize=(12, 6))
        
        # Top-1精度
        plt.plot(epochs, train_top1, 'o-', color=self.colors['train'], 
                label='Train Top-1', linewidth=2, markersize=3)
        plt.plot(epochs, val_top1, 's-', color=self.colors['val'], 
                label='Val Top-1', linewidth=2, markersize=3)
        
        # Top-5精度（如果提供）
        if train_top5 and val_top5:
            plt.plot(epochs[:len(train_top5)], train_top5, 'o--', color=self.colors['train'], 
                    alpha=0.7, label='Train Top-5', linewidth=2, markersize=2)
            plt.plot(epochs[:len(val_top5)], val_top5, 's--', color=self.colors['val'], 
                    alpha=0.7, label='Val Top-5', linewidth=2, markersize=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"精度曲线已保存: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'accuracy_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_enhanced_map_curves(self,
                                train_maps: List[float],
                                val_maps: List[float],
                                map_50: List[float],
                                map_75: Optional[List[float]] = None,
                                map_small: Optional[List[float]] = None,
                                map_medium: Optional[List[float]] = None,
                                map_large: Optional[List[float]] = None,
                                title: str = "Enhanced YOLOv3 mAP Analysis",
                                save_path: str = None):
        """
        增强版mAP曲线绘制，包含更详细的mAP分析
        Args:
            train_maps: 训练集mAP
            val_maps: 验证集mAP
            map_50: mAP@0.5
            map_75: mAP@0.75（可选）
            map_small: 小目标mAP（可选）
            map_medium: 中等目标mAP（可选）
            map_large: 大目标mAP（可选）
            title: 图表标题
            save_path: 保存路径
        """
        epochs = range(1, len(train_maps) + 1)
        
        # 确定子图布局
        has_size_maps = map_small and map_medium and map_large
        if has_size_maps:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 基本mAP曲线
        ax1.plot(epochs, train_maps, 'o-', color=self.colors['train'], 
                label='Train mAP', linewidth=2, markersize=3)
        ax1.plot(epochs, val_maps, 's-', color=self.colors['val'], 
                label='Val mAP', linewidth=2, markersize=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP')
        ax1.set_title('Mean Average Precision')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # mAP@不同IoU阈值
        ax2.plot(epochs[:len(map_50)], map_50, 'o-', 
                color=self.colors['map50'], label='mAP@0.5', linewidth=3, markersize=4)
        if map_75:
            ax2.plot(epochs[:len(map_75)], map_75, 's-', 
                    color=self.colors['map75'], label='mAP@0.75', linewidth=2, markersize=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.set_title('mAP at Different IoU Thresholds')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        if has_size_maps:
            # 不同尺寸目标的mAP
            ax3.plot(epochs[:len(map_small)], map_small, 'o-', 
                    color='#FF6B6B', label='Small Objects', linewidth=2, markersize=3)
            ax3.plot(epochs[:len(map_medium)], map_medium, 's-', 
                    color='#4ECDC4', label='Medium Objects', linewidth=2, markersize=3)
            ax3.plot(epochs[:len(map_large)], map_large, '^-', 
                    color='#45B7D1', label='Large Objects', linewidth=2, markersize=3)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('mAP')
            ax3.set_title('mAP by Object Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # mAP改进趋势
            if len(map_50) > 1:
                map_50_improvement = [map_50[i] - map_50[i-1] for i in range(1, len(map_50))]
                ax4.bar(range(2, len(map_50) + 1), map_50_improvement, 
                       color=self.colors['map50'], alpha=0.7, label='mAP@0.5 Improvement')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('mAP Improvement')
                ax4.set_title('mAP@0.5 Improvement per Epoch')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"增强版mAP曲线已保存: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'enhanced_map_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def visualize_anchors(self,
                         anchor_boxes: List[List[float]],
                         input_size: int = 416,
                         title: str = "YOLOv3 Anchor Boxes",
                         save_path: str = None):
        """
        可视化YOLOv3的anchor boxes
        Args:
            anchor_boxes: 嵌套列表，每个子列表包含 [width, height] 对
            input_size: 输入图像尺寸
            title: 图表标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # 设置坐标轴
        ax.set_xlim(0, input_size)
        ax.set_ylim(0, input_size)
        ax.set_aspect('equal')
        
        # 绘制网格
        ax.grid(True, alpha=0.3)
        
        # 绘制anchor boxes（在图像中心）
        center_x, center_y = input_size // 2, input_size // 2
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
        
        for i, anchors in enumerate(anchor_boxes):
            scale_name = f"Scale {i+1}"
            for j, (w, h) in enumerate(anchors):
                color = colors[(i * 3 + j) % len(colors)]
                
                # 计算anchor box的坐标
                x1 = center_x - w / 2
                y1 = center_y - h / 2
                
                # 绘制矩形
                rect = patches.Rectangle(
                    (x1, y1), w, h,
                    linewidth=2, edgecolor=color, facecolor='none',
                    label=f"{scale_name} Anchor {j+1} ({w:.0f}x{h:.0f})"
                )
                ax.add_patch(rect)
        
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Height (pixels)')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 添加中心点
        ax.plot(center_x, center_y, 'ko', markersize=8, label='Center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Anchor可视化已保存: {save_path}")
        else:
            save_path = os.path.join(self.save_dir, 'anchor_boxes.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()

class YOLOTrainingVisualizer(YOLOv3TrainingVisualizer):
    """为了兼容性保留的别名"""
    pass
