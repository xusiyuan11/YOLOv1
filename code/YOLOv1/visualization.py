"""
YOLO 训练过程可视化模块 - 中文乱码修复版
使用英文标签避免中文字体问题
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import os
import json
import torch
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 使用英文界面避免中文字体问题
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class YOLOTrainingVisualizer:
    """YOLO训练过程可视化器 - 英文版"""
    
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
            plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 颜色方案
        self.colors = {
            'train': '#2E86AB',
            'val': '#A23B72',
            'loss': '#F18F01',
            'acc': '#C73E1D',
            'map': '#7209B7'
        }
        
        print(f"训练可视化器初始化完成，保存目录: {save_dir}")
    
    def plot_training_curves(self, 
                           train_losses: List[float],
                           val_losses: List[float],
                           train_accuracies: Optional[List[float]] = None,
                           val_accuracies: Optional[List[float]] = None,
                           train_maps: Optional[List[float]] = None,
                           val_maps: Optional[List[float]] = None,
                           title: str = "Training Curves",
                           save_name: str = "training_curves.png"):
        """
        绘制训练曲线（英文版）
        """
        epochs = range(1, len(train_losses) + 1)
        
        # 确定子图数量
        subplot_count = 1  # 至少有损失图
        if train_accuracies and val_accuracies:
            subplot_count += 1
        if train_maps and val_maps:
            subplot_count += 1
        
        fig, axes = plt.subplots(1, subplot_count, figsize=(6*subplot_count, 5))
        if subplot_count == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 1. 损失曲线
        ax = axes[plot_idx]
        ax.plot(epochs, train_losses, 'o-', color=self.colors['train'], 
                label='Train Loss', linewidth=2, markersize=4)
        ax.plot(epochs, val_losses, 's-', color=self.colors['val'], 
                label='Val Loss', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
        # 2. 精度曲线（如果有）
        if train_accuracies and val_accuracies and plot_idx < len(axes):
            ax = axes[plot_idx]
            ax.plot(epochs[:len(train_accuracies)], train_accuracies, 'o-', 
                    color=self.colors['train'], label='Train Acc', linewidth=2, markersize=4)
            ax.plot(epochs[:len(val_accuracies)], val_accuracies, 's-', 
                    color=self.colors['val'], label='Val Acc', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Accuracy Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # 3. mAP曲线（如果有）
        if train_maps and val_maps and plot_idx < len(axes):
            ax = axes[plot_idx]
            ax.plot(epochs[:len(train_maps)], train_maps, 'o-', 
                    color=self.colors['train'], label='Train mAP', linewidth=2, markersize=4)
            ax.plot(epochs[:len(val_maps)], val_maps, 's-', 
                    color=self.colors['val'], label='Val mAP', linewidth=2, markersize=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.set_title('mAP Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training curves saved: {save_path}")
    
    def plot_loss_components(self, 
                           loss_history: List[Dict],
                           title: str = "Loss Components Analysis",
                           save_name: str = "loss_components.png"):
        """
        绘制损失组件分析图（英文版）
        """
        if len(loss_history) == 0:
            print("No loss history data, skipping loss components analysis")
            return
        
        epochs = range(1, len(loss_history) + 1)
        
        # 提取各组件损失
        total_losses = [item.get('total_loss', 0) for item in loss_history]
        coord_losses = [item.get('coord_loss', 0) for item in loss_history]
        conf_obj_losses = [item.get('conf_loss_obj', 0) for item in loss_history]
        conf_noobj_losses = [item.get('conf_loss_noobj', 0) for item in loss_history]
        class_losses = [item.get('class_loss', 0) for item in loss_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 总损失
        ax1.plot(epochs, total_losses, 'o-', color='red', linewidth=2, markersize=3)
        ax1.set_title('Total Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # 各组件损失对比
        ax2.plot(epochs, coord_losses, 'o-', label='Coord Loss', linewidth=2, markersize=3)
        ax2.plot(epochs, conf_obj_losses, 's-', label='Conf Loss (Obj)', linewidth=2, markersize=3)
        ax2.plot(epochs, conf_noobj_losses, '^-', label='Conf Loss (NoObj)', linewidth=2, markersize=3)
        ax2.plot(epochs, class_losses, 'd-', label='Class Loss', linewidth=2, markersize=3)
        ax2.set_title('Loss Components', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 损失比例堆叠图
        if len(epochs) > 0:
            bottom = np.zeros(len(epochs))
            components = [coord_losses, conf_obj_losses, conf_noobj_losses, class_losses]
            labels = ['Coord Loss', 'Conf Loss (Obj)', 'Conf Loss (NoObj)', 'Class Loss']
            colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
            
            for comp, label, color in zip(components, labels, colors):
                ax3.bar(epochs, comp, bottom=bottom, label=label, color=color, alpha=0.8)
                bottom += np.array(comp)
        
        ax3.set_title('Stacked Loss Components', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        
        # 损失比例饼图（最后一个epoch）
        if loss_history:
            last_losses = loss_history[-1]
            pie_data = [
                last_losses.get('coord_loss', 0),
                last_losses.get('conf_loss_obj', 0),
                last_losses.get('conf_loss_noobj', 0),
                last_losses.get('class_loss', 0)
            ]
            pie_labels = ['Coord Loss', 'Conf Loss (Obj)', 'Conf Loss (NoObj)', 'Class Loss']
            pie_colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
            
            ax4.pie(pie_data, labels=pie_labels, colors=pie_colors, 
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'Final Loss Distribution (Epoch {len(loss_history)})', fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Loss components analysis saved: {save_path}")
    
    def plot_iou_distribution(self, 
                            ious: List[float],
                            title: str = "IoU Distribution Analysis",
                            save_name: str = "iou_distribution.png"):
        """
        绘制IoU分布分析（英文版）
        """
        if len(ious) == 0:
            print("No IoU data, skipping IoU distribution analysis")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # IoU直方图
        ax1.hist(ious, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(ious), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ious):.3f}')
        ax1.axvline(np.median(ious), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ious):.3f}')
        ax1.set_title('IoU Distribution Histogram', fontweight='bold')
        ax1.set_xlabel('IoU Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # IoU箱线图
        bp = ax2.boxplot(ious, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_title('IoU Box Plot', fontweight='bold')
        ax2.set_ylabel('IoU Value')
        ax2.grid(True, alpha=0.3)
        
        # IoU累积分布函数
        sorted_ious = np.sort(ious)
        cumulative = np.arange(1, len(sorted_ious) + 1) / len(sorted_ious)
        ax3.plot(sorted_ious, cumulative, 'b-', linewidth=2)
        ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='IoU=0.5')
        ax3.axvline(0.75, color='orange', linestyle='--', linewidth=2, label='IoU=0.75')
        ax3.set_title('IoU Cumulative Distribution', fontweight='bold')
        ax3.set_xlabel('IoU Value')
        ax3.set_ylabel('Cumulative Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # IoU统计信息
        stats_text = f"""
        IoU Statistics:
        Total: {len(ious)}
        Mean: {np.mean(ious):.4f}
        Median: {np.median(ious):.4f}
        Std: {np.std(ious):.4f}
        Min: {np.min(ious):.4f}
        Max: {np.max(ious):.4f}
        
        Threshold Stats:
        IoU ≥ 0.5: {(np.array(ious) >= 0.5).sum()}/{len(ious)} ({100*(np.array(ious) >= 0.5).mean():.1f}%)
        IoU ≥ 0.75: {(np.array(ious) >= 0.75).sum()}/{len(ious)} ({100*(np.array(ious) >= 0.75).mean():.1f}%)
        IoU ≥ 0.9: {(np.array(ious) >= 0.9).sum()}/{len(ious)} ({100*(np.array(ious) >= 0.9).mean():.1f}%)
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_title('IoU Statistics', fontweight='bold')
        ax4.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"IoU distribution analysis saved: {save_path}")
    
    def plot_training_summary(self, 
                            checkpoint_data: Dict,
                            title: str = "Training Summary Report",
                            save_name: str = "training_summary.png"):
        """
        绘制训练总结报告
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 最终指标摘要
        metrics_text = f"""
        Training Summary:
        
        Final Epoch: {checkpoint_data.get('epoch', 'N/A')}
        Best Loss: {checkpoint_data.get('best_loss', 0):.4f}
        Best Accuracy: {checkpoint_data.get('best_acc', 0):.2f}%
        
        Training Configuration:
        Input Size: {checkpoint_data.get('input_size', 448)}
        Batch Size: {checkpoint_data.get('batch_size', 16)}
        Learning Rate: {checkpoint_data.get('learning_rate', 0.001)}
        
        Model Performance:
        Status: Training Completed
        Total Training Time: {checkpoint_data.get('training_time', 'N/A')}
        """
        
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax1.set_title('Training Summary', fontweight='bold', fontsize=14)
        ax1.axis('off')
        
        # 2. 损失趋势（如果有数据）
        if 'train_losses' in checkpoint_data and checkpoint_data['train_losses']:
            epochs = range(1, len(checkpoint_data['train_losses']) + 1)
            ax2.plot(epochs, checkpoint_data['train_losses'], 'b-', linewidth=2, label='Train Loss')
            if 'val_losses' in checkpoint_data and checkpoint_data['val_losses']:
                ax2.plot(epochs, checkpoint_data['val_losses'], 'r-', linewidth=2, label='Val Loss')
            ax2.set_title('Loss Curve', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No loss data available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Loss Curve', fontweight='bold')
            ax2.axis('off')
        
        # 3. 准确率趋势（如果有数据）
        if 'train_accuracies' in checkpoint_data and checkpoint_data['train_accuracies']:
            epochs = range(1, len(checkpoint_data['train_accuracies']) + 1)
            ax3.plot(epochs, checkpoint_data['train_accuracies'], 'g-', linewidth=2, label='Train Acc')
            if 'val_accuracies' in checkpoint_data and checkpoint_data['val_accuracies']:
                ax3.plot(epochs, checkpoint_data['val_accuracies'], 'orange', linewidth=2, label='Val Acc')
            ax3.set_title('Accuracy Curve', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Accuracy (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No accuracy data available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Accuracy Curve', fontweight='bold')
            ax3.axis('off')
        
        # 4. 性能指标饼图
        if checkpoint_data.get('best_acc', 0) > 0:
            # 创建性能指标饼图
            labels = ['Accuracy', 'Error Rate']
            sizes = [checkpoint_data.get('best_acc', 0), 100 - checkpoint_data.get('best_acc', 0)]
            colors = ['lightgreen', 'lightcoral']
            explode = (0.1, 0)  # 突出显示准确率
            
            ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax4.set_title('Final Performance', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No performance data available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Final Performance', fontweight='bold')
            ax4.axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training summary saved: {save_path}")
    
    def save_training_log(self, 
                         checkpoint_data: Dict,
                         log_name: str = "training_log.json"):
        """
        保存训练日志为JSON文件
        """
        log_path = os.path.join(self.save_dir, log_name)
        
        # 准备日志数据
        log_data = {
            'training_summary': {
                'timestamp': checkpoint_data.get('timestamp', 'N/A'),
                'final_epoch': checkpoint_data.get('epoch', 0),
                'best_loss': checkpoint_data.get('best_loss', 0),
                'best_accuracy': checkpoint_data.get('best_acc', 0),
                'training_time': checkpoint_data.get('training_time', 'N/A'),
                'model_config': {
                    'input_size': checkpoint_data.get('input_size', 448),
                    'batch_size': checkpoint_data.get('batch_size', 16),
                    'learning_rate': checkpoint_data.get('learning_rate', 0.001),
                    'num_classes': checkpoint_data.get('num_classes', 20)
                }
            },
            'training_history': {
                'train_losses': checkpoint_data.get('train_losses', []),
                'val_losses': checkpoint_data.get('val_losses', []),
                'train_accuracies': checkpoint_data.get('train_accuracies', []),
                'val_accuracies': checkpoint_data.get('val_accuracies', []),
                'learning_rates': checkpoint_data.get('learning_rates', [])
            }
        }
        
        # 保存JSON文件
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
        
        print(f"Training log saved: {log_path}")


def create_visualizer(save_dir: str = './visualizations'):
    """
    创建YOLO训练可视化器实例
    """
    return YOLOTrainingVisualizer(save_dir)


if __name__ == "__main__":
    # 示例用法
    print("YOLO Visualization Module Test (English Version)")
    
    # 创建可视化器
    visualizer = create_visualizer('./test_visualizations')
    
    # 测试训练曲线
    test_train_losses = [0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.22]
    test_val_losses = [0.9, 0.7, 0.6, 0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.32]
    test_train_maps = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.52, 0.55, 0.57, 0.58]
    test_val_maps = [0.08, 0.18, 0.28, 0.38, 0.42, 0.48, 0.49, 0.52, 0.54, 0.55]
    
    visualizer.plot_training_curves(
        test_train_losses, test_val_losses, 
        train_maps=test_train_maps, val_maps=test_val_maps,
        title="Test Training Curves"
    )
    
    # 测试IoU分布
    test_ious = np.random.beta(2, 2, 1000)
    visualizer.plot_iou_distribution(test_ious, title="Test IoU Distribution")
    
    print("Test completed!")
