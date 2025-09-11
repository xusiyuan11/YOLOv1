"""
SwinYOLO可视化模块
包含训练损失、mAP曲线图和性能分析可视化
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json
import os
from typing import List, Dict, Optional
import seaborn as sns

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        learning_rates: List[float] = None,
                        save_path: str = None,
                        title: str = "SwinYOLO训练曲线"):
    """
    绘制训练损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        learning_rates: 学习率列表
        save_path: 保存路径
        title: 图表标题
    """
    epochs = range(1, len(train_losses) + 1)
    
    if learning_rates:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失值')
    ax1.set_title(f'{title} - 损失曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加最佳点标记
    min_val_idx = np.argmin(val_losses)
    ax1.scatter(min_val_idx + 1, val_losses[min_val_idx], 
               color='red', s=100, marker='*', zorder=5,
               label=f'最佳验证损失: {val_losses[min_val_idx]:.4f}')
    ax1.legend()
    
    # 学习率曲线
    if learning_rates:
        ax2.plot(epochs, learning_rates, 'g-', label='学习率', linewidth=2, marker='^', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('学习率')
        ax2.set_title('学习率变化曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # 对数刻度更好显示学习率变化
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练曲线已保存到: {save_path}")
    
    plt.show()
    return fig


def plot_loss_components(train_losses_dict: Dict[str, List[float]], 
                        val_losses_dict: Dict[str, List[float]],
                        save_path: str = None,
                        title: str = "SwinYOLO损失组件"):
    """
    绘制损失组件曲线（坐标损失、置信度损失、分类损失）
    
    Args:
        train_losses_dict: 训练损失字典，包含各组件损失
        val_losses_dict: 验证损失字典，包含各组件损失
        save_path: 保存路径
        title: 图表标题
    """
    epochs = range(1, len(train_losses_dict['total_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 损失组件名称和颜色
    loss_components = {
        'total_loss': ('总损失', 'blue'),
        'coord_loss': ('坐标损失', 'green'), 
        'conf_loss': ('置信度损失', 'orange'),
        'class_loss': ('分类损失', 'purple')
    }
    
    for idx, (loss_name, (loss_label, color)) in enumerate(loss_components.items()):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        if loss_name in train_losses_dict and loss_name in val_losses_dict:
            ax.plot(epochs, train_losses_dict[loss_name], 
                   color=color, linestyle='-', label=f'训练{loss_label}', 
                   linewidth=2, marker='o', markersize=3)
            ax.plot(epochs, val_losses_dict[loss_name], 
                   color=color, linestyle='--', label=f'验证{loss_label}', 
                   linewidth=2, marker='s', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('损失值')
        ax.set_title(f'{loss_label}曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 损失组件曲线已保存到: {save_path}")
    
    plt.show()
    return fig


def plot_map_curves(map_history: List[Dict[str, float]], 
                   save_path: str = None,
                   title: str = "SwinYOLO mAP曲线",
                   top_k_classes: int = 5):
    """
    绘制mAP曲线和top-k类别AP曲线
    
    Args:
        map_history: mAP历史记录列表
        save_path: 保存路径
        title: 图表标题
        top_k_classes: 显示top-k个类别的AP
    """
    if not map_history:
        print("⚠️ 没有mAP历史数据")
        return
    
    epochs = range(1, len(map_history) + 1)
    
    # 提取mAP值
    map_values = [result['mAP'] for result in map_history]
    
    # 提取类别AP值
    class_ap_dict = {}
    for result in map_history:
        for key, value in result.items():
            if key.startswith('class_'):
                if key not in class_ap_dict:
                    class_ap_dict[key] = []
                class_ap_dict[key].append(value)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # mAP曲线
    ax1.plot(epochs, map_values, 'b-', label='mAP@0.5', linewidth=3, marker='o', markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('mAP值')
    ax1.set_title(f'{title} - mAP曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 添加最佳mAP标记
    if map_values:
        max_map_idx = np.argmax(map_values)
        ax1.scatter(max_map_idx + 1, map_values[max_map_idx], 
                   color='red', s=150, marker='*', zorder=5)
        ax1.annotate(f'最佳mAP: {map_values[max_map_idx]:.4f}', 
                    xy=(max_map_idx + 1, map_values[max_map_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Top-k类别AP曲线
    if class_ap_dict:
        # 计算每个类别的平均AP，选择top-k
        avg_aps = {k: np.mean(v) for k, v in class_ap_dict.items()}
        top_classes = sorted(avg_aps.items(), key=lambda x: x[1], reverse=True)[:top_k_classes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_classes)))
        
        for i, (class_name, _) in enumerate(top_classes):
            if class_name in class_ap_dict:
                class_id = int(class_name.split('_')[1])
                ax2.plot(epochs, class_ap_dict[class_name], 
                        color=colors[i], label=f'类别{class_id}', 
                        linewidth=2, marker='o', markersize=4)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AP值')
        ax2.set_title(f'Top-{top_k_classes} 类别AP曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ mAP曲线已保存到: {save_path}")
    
    plt.show()
    return fig


def plot_performance_comparison(results: Dict[str, Dict[str, float]], 
                              save_path: str = None,
                              title: str = "模型性能对比"):
    """
    绘制不同模型或不同阶段的性能对比图
    
    Args:
        results: 性能结果字典 {'模型名': {'mAP': 0.5, 'AP_class_0': 0.6, ...}}
        save_path: 保存路径
        title: 图表标题
    """
    if not results:
        print("⚠️ 没有性能对比数据")
        return
    
    # 提取mAP数据
    models = list(results.keys())
    map_values = [results[model].get('mAP', 0) for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # mAP柱状图
    bars = ax1.bar(models, map_values, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    ax1.set_ylabel('mAP值')
    ax1.set_title('模型mAP对比')
    ax1.set_ylim(0, 1)
    
    # 添加数值标签
    for bar, value in zip(bars, map_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 类别AP雷达图（选择第一个模型）
    if len(results) > 0:
        first_model = list(results.keys())[0]
        class_aps = []
        class_labels = []
        
        for key, value in results[first_model].items():
            if key.startswith('class_'):
                class_id = int(key.split('_')[1])
                class_aps.append(value)
                class_labels.append(f'类别{class_id}')
        
        if class_aps:
            # 只显示前8个类别的雷达图
            n_classes = min(8, len(class_aps))
            angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False).tolist()
            
            # 闭合雷达图
            class_aps_plot = class_aps[:n_classes] + [class_aps[0]]
            angles += angles[:1]
            
            ax2 = plt.subplot(122, projection='polar')
            ax2.plot(angles, class_aps_plot, 'o-', linewidth=2, label=first_model)
            ax2.fill(angles, class_aps_plot, alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(class_labels[:n_classes])
            ax2.set_ylim(0, 1)
            ax2.set_title(f'{first_model} - 类别AP雷达图')
            ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 性能对比图已保存到: {save_path}")
    
    plt.show()
    return fig


def create_training_summary_plot(history_file: str, 
                                save_dir: str = None,
                                title: str = "SwinYOLO训练总结"):
    """
    从训练历史文件创建完整的训练总结图
    
    Args:
        history_file: 训练历史JSON文件路径
        save_dir: 保存目录
        title: 图表标题
    """
    if not os.path.exists(history_file):
        print(f"⚠️ 训练历史文件不存在: {history_file}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    # 创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 绘制训练曲线
    if 'train_losses' in history and 'val_losses' in history:
        train_curve_path = os.path.join(save_dir, 'training_curves.png') if save_dir else None
        plot_training_curves(
            history['train_losses'], 
            history['val_losses'],
            history.get('learning_rates'),
            save_path=train_curve_path,
            title=title
        )
    
    # 绘制损失组件曲线
    if 'train_losses_components' in history and 'val_losses_components' in history:
        components_path = os.path.join(save_dir, 'loss_components.png') if save_dir else None
        plot_loss_components(
            history['train_losses_components'],
            history['val_losses_components'],
            save_path=components_path,
            title=title
        )
    
    # 绘制mAP曲线
    if 'map_history' in history:
        map_path = os.path.join(save_dir, 'map_curves.png') if save_dir else None
        plot_map_curves(
            history['map_history'],
            save_path=map_path,
            title=title
        )
    
    print(f"🎨 训练总结图表已生成完成！")


def save_training_gif(figures_dir: str, output_path: str, duration: int = 500):
    """
    将训练过程的图表制作成GIF动画
    
    Args:
        figures_dir: 图表文件目录
        output_path: GIF输出路径
        duration: 每帧持续时间(毫秒)
    """
    try:
        from PIL import Image
        import glob
        
        # 获取所有PNG文件
        png_files = sorted(glob.glob(os.path.join(figures_dir, '*.png')))
        
        if not png_files:
            print("⚠️ 没有找到PNG文件")
            return
        
        # 创建GIF
        images = []
        for png_file in png_files:
            img = Image.open(png_file)
            images.append(img)
        
        # 保存GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        
        print(f"🎬 训练动画已保存到: {output_path}")
        
    except ImportError:
        print("⚠️ 需要安装PIL库: pip install Pillow")
    except Exception as e:
        print(f"❌ 创建GIF失败: {e}")


# 类别名称映射（VOC数据集）
VOC_CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def plot_class_wise_performance(map_results: Dict[str, float],
                               save_path: str = None,
                               title: str = "各类别性能分析"):
    """
    绘制各类别的详细性能分析图
    
    Args:
        map_results: mAP结果字典
        save_path: 保存路径
        title: 图表标题
    """
    # 提取类别AP
    class_aps = []
    class_names = []
    
    for i in range(20):  # VOC有20个类别
        key = f'class_{i}'
        if key in map_results:
            class_aps.append(map_results[key])
            class_names.append(VOC_CLASS_NAMES[i] if i < len(VOC_CLASS_NAMES) else f'类别{i}')
    
    if not class_aps:
        print("⚠️ 没有类别AP数据")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 柱状图
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_aps)))
    bars = ax1.bar(range(len(class_aps)), class_aps, color=colors)
    ax1.set_xlabel('类别')
    ax1.set_ylabel('AP值')
    ax1.set_title(f'{title} - 各类别AP')
    ax1.set_xticks(range(len(class_aps)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, ap in zip(bars, class_aps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ap:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 添加平均线
    avg_ap = np.mean(class_aps)
    ax1.axhline(y=avg_ap, color='red', linestyle='--', linewidth=2, 
                label=f'平均AP: {avg_ap:.3f}')
    ax1.legend()
    
    # 性能分布直方图
    ax2.hist(class_aps, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('AP值范围')
    ax2.set_ylabel('类别数量')
    ax2.set_title('AP值分布直方图')
    ax2.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax2.axvline(x=avg_ap, color='red', linestyle='--', linewidth=2, label=f'平均值: {avg_ap:.3f}')
    ax2.axvline(x=np.median(class_aps), color='green', linestyle='--', linewidth=2, 
                label=f'中位数: {np.median(class_aps):.3f}')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 类别性能分析图已保存到: {save_path}")
    
    plt.show()
    return fig


if __name__ == "__main__":
    print("📊 SwinYOLO可视化模块测试")
    
    # 生成测试数据
    epochs = 50
    train_losses = [5.0 * np.exp(-0.1 * i) + 0.5 + 0.1 * np.random.randn() for i in range(epochs)]
    val_losses = [4.8 * np.exp(-0.08 * i) + 0.6 + 0.1 * np.random.randn() for i in range(epochs)]
    learning_rates = [0.001 * (0.9 ** (i // 10)) for i in range(epochs)]
    
    # 测试训练曲线
    plot_training_curves(train_losses, val_losses, learning_rates, title="测试训练曲线")
    
    print("✅ 可视化模块测试完成！")
