"""
SwinYOLO训练脚本
简化的训练流程，专注于Swin Transformer + YOLO检测
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import json
import numpy as np

from SwinYOLO import create_swin_yolo
from dataset import VOC_Detection_Set  # 使用现有的数据集
from evaluation import evaluate_model, decode_predictions, apply_nms_to_detections
from visualization import (plot_training_curves, plot_loss_components, plot_map_curves, 
                          create_training_summary_plot, plot_class_wise_performance)


def custom_collate_fn(batch):
    """自定义的collate函数，正确处理[gt, mask_pos, mask_neg]格式的targets"""
    images = []
    targets = []
    
    for sample in batch:
        img, target_data = sample
        images.append(img)
        
        # 检查target_data的类型
        if isinstance(target_data, list) and len(target_data) == 3:
            # 真实数据集格式: [gt, mask_pos, mask_neg]
            targets.append(target_data)
        elif isinstance(target_data, torch.Tensor):
            # 虚拟数据集格式: 直接是一个张量
            # 将其转换为兼容格式
            targets.append(target_data)
        else:
            targets.append(target_data)
    
    # 将图像堆叠为批处理张量
    images = torch.stack(images)
    
    return images, targets


class SwinYOLOTrainer:
    """SwinYOLO训练器"""
    
    def __init__(self, 
                 num_classes=20,
                 input_size=448,
                 grid_size=7,
                 device='cuda',
                 save_dir='./checkpoints/swin_yolo'):
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.grid_size = grid_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建模型和损失函数
        self.model, self.loss_fn = create_swin_yolo(
            num_classes=num_classes,
            input_size=input_size,
            grid_size=grid_size
        )
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 详细损失记录
        self.train_losses_detailed = {
            'total_loss': [],
            'coord_loss': [],
            'conf_loss': [],
            'class_loss': []
        }
        self.val_losses_detailed = {
            'total_loss': [],
            'coord_loss': [],
            'conf_loss': [],
            'class_loss': []
        }
        
        # mAP记录
        self.map_history = []
        self.map_epochs = []
        
        print(f"✅ SwinYOLO训练器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   保存目录: {self.save_dir}")
    
    def create_optimizer(self, base_lr=0.001, backbone_lr_ratio=0.1, weight_decay=0.0005):
        """创建优化器，使用不同的学习率"""
        param_groups = self.model.get_parameter_groups(base_lr, backbone_lr_ratio)
        
        optimizer = optim.Adam([
            {'params': group['params'], 'lr': group['lr']} 
            for group in param_groups
        ], weight_decay=weight_decay)
        
        return optimizer
    
    def create_scheduler(self, optimizer, epochs):
        """创建学习率调度器"""
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=1e-6
        )
        return scheduler
    
    def train_epoch(self, train_loader, optimizer, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_coord_loss = 0.0
        total_conf_loss = 0.0
        total_class_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'训练 Epoch {epoch+1}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # 处理targets：支持两种格式
            if isinstance(targets, list) and len(targets) > 0:
                # 检查第一个target的格式来确定数据类型
                first_target = targets[0]
                
                if isinstance(first_target, list) and len(first_target) == 3:
                    # 真实数据集格式: [gt, mask_pos, mask_neg]
                    batch_gt = []
                    batch_mask_pos = []
                    batch_mask_neg = []
                    
                    for target in targets:
                        gt, mask_pos, mask_neg = target
                        # 确保gt是tensor
                        if isinstance(gt, np.ndarray):
                            gt = torch.from_numpy(gt).float()
                        batch_gt.append(gt)
                        batch_mask_pos.append(mask_pos)
                        batch_mask_neg.append(mask_neg)
                    
                    # 堆叠为批处理张量
                    targets = [
                        torch.stack(batch_gt),
                        torch.stack(batch_mask_pos), 
                        torch.stack(batch_mask_neg)
                    ]
                    
                elif isinstance(first_target, torch.Tensor):
                    # 虚拟数据集格式: 直接是张量
                    targets = torch.stack(targets)
                else:
                    raise ValueError(f"不支持的target格式: {type(first_target)}")
            
            # 将targets移动到设备
            if isinstance(targets, list):
                targets = [t.to(self.device) for t in targets]
            else:
                targets = targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            predictions = self.model(images)
            
            # 计算损失
            # targets现在是[batch_gt, batch_mask_pos, batch_mask_neg]的列表
            if isinstance(targets, list) and len(targets) >= 1:
                # 使用gt作为主要目标，mask用于其他目的
                gt_targets = targets[0]  # batch_gt
                loss_dict = self.loss_fn(predictions, gt_targets)
            else:
                loss_dict = self.loss_fn(predictions, targets)
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_coord_loss += loss_dict['coord_loss'].item()
            total_conf_loss += loss_dict['conf_loss'].item()
            total_class_loss += loss_dict['class_loss'].item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Coord': f'{loss_dict["coord_loss"].item():.4f}',
                'Conf': f'{loss_dict["conf_loss"].item():.4f}',
                'Class': f'{loss_dict["class_loss"].item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_coord_loss = total_coord_loss / len(train_loader)
        avg_conf_loss = total_conf_loss / len(train_loader)
        avg_class_loss = total_class_loss / len(train_loader)
        
        return {
            'total_loss': avg_loss,
            'coord_loss': avg_coord_loss,
            'conf_loss': avg_conf_loss,
            'class_loss': avg_class_loss
        }
    
    def validate_epoch(self, val_loader, epoch):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_coord_loss = 0.0
        total_conf_loss = 0.0
        total_class_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'验证 Epoch {epoch+1}')
            
            for images, targets in progress_bar:
                images = images.to(self.device)
                
                # 处理targets：支持两种格式
                if isinstance(targets, list) and len(targets) > 0:
                    # 检查第一个target的格式来确定数据类型
                    first_target = targets[0]
                    
                    if isinstance(first_target, list) and len(first_target) == 3:
                        # 真实数据集格式: [gt, mask_pos, mask_neg]
                        batch_gt = []
                        batch_mask_pos = []
                        batch_mask_neg = []
                        
                        for target in targets:
                            gt, mask_pos, mask_neg = target
                            # 确保gt是tensor
                            if isinstance(gt, np.ndarray):
                                gt = torch.from_numpy(gt).float()
                            batch_gt.append(gt)
                            batch_mask_pos.append(mask_pos)
                            batch_mask_neg.append(mask_neg)
                        
                        # 堆叠为批处理张量
                        targets = [
                            torch.stack(batch_gt),
                            torch.stack(batch_mask_pos), 
                            torch.stack(batch_mask_neg)
                        ]
                        
                    elif isinstance(first_target, torch.Tensor):
                        # 虚拟数据集格式: 直接是张量
                        targets = torch.stack(targets)
                    else:
                        raise ValueError(f"不支持的target格式: {type(first_target)}")
                
                # 将targets移动到设备
                if isinstance(targets, list):
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                predictions = self.model(images)
                
                # 计算损失
                if isinstance(targets, list) and len(targets) >= 1:
                    # 使用gt作为主要目标
                    gt_targets = targets[0]  # batch_gt
                    loss_dict = self.loss_fn(predictions, gt_targets)
                else:
                    loss_dict = self.loss_fn(predictions, targets)
                
                total_loss += loss_dict['total_loss'].item()
                total_coord_loss += loss_dict['coord_loss'].item()
                total_conf_loss += loss_dict['conf_loss'].item()
                total_class_loss += loss_dict['class_loss'].item()
                
                progress_bar.set_postfix({
                    'Val Loss': f'{loss_dict["total_loss"].item():.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_coord_loss = total_coord_loss / len(val_loader)
        avg_conf_loss = total_conf_loss / len(val_loader)
        avg_class_loss = total_class_loss / len(val_loader)
        
        return {
            'total_loss': avg_loss,
            'coord_loss': avg_coord_loss,
            'conf_loss': avg_conf_loss,
            'class_loss': avg_class_loss
        }
    
    def evaluate_map(self, val_loader, conf_threshold=0.1, iou_threshold=0.5):
        """评估模型的mAP指标"""
        print("📊 开始评估mAP...")
        
        map_results = evaluate_model(
            model=self.model,
            dataloader=val_loader,
            device=self.device,
            num_classes=self.num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        print(f"✅ mAP评估完成:")
        print(f"   mAP@0.5: {map_results['mAP']:.4f}")
        
        # 显示前5个类别的AP
        class_aps = [(k, v) for k, v in map_results.items() if k.startswith('class_')]
        class_aps.sort(key=lambda x: x[1], reverse=True)
        
        print("   Top 5 类别AP:")
        for i, (class_name, ap) in enumerate(class_aps[:5]):
            class_id = int(class_name.split('_')[1])
            print(f"     类别{class_id}: {ap:.4f}")
        
        return map_results
    
    def plot_real_time_results(self, current_epoch):
        """绘制实时训练结果"""
        print(f"📊 生成Epoch {current_epoch}的训练图表...")
        
        vis_dir = os.path.join(self.save_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 绘制训练曲线
        if len(self.train_losses) > 1:
            train_curve_path = os.path.join(vis_dir, f'training_curves_epoch_{current_epoch}.png')
            plot_training_curves(
                self.train_losses, 
                self.val_losses,
                self.learning_rates if self.learning_rates else None,
                save_path=train_curve_path,
                title=f"SwinYOLO训练曲线 (Epoch {current_epoch})"
            )
        
        # 绘制损失组件
        if len(self.train_losses_detailed['total_loss']) > 1:
            components_path = os.path.join(vis_dir, f'loss_components_epoch_{current_epoch}.png')
            plot_loss_components(
                self.train_losses_detailed,
                self.val_losses_detailed,
                save_path=components_path,
                title=f"SwinYOLO损失组件 (Epoch {current_epoch})"
            )
        
        # 绘制mAP曲线
        if len(self.map_history) > 1:
            map_path = os.path.join(vis_dir, f'map_curves_epoch_{current_epoch}.png')
            plot_map_curves(
                self.map_history,
                save_path=map_path,
                title=f"SwinYOLO mAP曲线 (Epoch {current_epoch})"
            )
            
            # 最新的类别性能分析
            latest_map = self.map_history[-1]
            class_perf_path = os.path.join(vis_dir, f'class_performance_epoch_{current_epoch}.png')
            plot_class_wise_performance(
                latest_map,
                save_path=class_perf_path,
                title=f"各类别性能分析 (Epoch {current_epoch})"
            )
    
    def create_final_visualization(self):
        """创建训练完成后的完整可视化"""
        print("🎨 创建最终训练可视化...")
        
        vis_dir = os.path.join(self.save_dir, 'final_results')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 最终训练曲线
        final_train_path = os.path.join(vis_dir, 'final_training_curves.png')
        plot_training_curves(
            self.train_losses, 
            self.val_losses,
            self.learning_rates if self.learning_rates else None,
            save_path=final_train_path,
            title="SwinYOLO最终训练曲线"
        )
        
        # 最终损失组件
        final_components_path = os.path.join(vis_dir, 'final_loss_components.png')
        plot_loss_components(
            self.train_losses_detailed,
            self.val_losses_detailed,
            save_path=final_components_path,
            title="SwinYOLO最终损失组件"
        )
        
        # 最终mAP曲线
        if self.map_history:
            final_map_path = os.path.join(vis_dir, 'final_map_curves.png')
            plot_map_curves(
                self.map_history,
                save_path=final_map_path,
                title="SwinYOLO最终mAP曲线"
            )
            
            # 最终类别性能
            final_class_path = os.path.join(vis_dir, 'final_class_performance.png')
            final_map = self.map_history[-1]
            plot_class_wise_performance(
                final_map,
                save_path=final_class_path,
                title="最终各类别性能分析"
            )
            
            print(f"🏆 最佳mAP: {max(result['mAP'] for result in self.map_history):.4f}")
        
        return vis_dir
    
    def save_checkpoint(self, epoch, optimizer, scheduler, best_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_loss': best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': {
                'num_classes': self.num_classes,
                'input_size': self.input_size,
                'grid_size': self.grid_size
            }
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ 最佳模型已保存: {best_path}")
    
    def load_checkpoint(self, checkpoint_path, optimizer=None, scheduler=None):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        return checkpoint['epoch'], checkpoint.get('best_loss', float('inf'))
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs=100,
              base_lr=0.001,
              backbone_lr_ratio=0.1,
              weight_decay=0.0005,
              resume_from=None,
              freeze_backbone_epochs=10):
        """
        训练SwinYOLO
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            base_lr: 基础学习率
            backbone_lr_ratio: backbone学习率比例
            weight_decay: 权重衰减
            resume_from: 恢复训练的检查点路径
            freeze_backbone_epochs: 冻结backbone的轮数
        """
        print("="*60)
        print("🚀 开始SwinYOLO训练")
        print("="*60)
        
        # 创建优化器和调度器
        optimizer = self.create_optimizer(base_lr, backbone_lr_ratio, weight_decay)
        scheduler = self.create_scheduler(optimizer, epochs)
        
        # 恢复训练
        start_epoch = 0
        best_loss = float('inf')
        
        if resume_from and os.path.exists(resume_from):
            print(f"📁 从检查点恢复训练: {resume_from}")
            start_epoch, best_loss = self.load_checkpoint(resume_from, optimizer, scheduler)
        
        # 前几个epoch冻结backbone
        if start_epoch < freeze_backbone_epochs:
            self.model.freeze_backbone()
            print(f"🔒 前{freeze_backbone_epochs}个epoch将冻结Swin backbone")
        
        # 训练循环
        for epoch in range(start_epoch, epochs):
            # 解冻backbone
            if epoch == freeze_backbone_epochs:
                self.model.unfreeze_backbone()
                print(f"🔓 Epoch {epoch}: 解冻Swin backbone，开始端到端训练")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # 更新学习率
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                self.learning_rates.append(current_lr)
            
            # 记录损失
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['total_loss'])
            
            # 记录详细损失
            for key in self.train_losses_detailed.keys():
                self.train_losses_detailed[key].append(train_metrics[key])
                self.val_losses_detailed[key].append(val_metrics[key])
            
            # 打印结果
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练 - 总损失: {train_metrics['total_loss']:.4f}, "
                  f"坐标: {train_metrics['coord_loss']:.4f}, "
                  f"置信度: {train_metrics['conf_loss']:.4f}, "
                  f"分类: {train_metrics['class_loss']:.4f}")
            print(f"  验证 - 总损失: {val_metrics['total_loss']:.4f}, "
                  f"坐标: {val_metrics['coord_loss']:.4f}, "
                  f"置信度: {val_metrics['conf_loss']:.4f}, "
                  f"分类: {val_metrics['class_loss']:.4f}")
            if scheduler:
                print(f"  学习率: {current_lr:.6f}")
            
            # 保存检查点
            is_best = val_metrics['total_loss'] < best_loss
            if is_best:
                best_loss = val_metrics['total_loss']
            
            # 每5个epoch进行mAP评估 (更频繁)
            if (epoch + 1) % 5 == 0:
                map_results = self.evaluate_map(val_loader, conf_threshold=0.1, iou_threshold=0.5)
                self.map_history.append(map_results)
                self.map_epochs.append(epoch + 1)
                print(f"  Epoch {epoch+1} mAP: {map_results['mAP']:.4f}")
                
                # 每10个epoch生成实时图表
                if (epoch + 1) % 10 == 0:
                    self.plot_real_time_results(epoch + 1)
            
            # 每10个epoch保存一次
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(epoch + 1, optimizer, scheduler, best_loss, is_best)
            
            print()
        
        print("🎉 训练完成!")
        print(f"最佳验证损失: {best_loss:.4f}")
        
        # 创建最终可视化
        final_vis_dir = self.create_final_visualization()
        
        # 保存完整训练历史
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'train_losses_detailed': self.train_losses_detailed,
                'val_losses_detailed': self.val_losses_detailed,
                'map_history': self.map_history,
                'map_epochs': self.map_epochs,
                'best_loss': best_loss,
                'final_visualization_dir': final_vis_dir
            }, f, indent=2)
        
        print(f"📊 训练历史已保存到: {history_path}")
        print(f"🎨 最终可视化图表已保存到: {final_vis_dir}")


def main():
    """主训练函数"""
    # 配置参数
    config = {
        'num_classes': 20,  # VOC数据集
        'input_size': 448,
        'grid_size': 7,
        'batch_size': 16,  # 根据GPU内存调整
        'epochs': 100,
        'base_lr': 0.001,
        'backbone_lr_ratio': 0.1,
        'weight_decay': 0.0005,
        'freeze_backbone_epochs': 10
    }
    
    # 创建训练器
    trainer = SwinYOLOTrainer(
        num_classes=config['num_classes'],
        input_size=config['input_size'],
        grid_size=config['grid_size']
    )
    
    # 数据集配置（需要根据实际路径调整）
    try:
        # 加载完整数据集
        full_dataset = VOC_Detection_Set(
            voc2012_jpeg_dir='../../data/VOC2012/VOCdevkit/VOC2012/JPEGImages',
            voc2012_anno_dir='../../data/VOC2012/VOCdevkit/VOC2012/Annotations', 
            class_file='voc_classes.txt',
            input_size=config['input_size'],
            grid_size=config['grid_size'],
            is_train=True
        )
        
        # 分割训练集和验证集（80%训练，20%验证）
        from torch.utils.data import random_split
        
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子保证可重复性
        )
        
        print(f"✅ 数据集加载成功")
        print(f"   总样本数: {total_size}")
        print(f"   训练集: {len(train_dataset)} 样本 ({train_size/total_size*100:.1f}%)")
        print(f"   验证集: {len(val_dataset)} 样本 ({val_size/total_size*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("使用虚拟数据进行演示...")
        
        # 创建虚拟数据集
        from torch.utils.data import TensorDataset
        
        # 虚拟训练集（80%）
        train_images = torch.randn(800, 3, config['input_size'], config['input_size'])
        train_targets = torch.randn(800, config['grid_size'], config['grid_size'], 
                                   2*5 + config['num_classes'])  # 应该是30维：10(边界框) + 20(类别)
        train_dataset = TensorDataset(train_images, train_targets)
        
        # 虚拟验证集（20%）
        val_images = torch.randn(200, 3, config['input_size'], config['input_size'])
        val_targets = torch.randn(200, config['grid_size'], config['grid_size'],
                                 2*5 + config['num_classes'])  # 应该是30维：10(边界框) + 20(类别)
        val_dataset = TensorDataset(val_images, val_targets)
        
        print(f"✅ 虚拟数据集创建成功")
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   验证集: {len(val_dataset)} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    # 开始训练
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        base_lr=config['base_lr'],
        backbone_lr_ratio=config['backbone_lr_ratio'],
        weight_decay=config['weight_decay'],
        freeze_backbone_epochs=config['freeze_backbone_epochs']
    )


if __name__ == "__main__":
    main()
