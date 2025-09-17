import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import matplotlib.pyplot as plt

from NetModel import create_detection_model
from YOLOLoss import YOLOLoss
from OPT import create_yolo_optimizer, create_adam_optimizer
from Utils import load_hyperparameters, save_checkpoint, load_checkpoint
from dataset import VOC_Detection_Set
from visualization import YOLOTrainingVisualizer


class DetectionTrainer:
    
    def __init__(self, 
                 hyperparameters: Dict = None,
                 save_dir: str = './checkpoints/detection',
                 enable_visualization: bool = True):

        if hyperparameters is None:
            hyperparameters = load_hyperparameters()
        self.hyperparameters = hyperparameters
        self.save_dir = save_dir
        self.enable_visualization = enable_visualization
        
        self.device = torch.device(hyperparameters.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        os.makedirs(save_dir, exist_ok=True)
        
        if self.enable_visualization:
            self.visualizer = YOLOTrainingVisualizer(save_dir=os.path.join(save_dir, 'visualizations'))
        else:
            self.visualizer = None
        
        self.train_losses = []
        self.val_losses = []
        self.train_maps = []
        self.val_maps = []
        
        self.map_calculation_config = {
            'calculate_interval': 2,
            'fast_mode': True,
            'sample_ratio': 0.3,
            'confidence_threshold': 0.1,
            'max_detections': 100,
        }
        
        print(f"检测头训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"保存目录: {self.save_dir}")
        print(f"可视化: {'启用' if self.enable_visualization else '禁用'}")
        print(f"mAP计算优化: 间隔{self.map_calculation_config['calculate_interval']}轮, 快速模式: {self.map_calculation_config['fast_mode']}")
    
    def configure_map_calculation(self, calculate_interval=None, fast_mode=None, 
                                 sample_ratio=None, confidence_threshold=None, max_detections=None):
        if calculate_interval is not None:
            self.map_calculation_config['calculate_interval'] = calculate_interval
        if fast_mode is not None:
            self.map_calculation_config['fast_mode'] = fast_mode
        if sample_ratio is not None:
            self.map_calculation_config['sample_ratio'] = sample_ratio
        if confidence_threshold is not None:
            self.map_calculation_config['confidence_threshold'] = confidence_threshold
        if max_detections is not None:
            self.map_calculation_config['max_detections'] = max_detections
            
        print(f" mAP计算配置已更新:")
        for key, value in self.map_calculation_config.items():
            print(f"  {key}: {value}")
    
    def create_datasets(self, voc_config: Dict):

        voc2012_jpeg_dir = voc_config['voc2012_jpeg_dir']
        voc2012_anno_dir = voc_config['voc2012_anno_dir']
        class_file = voc_config.get('class_file', './voc_classes.txt')

        input_size = self.hyperparameters.get('input_size', 448)
        grid_count = 7
        
        print(f"数据集配置: input_size={input_size}, grid_size={grid_count}")
        
        full_dataset = VOC_Detection_Set(
            voc2012_jpeg_dir=voc2012_jpeg_dir,
            voc2012_anno_dir=voc2012_anno_dir,
            class_file=class_file,
            input_size=input_size,
            grid_size=grid_count
        )
        
        total_size = len(full_dataset)
        train_size = int(0.6 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"总样本数: {total_size}")
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset
    
    def plot_training_progress(self, epoch):
        if not self.enable_visualization or self.visualizer is None:
            return
        
        if len(self.train_losses) > 0:
            self.visualizer.plot_training_curves(
                train_losses=self.train_losses,
                val_losses=self.val_losses,
                train_maps=self.train_maps if self.train_maps else None,
                val_maps=self.val_maps if self.val_maps else None,
                title=f"YOLO训练进度 - Epoch {epoch}",
                save_name=f"training_progress_epoch_{epoch}.png"
            )
    
    def create_training_summary(self):
        if not self.enable_visualization or self.visualizer is None:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        ax1.plot(epochs, self.train_losses, 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if self.train_maps and self.val_maps:
            ax2.plot(epochs, self.train_maps, 'g-', label='训练mAP', linewidth=2)
            ax2.plot(epochs, self.val_maps, 'orange', label='验证mAP', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mAP')
            ax2.set_title('平均精度均值(mAP)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'mAP数据不可用', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('mAP进度')
        
        best_val_map = max(self.val_maps) if self.val_maps else 0
        final_train_loss = self.train_losses[-1] if self.train_losses else 0
        final_val_loss = self.val_losses[-1] if self.val_losses else 0
        
        stats_text = f"""训练总结报告
        
最佳验证mAP: {best_val_map:.4f}
最终训练损失: {final_train_loss:.4f}
最终验证损失: {final_val_loss:.4f}
训练轮数: {len(self.train_losses)}
输入尺寸: {self.hyperparameters.get('input_size', 448)}
网格大小: {self.hyperparameters.get('grid_size', 7)}

        """
        
        ax3.text(0.1, 0.9, stats_text, fontsize=11, transform=ax3.transAxes, 
                verticalalignment='top', fontfamily='monospace')
        ax3.set_title('训练统计')
        ax3.axis('off')
        
        if len(self.val_losses) > 1:
            loss_improvement = [(self.val_losses[0] - val_loss) for val_loss in self.val_losses]
            ax4.plot(epochs, loss_improvement, 'purple', label='损失改进', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('损失改进量')
            ax4.set_title('训练改进趋势')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        summary_path = os.path.join(self.visualizer.save_dir, 'training_summary.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"训练总结报告已保存至: {summary_path}")
    
    def calculate_map(self, model, val_loader, fast_mode=None):
        if fast_mode is None:
            fast_mode = self.map_calculation_config['fast_mode']
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        if fast_mode:
            sample_ratio = self.map_calculation_config['sample_ratio']
            max_batches = max(1, int(len(val_loader) * sample_ratio))
            print(f"  快速mAP模式: 使用{max_batches}/{len(val_loader)}批数据")
        else:
            max_batches = len(val_loader)
        
        processed_batches = 0
        with torch.no_grad():
            for images, targets in val_loader:
                if processed_batches >= max_batches:
                    break
                    
                images = images.to(self.device)
                if isinstance(targets, list) and len(targets) == 3:
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                predictions = model(images)
                
                batch_size = images.size(0)
                for i in range(batch_size):
                    pred_boxes, pred_scores, pred_classes = self._decode_predictions(
                        predictions[i], 
                        confidence_threshold=self.map_calculation_config['confidence_threshold'], 
                        nms_threshold=0.4,
                        max_detections=self.map_calculation_config['max_detections']
                    )
                    
                    if isinstance(targets, list):
                        gt_boxes, gt_classes = self._decode_targets(targets[0][i])
                    else:
                        gt_boxes, gt_classes = self._decode_targets(targets[i])
                    
                    all_predictions.append({
                        'boxes': pred_boxes,
                        'scores': pred_scores, 
                        'class_ids': pred_classes
                    })
                    all_targets.append({
                        'boxes': gt_boxes,
                        'class_ids': gt_classes
                    })
                
                processed_batches += 1
        
        if len(all_predictions) == 0:
            return 0.0
            
        from Utils import calculate_map
        map_result = calculate_map(all_predictions, all_targets, num_classes=20)
        return map_result.get('mAP', 0.0)
    
    def _decode_predictions(self, predictions, confidence_threshold=0.3, nms_threshold=0.4, max_detections=None):
        if max_detections is None:
            max_detections = 100
            
        if predictions.dim() == 1:
            S = 7
            B = 2
            C = 20
            expected_size = S * S * (B * 5 + C)
            if predictions.size(0) == expected_size:
                pred = predictions.reshape(S, S, B * 5 + C)
            else:
                return [], [], []
        elif predictions.dim() == 4 and predictions.size(0) == 1:
            pred = predictions[0]
            S = pred.size(0)
            B = 2
            C = 20
        elif predictions.dim() == 3:
            pred = predictions
            S = pred.size(0)
            B = 2
            C = 20
        else:
            return [], [], []
        
        boxes = []
        scores = []
        classes = []
               
        for i in range(S):
            for j in range(S):
                for b in range(B):
                    conf_idx = b * 5 + 4
                    confidence = pred[i, j, conf_idx].item()
                    
                    if confidence > confidence_threshold:
                        x_idx = b * 5
                        y_idx = b * 5 + 1
                        w_idx = b * 5 + 2
                        h_idx = b * 5 + 3
                        
                        x = pred[i, j, x_idx].item()
                        y = pred[i, j, y_idx].item()
                        w = pred[i, j, w_idx].item()
                        h = pred[i, j, h_idx].item()
                        
                        x = (j + x) / S
                        y = (i + y) / S
                        
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        
                        class_probs = pred[i, j, B*5:B*5+C]
                        class_idx = torch.argmax(class_probs).item()
                        class_prob = class_probs[class_idx].item()
                        
                        final_score = confidence * class_prob
                        
                        boxes.append([x1, y1, x2, y2])
                        scores.append(final_score)
                        classes.append(class_idx)
        
        if len(boxes) == 0:
            return [], [], []
            
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        classes = torch.tensor(classes)
        
        keep_indices = self._apply_nms(boxes, scores, nms_threshold)
        
        if len(keep_indices) > 0:
            keep_indices = torch.tensor(keep_indices)
            
            if max_detections and len(keep_indices) > max_detections:
                sorted_indices = keep_indices[torch.argsort(scores[keep_indices], descending=True)[:max_detections]]
                keep_indices = sorted_indices
            
            return boxes[keep_indices].numpy(), scores[keep_indices].numpy(), classes[keep_indices].numpy()
        else:
            return np.array([]), np.array([]), np.array([])
    
    def _decode_targets(self, target):
        if target.dim() != 3:
            return np.array([]), np.array([])
            
        S = target.size(0)
        
        boxes = []
        classes = []
        
        for i in range(S):
            for j in range(S):
                confidence = target[i, j, 4].item()
                if confidence > 0.5:
                    x = target[i, j, 0].item()
                    y = target[i, j, 1].item()
                    w = target[i, j, 2].item()
                    h = target[i, j, 3].item()
                    
                    x = (j + x) / S
                    y = (i + y) / S
                    
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    
                    class_probs = target[i, j, 10:30]
                    class_idx = torch.argmax(class_probs).item()
                    
                    boxes.append([x1, y1, x2, y2])
                    classes.append(class_idx)
        
        if len(boxes) == 0:
            return np.array([]), np.array([])
            
        return np.array(boxes), np.array(classes)
    
    def _apply_nms(self, boxes, scores, nms_threshold):
        if len(boxes) == 0:
            return []
            
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        _, order = scores.sort(descending=True)
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
                
            xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
            yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
            xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
            yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            mask = iou <= nms_threshold
            order = order[1:][mask]
        
        return keep
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch, freeze_backbone=False):
        model.train()
        
        if freeze_backbone:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'检测训练 Epoch {epoch+1}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            
            if isinstance(targets, list) and len(targets) == 3:
                targets = [t.to(self.device) for t in targets]
            else:
                targets = targets.to(self.device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss_output = criterion(predictions, targets)
            
            if isinstance(loss_output, tuple):
                loss, loss_dict = loss_output
            else:
                loss = loss_output
                loss_dict = {}
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'Backbone': 'Frozen' if freeze_backbone else 'Unfrozen'
            })
            
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def test_final_model(self, model, test_dataset, criterion):
        print("\n" + "="*60)
        print("最终测试集评估（无数据泄露）")
        print("="*60)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images = images.to(self.device)
                
                if isinstance(targets, list) and len(targets) == 3:
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                total_loss += loss.item()
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / len(test_loader)
        
        try:
            from Utils import calculate_map
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            map_result = calculate_map(
                all_predictions, 
                all_targets,
                conf_threshold=0.05,
                nms_threshold=0.5,
                input_size=self.hyperparameters.get('input_size', 448),
                grid_size=self.hyperparameters.get('grid_size', 7),
                num_classes=self.hyperparameters.get('num_classes', 20)
            )
            test_map = map_result['mAP']
        except Exception as e:
            print(f"mAP计算失败: {e}")
            test_map = 0.0
        
        print(f"最终测试集结果:")
        print(f"  测试损失: {avg_loss:.4f}")
        print(f"  测试mAP: {test_map:.4f}")
        print("="*60)
        
        return avg_loss, test_map

    def validate_epoch(self, model, val_loader, criterion, epoch):

        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'检测验证 Epoch {epoch+1}')
            
            for images, targets in progress_bar:
                images = images.to(self.device)
                if isinstance(targets, list) and len(targets) == 3:
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                predictions = model(images)
                loss_output = criterion(predictions, targets)
                
                if isinstance(loss_output, tuple):
                    loss, loss_dict = loss_output
                    if len(progress_bar.iterable) > 0 and progress_bar.n == 1:
                        print(f"\n验证损失详情:")
                        print(f"  总损失: {loss.item():.6f}")
                        print(f"  坐标损失: {loss_dict['coord_loss']:.6f}")
                        print(f"  置信度损失(有目标): {loss_dict['conf_loss_obj']:.6f}")
                        print(f"  置信度损失(无目标): {loss_dict['conf_loss_noobj']:.6f}")
                        print(f"  分类损失: {loss_dict['class_loss']:.6f}")
                else:
                    loss = loss_output
                    
                total_loss += loss.item()
                
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        
        map_score = self.calculate_map(model, val_loader)
        
        return avg_loss, map_score
    
    def train(self, 
              voc_config: Dict,
              backbone_path: str,
              epochs: int = 50,
              freeze_epochs: int = 10,
              resume_from: str = None):

        print("="*60)
        print("阶段2：检测头训练")
        print("="*60)
        
        if not os.path.exists(backbone_path):
            raise FileNotFoundError(f"预训练backbone文件不存在: {backbone_path}")
        
        train_dataset, val_dataset, test_dataset = self.create_datasets(voc_config)
        
        self.test_dataset = test_dataset
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=voc_config.get('batch_size', 8),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=voc_config.get('batch_size', 8),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        input_size = self.hyperparameters.get('input_size', 448)
        grid_cell_size = self.hyperparameters.get('grid_size', 64)
        grid_count = input_size // grid_cell_size
        use_efficient = self.hyperparameters.get('use_efficient_backbone', True)
        num_classes = voc_config.get('num_classes', 20)
        
        model = create_detection_model(
            class_num=num_classes,
            input_size=input_size,
            grid_size=grid_count,
            use_efficient_backbone=use_efficient,
            pretrained_backbone_path=backbone_path
        ).to(self.device)
        
        print(f"检测模型创建完成")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"类别数: {num_classes}")
        print(f"预训练backbone已加载: {backbone_path}")
        
        criterion = YOLOLoss(
            grid_size=grid_count,
            num_boxes=2,
            num_classes=num_classes,
            lambda_coord=5.0,
            lambda_noobj=0.5
        )
        
        backbone_params = []
        detection_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                detection_params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': voc_config.get('backbone_lr', 0.0001)},
            {'params': detection_params, 'lr': voc_config.get('detection_lr', 0.001)}
        ], weight_decay=self.hyperparameters.get('weight_decay', 0.0005))
        
        start_epoch = 0
        best_map = 0.0
        
        if resume_from and os.path.exists(resume_from):
            print(f"从 {resume_from} 恢复训练...")
            checkpoint = load_checkpoint(resume_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_map = checkpoint.get('best_map', 0.0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_maps = checkpoint.get('train_maps', [])
            self.val_maps = checkpoint.get('val_maps', [])
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        print(f"开始训练，共 {epochs} 个epoch")
        print(f"前 {freeze_epochs} 个epoch将冻结backbone")
        
        for epoch in range(start_epoch, epochs):
            freeze_backbone = epoch < freeze_epochs
            
            train_loss = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch, freeze_backbone
            )
            
            val_loss, val_map = self.validate_epoch(model, val_loader, criterion, epoch)
            
            scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            calculate_interval = self.map_calculation_config['calculate_interval']
            if (epoch + 1) % calculate_interval == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"  🔍 计算mAP (第{epoch+1}轮)...")
                start_time = time.time()
                
                train_map = self.calculate_map(model, train_loader, fast_mode=True)  # 训练集快速mAP
                map_time = time.time() - start_time
                print(f"  ⏱️ mAP计算耗时: {map_time:.1f}秒")
                
                self.train_maps.append(train_map)
                self.val_maps.append(val_map)
            else:
                if len(self.train_maps) > 0:
                    self.train_maps.append(self.train_maps[-1])
                    self.val_maps.append(self.val_maps[-1])
                else:
                    self.train_maps.append(0.0)
                    self.val_maps.append(val_map)
                
                train_map = self.train_maps[-1]
                print(f"  ⏭️ 跳过mAP计算 (第{epoch+1}轮)")
            
            if self.enable_visualization and (epoch + 1) % 5 == 0:
                self.plot_training_progress(epoch + 1)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练 - Loss: {train_loss:.4f}, mAP: {train_map:.4f}")
            print(f"  验证 - Loss: {val_loss:.4f}, mAP: {val_map:.4f}")
            print(f"  学习率: {scheduler.get_last_lr()}")
            print(f"  Backbone: {'冻结' if freeze_backbone else '训练'}")
            
            if val_map > best_map:
                best_map = val_map
                best_model_path = os.path.join(self.save_dir, 'best_detection_model.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_map': best_map,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_maps': self.train_maps,
                    'val_maps': self.val_maps,
                    'hyperparameters': self.hyperparameters
                }, best_model_path)
                print(f"  新的最佳模型已保存: {best_model_path}")
            
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'detection_checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_map': best_map,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_maps': self.train_maps,
                    'val_maps': self.val_maps,
                    'hyperparameters': self.hyperparameters
                }, checkpoint_path)
        
        print(f"训练完成！最佳验证mAP: {best_map:.4f}")
        
        print("\n正在加载最佳模型进行最终测试...")
        best_model_path = os.path.join(self.save_dir, 'best_detection_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("已加载最佳模型")
            
            final_test_loss, final_test_map = self.test_final_model(model, self.test_dataset, criterion)
            
            final_results = {
                'validation_map': best_map,
                'final_test_map': final_test_map,
                'final_test_loss': final_test_loss,
                'note': '这是在独立测试集上的无偏估计结果'
            }
            
            import json
            results_path = os.path.join(self.save_dir, 'final_test_results.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            print(f"最终测试结果已保存到: {results_path}")
        
        if self.enable_visualization:
            self.plot_training_progress(epochs)
            self.create_training_summary()
        
        return best_map


def main():
    hyperparameters = load_hyperparameters()
    
    voc_config = {
        'voc2012_jpeg_dir': '../../data/VOC2012/VOCdevkit/VOC2012/JPEGImages',
        'voc2012_anno_dir': '../../data/VOC2012/VOCdevkit/VOC2012/Annotations',
        'batch_size': 8,
        'backbone_lr': 0.0001,
        'detection_lr': 0.001,
        'num_classes': 20
    }
    
    backbone_path = './checkpoints/classification/best_classification_model.pth'
    
    if not os.path.exists(backbone_path):
        print(f"错误：找不到预训练backbone文件: {backbone_path}")
        print("请先运行 Train_Classification.py 完成分类预训练")
        return
    
    trainer = DetectionTrainer(
        hyperparameters=hyperparameters,
        save_dir='./checkpoints/detection'
    )
    
    best_map = trainer.train(
        voc_config=voc_config,
        backbone_path=backbone_path,
        epochs=50,
        freeze_epochs=10,
        resume_from=None
    )
    
    print(f"检测训练完成！")
    if best_map is not None:
        print(f"最佳mAP: {best_map:.4f}")
    else:
        print("训练过程中出现问题，未能获得有效的mAP值")


if __name__ == "__main__":
    main()