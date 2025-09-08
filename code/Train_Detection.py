import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

from NetModel import create_detection_model
from YOLOLoss import YOLOLoss
from OPT import create_yolo_optimizer, create_adam_optimizer
from Utils import load_hyperparameters, save_checkpoint, load_checkpoint
from dataset import VOC_Detection_Set


class DetectionTrainer:
    """检测头训练器"""
    
    def __init__(self, 
                 hyperparameters: Dict = None,
                 save_dir: str = './checkpoints/detection'):

        # 加载超参数
        if hyperparameters is None:
            hyperparameters = load_hyperparameters()
        self.hyperparameters = hyperparameters
        self.save_dir = save_dir
        
        # 设置设备
        self.device = torch.device(hyperparameters.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        self.train_maps = []
        self.val_maps = []
        
        print(f"检测头训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"保存目录: {self.save_dir}")
    
    def create_datasets(self, voc_config: Dict):

        voc2012_jpeg_dir = voc_config['voc2012_jpeg_dir']
        voc2012_anno_dir = voc_config['voc2012_anno_dir']
        class_file = voc_config.get('class_file', './voc_classes.txt')

        # 创建完整的数据集
        input_size = self.hyperparameters.get('input_size', 448)
        grid_cell_size = self.hyperparameters.get('grid_size', 64)
        grid_count = input_size // grid_cell_size  # 计算网格数量
        
        full_dataset = VOC_Detection_Set(
            voc2012_jpeg_dir=voc2012_jpeg_dir,
            voc2012_anno_dir=voc2012_anno_dir,
            class_file=class_file,
            input_size=input_size,
            grid_size=grid_count
        )
        
        # 划分训练集和验证集 (80%训练，20%验证)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子确保可复现
        )

        print(f"总样本数: {total_size}")
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")

        return train_dataset, val_dataset
    
    def calculate_map(self, model, val_loader):
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                if isinstance(targets, list) and len(targets) == 3:
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                # 获取模型预测
                predictions = model(images)
                
                # 将预测结果转换为检测框格式
                batch_size = images.size(0)
                for i in range(batch_size):
                    pred_boxes, pred_scores, pred_classes = self._decode_predictions(
                        predictions[i], confidence_threshold=0.5, nms_threshold=0.4
                    )
                    
                    # 获取真实标签
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
        
        # 计算mAP
        from Utils import calculate_map
        map_result = calculate_map(all_predictions, all_targets, num_classes=20)
        return map_result.get('mAP', 0.0)
    
    def _decode_predictions(self, predictions, confidence_threshold=0.5, nms_threshold=0.4):
        """将模型输出转换为检测框格式"""
        # 检查预测的实际形状
        if predictions.dim() == 1:
            # 如果是1维，先reshape
            # 假设输出是 (S*S*(B*5+C),) 需要reshape为 (S, S, B*5+C)
            S = 7  # grid size
            B = 2  # number of boxes per cell
            C = 20  # number of classes
            expected_size = S * S * (B * 5 + C)
            if predictions.size(0) == expected_size:
                pred = predictions.reshape(S, S, B * 5 + C)
            else:
                return [], [], []
        elif predictions.dim() == 4 and predictions.size(0) == 1:
            # shape: (1, S, S, B*5+C)
            pred = predictions[0]
            S = pred.size(0)
            B = 2
            C = 20
        elif predictions.dim() == 3:
            # shape: (S, S, B*5+C)
            pred = predictions
            S = pred.size(0)
            B = 2
            C = 20
        else:
            # 不支持的形状，返回空
            return [], [], []
        
        boxes = []
        scores = []
        classes = []
               
        for i in range(S):
            for j in range(S):
                for b in range(B):
                    # 提取置信度
                    conf_idx = b * 5 + 4
                    confidence = pred[i, j, conf_idx].item()
                    
                    if confidence > confidence_threshold:
                        # 提取边界框坐标
                        x_idx = b * 5
                        y_idx = b * 5 + 1
                        w_idx = b * 5 + 2
                        h_idx = b * 5 + 3
                        
                        x = pred[i, j, x_idx].item()
                        y = pred[i, j, y_idx].item()
                        w = pred[i, j, w_idx].item()
                        h = pred[i, j, h_idx].item()
                        
                        # 转换为绝对坐标
                        x = (j + x) / S
                        y = (i + y) / S
                        w = w * w
                        h = h * h
                        
                        # 转换为 (x1, y1, x2, y2) 格式
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        
                        # 提取类别概率
                        class_probs = pred[i, j, B*5:B*5+C]
                        class_idx = torch.argmax(class_probs).item()
                        class_prob = class_probs[class_idx].item()
                        
                        # 最终分数 = 置信度 × 类别概率
                        final_score = confidence * class_prob
                        
                        boxes.append([x1, y1, x2, y2])
                        scores.append(final_score)
                        classes.append(class_idx)
        
        if len(boxes) == 0:
            return [], [], []
            
        # 转换为张量
        boxes = torch.tensor(boxes)
        scores = torch.tensor(scores)
        classes = torch.tensor(classes)
        
        # 应用NMS
        keep_indices = self._apply_nms(boxes, scores, nms_threshold)
        
        if len(keep_indices) > 0:
            keep_indices = torch.tensor(keep_indices)
            return boxes[keep_indices].numpy(), scores[keep_indices].numpy(), classes[keep_indices].numpy()
        else:
            return np.array([]), np.array([]), np.array([])
    
    def _decode_targets(self, target):
        """将目标标签转换为检测框格式"""
        # target应该是单个样本的标签，形状为 (S, S, 5+C)
        # 如果是3维张量且已经是正确格式，直接使用
        if target.dim() == 2:
            # 如果是2维，可能需要reshape
            S = 7
            C = 20
            expected_size = S * S * (5 + C)
            if target.numel() == expected_size:
                target = target.reshape(S, S, 5 + C)
            else:
                return np.array([]), np.array([])
        elif target.dim() != 3:
            return np.array([]), np.array([])
            
        S = target.size(0)
        
        boxes = []
        classes = []
        
        for i in range(S):
            for j in range(S):
                # 检查是否有目标
                confidence = target[i, j, 4].item()
                if confidence > 0.5:  # 如果该网格有目标
                    # 提取坐标
                    x = target[i, j, 0].item()
                    y = target[i, j, 1].item()
                    w = target[i, j, 2].item()
                    h = target[i, j, 3].item()
                    
                    # 转换为绝对坐标
                    x = (j + x) / S
                    y = (i + y) / S
                    
                    # 转换为 (x1, y1, x2, y2) 格式
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    
                    # 提取类别
                    class_probs = target[i, j, 5:]
                    class_idx = torch.argmax(class_probs).item()
                    
                    boxes.append([x1, y1, x2, y2])
                    classes.append(class_idx)
        
        if len(boxes) == 0:
            return np.array([]), np.array([])
            
        return np.array(boxes), np.array(classes)
    
    def _apply_nms(self, boxes, scores, nms_threshold):
        """应用非极大值抑制"""
        if len(boxes) == 0:
            return []
            
        # 计算面积
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # 按分数排序
        _, order = scores.sort(descending=True)
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
                
            # 计算IoU
            xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
            yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
            xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
            yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # 保留IoU小于阈值的框
            mask = iou <= nms_threshold
            order = order[1:][mask]
        
        return keep
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch, freeze_backbone=False):
        model.train()
        
        # 根据设置冻结或解冻backbone
        if freeze_backbone:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'检测训练 Epoch {epoch+1}')
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # targets是包含[gt, mask_pos, mask_neg]的列表，需要分别移动到GPU
            if isinstance(targets, list) and len(targets) == 3:
                targets = [t.to(self.device) for t in targets]
            else:
                targets = targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            predictions = model(images)
            loss_output = criterion(predictions, targets)
            
            # 处理损失输出：如果是元组，提取总损失
            if isinstance(loss_output, tuple):
                loss, loss_dict = loss_output
            else:
                loss = loss_output
                loss_dict = {}
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'Backbone': 'Frozen' if freeze_backbone else 'Unfrozen'
            })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
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
                
                loss = loss_output[0] if isinstance(loss_output, tuple) else loss_output
                total_loss += loss.item()
                
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算mAP
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
        
        # 检查backbone文件
        if not os.path.exists(backbone_path):
            raise FileNotFoundError(f"预训练backbone文件不存在: {backbone_path}")
        
        # 创建数据集
        train_dataset, val_dataset = self.create_datasets(voc_config)
        
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
        
        # 创建检测模型
        input_size = self.hyperparameters.get('input_size', 448)
        grid_cell_size = self.hyperparameters.get('grid_size', 64)  # 每个网格的像素大小
        grid_count = input_size // grid_cell_size  # 实际网格数量 (448//64=7)
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
        
        # 创建损失函数和优化器
        criterion = YOLOLoss(
            grid_size=grid_count,
            num_boxes=2,
            num_classes=num_classes,
            lambda_coord=5.0,
            lambda_noobj=0.5
        )
        
        # 分组参数，为backbone和检测头设置不同学习率
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
        
        # 恢复训练
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
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # 训练循环
        print(f"开始训练，共 {epochs} 个epoch")
        print(f"前 {freeze_epochs} 个epoch将冻结backbone")
        
        for epoch in range(start_epoch, epochs):
            # 确定是否冻结backbone
            freeze_backbone = epoch < freeze_epochs
            
            # 训练
            train_loss = self.train_epoch(
                model, train_loader, criterion, optimizer, epoch, freeze_backbone
            )
            
            # 验证
            val_loss, val_map = self.validate_epoch(model, val_loader, criterion, epoch)
            
            # 学习率调度
            scheduler.step()
            
            # 记录统计信息
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            train_map = self.calculate_map(model, train_loader)  # 训练集mAP
            self.train_maps.append(train_map)
            self.val_maps.append(val_map)
            
            # 打印epoch结果
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练 - Loss: {train_loss:.4f}, mAP: {train_map:.4f}")
            print(f"  验证 - Loss: {val_loss:.4f}, mAP: {val_map:.4f}")
            print(f"  学习率: {scheduler.get_last_lr()}")
            print(f"  Backbone: {'冻结' if freeze_backbone else '训练'}")
            
            # 保存最佳模型
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
            
            # 定期保存checkpoint
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
        
        return best_map


def main():
    """主函数：单独运行检测训练"""
    # 加载配置
    hyperparameters = load_hyperparameters()
    
    # VOC数据配置
    voc_config = {
        'voc2012_jpeg_dir': '../data/VOC2012/VOCdevkit/VOC2012/JPEGImages',
        'voc2012_anno_dir': '../data/VOC2012/VOCdevkit/VOC2012/Annotations',
        'batch_size': 8,
        'backbone_lr': 0.0001,  # backbone较小学习率
        'detection_lr': 0.001,  # 检测头较大学习率
        'num_classes': 20
    }
    
    # 预训练backbone路径（需要先运行分类训练）
    backbone_path = './checkpoints/classification/best_classification_model.pth'
    
    if not os.path.exists(backbone_path):
        print(f"错误：找不到预训练backbone文件: {backbone_path}")
        print("请先运行 Train_Classification.py 完成分类预训练")
        return
    
    # 创建训练器
    trainer = DetectionTrainer(
        hyperparameters=hyperparameters,
        save_dir='./checkpoints/detection'
    )
    
    # 开始训练
    best_map = trainer.train(
        voc_config=voc_config,
        backbone_path=backbone_path,
        epochs=50,
        freeze_epochs=10,  # 前10个epoch冻结backbone
        resume_from=None  # 如果要恢复训练，指定checkpoint路径
    )
    
    print(f"检测训练完成！")
    if best_map is not None:
        print(f"最佳mAP: {best_map:.4f}")
    else:
        print("训练过程中出现问题，未能获得有效的mAP值")


if __name__ == "__main__":
    main()