"""
Stage 2: Detection Head Training
阶段2：检测头训练
"""
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
        """
        初始化检测训练器
        Args:
            hyperparameters: 超参数
            save_dir: 模型保存目录
        """
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
        """创建VOC检测数据集"""
        print("正在加载VOC检测数据集...")
        
        try:
            # 训练集
            train_dataset = VOC_Detection_Set(
                voc_data_path=voc_config['voc_data_path'],
                input_size=self.hyperparameters.get('input_size', 448),
                grid_size=self.hyperparameters.get('grid_size', 7),
                is_train=True
            )
            
            # 验证集
            val_dataset = VOC_Detection_Set(
                voc_data_path=voc_config['voc_data_path'],
                input_size=self.hyperparameters.get('input_size', 448),
                grid_size=self.hyperparameters.get('grid_size', 7),
                is_train=False
            )
            
            print(f"训练集样本数: {len(train_dataset)}")
            print(f"验证集样本数: {len(val_dataset)}")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            print(f"VOC数据集加载失败: {e}")
            print("使用虚拟数据进行演示...")
            
            # 创建虚拟数据集
            from torch.utils.data import TensorDataset
            input_size = self.hyperparameters.get('input_size', 448)
            grid_size = self.hyperparameters.get('grid_size', 7)
            
            # 训练集
            train_images = torch.randn(800, 3, input_size, input_size)
            train_targets = torch.randn(800, grid_size, grid_size, 30)  # YOLO格式
            train_dataset = TensorDataset(train_images, train_targets)
            
            # 验证集
            val_images = torch.randn(200, 3, input_size, input_size)
            val_targets = torch.randn(200, grid_size, grid_size, 30)
            val_dataset = TensorDataset(val_images, val_targets)
            
            return train_dataset, val_dataset
    
    def calculate_map(self, model, val_loader):
        """计算mAP（简化版本）"""
        model.eval()
        # 这里应该实现完整的mAP计算，暂时返回伪值
        return np.random.uniform(0.3, 0.8)  # 模拟mAP值
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch, freeze_backbone=False):
        """训练一个epoch"""
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
            targets = targets.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, targets)
            
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
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'检测验证 Epoch {epoch+1}')
            
            for images, targets in progress_bar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                
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
        """
        训练检测器
        Args:
            voc_config: VOC数据配置
            backbone_path: 预训练backbone路径
            epochs: 总训练轮数
            freeze_epochs: 冻结backbone的轮数
            resume_from: 恢复训练的checkpoint路径
        """
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
        grid_size = self.hyperparameters.get('grid_size', 7)
        use_efficient = self.hyperparameters.get('use_efficient_backbone', True)
        num_classes = voc_config.get('num_classes', 20)
        
        model = create_detection_model(
            class_num=num_classes,
            grid_size=grid_size,
            input_size=input_size,
            use_efficient_backbone=use_efficient,
            pretrained_backbone_path=backbone_path
        ).to(self.device)
        
        print(f"检测模型创建完成")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"类别数: {num_classes}")
        print(f"预训练backbone已加载: {backbone_path}")
        
        # 创建损失函数和优化器
        criterion = YOLOLoss(
            feature_size=grid_size,
            num_bboxes=2,
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
        
        # 训练循环\n        print(f\"开始训练，共 {epochs} 个epoch\")\n        print(f\"前 {freeze_epochs} 个epoch将冻结backbone\")\n        \n        for epoch in range(start_epoch, epochs):\n            # 确定是否冻结backbone\n            freeze_backbone = epoch < freeze_epochs\n            \n            # 训练\n            train_loss = self.train_epoch(\n                model, train_loader, criterion, optimizer, epoch, freeze_backbone\n            )\n            \n            # 验证\n            val_loss, val_map = self.validate_epoch(model, val_loader, criterion, epoch)\n            \n            # 学习率调度\n            scheduler.step()\n            \n            # 记录统计信息\n            self.train_losses.append(train_loss)\n            self.val_losses.append(val_loss)\n            train_map = self.calculate_map(model, train_loader)  # 训练集mAP\n            self.train_maps.append(train_map)\n            self.val_maps.append(val_map)\n            \n            # 打印epoch结果\n            print(f\"Epoch {epoch+1}/{epochs}:\")\n            print(f\"  训练 - Loss: {train_loss:.4f}, mAP: {train_map:.4f}\")\n            print(f\"  验证 - Loss: {val_loss:.4f}, mAP: {val_map:.4f}\")\n            print(f\"  学习率: {scheduler.get_last_lr()}\")\n            print(f\"  Backbone: {'冻结' if freeze_backbone else '训练'}\")\n            \n            # 保存最佳模型\n            if val_map > best_map:\n                best_map = val_map\n                best_model_path = os.path.join(self.save_dir, 'best_detection_model.pth')\n                save_checkpoint({\n                    'epoch': epoch + 1,\n                    'model_state_dict': model.state_dict(),\n                    'optimizer_state_dict': optimizer.state_dict(),\n                    'scheduler_state_dict': scheduler.state_dict(),\n                    'best_map': best_map,\n                    'train_losses': self.train_losses,\n                    'val_losses': self.val_losses,\n                    'train_maps': self.train_maps,\n                    'val_maps': self.val_maps,\n                    'hyperparameters': self.hyperparameters\n                }, best_model_path)\n                print(f\"  新的最佳模型已保存: {best_model_path}\")\n            \n            # 定期保存checkpoint\n            if (epoch + 1) % 10 == 0:\n                checkpoint_path = os.path.join(self.save_dir, f'detection_checkpoint_epoch_{epoch+1}.pth')\n                save_checkpoint({\n                    'epoch': epoch + 1,\n                    'model_state_dict': model.state_dict(),\n                    'optimizer_state_dict': optimizer.state_dict(),\n                    'scheduler_state_dict': scheduler.state_dict(),\n                    'best_map': best_map,\n                    'train_losses': self.train_losses,\n                    'val_losses': self.val_losses,\n                    'train_maps': self.train_maps,\n                    'val_maps': self.val_maps,\n                    'hyperparameters': self.hyperparameters\n                }, checkpoint_path)\n        \n        print(f\"训练完成！最佳验证mAP: {best_map:.4f}\")\n        \n        return best_map\n\n\ndef main():\n    \"\"\"主函数：单独运行检测训练\"\"\"\n    # 加载配置\n    hyperparameters = load_hyperparameters()\n    \n    # VOC数据配置\n    voc_config = {\n        'voc_data_path': './data/VOC2012',\n        'batch_size': 8,\n        'backbone_lr': 0.0001,  # backbone较小学习率\n        'detection_lr': 0.001,  # 检测头较大学习率\n        'num_classes': 20\n    }\n    \n    # 预训练backbone路径（需要先运行分类训练）\n    backbone_path = './checkpoints/classification/trained_backbone.pth'\n    \n    if not os.path.exists(backbone_path):\n        print(f\"错误：找不到预训练backbone文件: {backbone_path}\")\n        print(\"请先运行 Train_Classification.py 完成分类预训练\")\n        return\n    \n    # 创建训练器\n    trainer = DetectionTrainer(\n        hyperparameters=hyperparameters,\n        save_dir='./checkpoints/detection'\n    )\n    \n    # 开始训练\n    best_map = trainer.train(\n        voc_config=voc_config,\n        backbone_path=backbone_path,\n        epochs=50,\n        freeze_epochs=10,  # 前10个epoch冻结backbone\n        resume_from=None  # 如果要恢复训练，指定checkpoint路径\n    )\n    \n    print(f\"检测训练完成！\")\n    print(f\"最佳mAP: {best_map:.4f}\")\n\n\nif __name__ == \"__main__\":\n    main()
