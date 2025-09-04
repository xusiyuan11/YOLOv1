"""
Stage 1: Backbone Classification Training
阶段1：骨干网络分类预训练
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import numpy as np
from typing import Dict

from NetModel import create_classification_model
from OPT import create_adam_optimizer
from Utils import load_hyperparameters, save_checkpoint, load_checkpoint
from dataset import COCO_Segmentation_Classification_Set


class BackboneClassificationTrainer:
    """骨干网络分类器训练器"""
    
    def __init__(self, 
                 hyperparameters: Dict = None,
                 save_dir: str = './checkpoints/classification'):
        """
        初始化分类训练器
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
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"骨干网络分类训练器初始化完成")
        print(f"设备: {self.device}")
        print(f"保存目录: {self.save_dir}")
    
    def create_datasets(self, coco_config: Dict):
        """创建COCO分类数据集"""
        print("正在加载COCO分割分类数据集...")
        
        try:
            # 训练集
            train_dataset = COCO_Segmentation_Classification_Set(
                imgs_path=coco_config['imgs_path'],
                coco_json=coco_config['coco_json'],
                input_size=self.hyperparameters.get('input_size', 448),
                min_area=coco_config.get('min_area', 1000),
                max_objects_per_image=coco_config.get('max_objects_per_image', 10)
            )
            
            # 分割训练集和验证集
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            print(f"训练集样本数: {len(train_dataset)}")
            print(f"验证集样本数: {len(val_dataset)}")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            print(f"COCO数据集加载失败: {e}")
            print("使用虚拟数据进行演示...")
            
            # 创建虚拟数据集
            from torch.utils.data import TensorDataset
            input_size = self.hyperparameters.get('input_size', 448)
            
            # 训练集
            train_images = torch.randn(800, 3, input_size, input_size)
            train_labels = torch.randint(0, 80, (800,))
            train_dataset = TensorDataset(train_images, train_labels)
            
            # 验证集
            val_images = torch.randn(200, 3, input_size, input_size)
            val_labels = torch.randint(0, 80, (200,))
            val_dataset = TensorDataset(val_images, val_labels)
            
            return train_dataset, val_dataset
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'分类训练 Epoch {epoch+1}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            accuracy = 100.0 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, model, val_loader, criterion, epoch):
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'分类验证 Epoch {epoch+1}')
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                predictions = model(images)
                loss = criterion(predictions, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                accuracy = 100.0 * correct / total
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Val Acc': f'{accuracy:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              coco_config: Dict,
              epochs: int = 30,
              resume_from: str = None):
        """
        训练分类器
        Args:
            coco_config: COCO数据配置
            epochs: 训练轮数
            resume_from: 恢复训练的checkpoint路径
        """
        print("="*60)
        print("阶段1：骨干网络分类预训练")
        print("="*60)
        
        # 创建数据集
        train_dataset, val_dataset = self.create_datasets(coco_config)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=coco_config.get('batch_size', 16),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=coco_config.get('batch_size', 16),
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 创建分类模型
        input_size = self.hyperparameters.get('input_size', 448)
        use_efficient = self.hyperparameters.get('use_efficient_backbone', True)
        num_classes = coco_config.get('num_classes', 80)
        
        model = create_classification_model(
            num_classes=num_classes,  # 修正参数名
            input_size=input_size,
            use_efficient_backbone=use_efficient
        ).to(self.device)
        
        print(f"分类模型创建完成")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"类别数: {num_classes}")
        
        # 创建损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = create_adam_optimizer(
            model,
            learning_rate=coco_config.get('learning_rate', 0.001),
            weight_decay=self.hyperparameters.get('weight_decay', 0.0005)
        )
        
        # 恢复训练
        start_epoch = 0
        best_acc = 0.0
        
        if resume_from and os.path.exists(resume_from):
            print(f"从 {resume_from} 恢复训练...")
            checkpoint = load_checkpoint(resume_from)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('best_acc', 0.0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.train_accuracies = checkpoint.get('train_accuracies', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        # 训练循环
        print(f"开始训练，共 {epochs} 个epoch")
        
        for epoch in range(start_epoch, epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion, epoch)
            
            # 记录统计信息
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 打印epoch结果
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = os.path.join(self.save_dir, 'best_classification_model.pth')
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    'val_losses': self.val_losses,
                    'val_accuracies': self.val_accuracies,
                    'hyperparameters': self.hyperparameters
                }, best_model_path)
                print(f"  新的最佳模型已保存: {best_model_path}")
            
            print()  # 空行分隔
        
        # 保存训练好的backbone
        backbone_path = os.path.join(self.save_dir, 'trained_backbone.pth')
        model.save_trained_backbone(backbone_path)
        print(f"训练完成！最佳验证精度: {best_acc:.2f}%")
        print(f"训练好的backbone已保存至: {backbone_path}")
        
        return backbone_path, best_acc


def main():
    """主函数：单独运行分类训练"""
    # 加载配置
    hyperparameters = load_hyperparameters()
    
    # COCO数据配置
    coco_config = {
        'imgs_path': '../data/COCO/train2017',
        'coco_json': '../data/COCO/annotations/instances_train2017.json',
        'batch_size': 16,
        'learning_rate': 0.001,
        'min_area': 1000,
        'max_objects_per_image': 10,
        'num_classes': 80
    }
    
    # 创建训练器
    trainer = BackboneClassificationTrainer(
        hyperparameters=hyperparameters,
        save_dir='./checkpoints/classification'
    )
    
    # 开始训练
    backbone_path, best_acc = trainer.train(
        coco_config=coco_config,
        epochs=30,
        resume_from=None  # 如果要恢复训练，指定checkpoint路径
    )
    
    print(f"分类训练完成！")
    print(f"最佳精度: {best_acc:.2f}%")
    print(f"Backbone路径: {backbone_path}")


if __name__ == "__main__":
    main()
