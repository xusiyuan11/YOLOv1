import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import os
import time
from tqdm import tqdm
import numpy as np
from typing import Dict

from NetModel import YOLOv1
from YOLOLoss import YOLOLoss
from OPT import create_yolo_optimizer, create_adam_optimizer
from Utils import (
    load_hyperparameters, save_checkpoint, load_checkpoint
)
from dataset import VOC_Detection_Set, COCO_Segmentation_Classification_Set


class TwoStageYOLOTrainer:
    """两阶段YOLO训练器：COCO分类预训练 + VOC检测微调"""
    
    def __init__(self, 
                 hyperparameters: Dict = None,
                 save_dir: str = './checkpoints'):
        """
        初始化两阶段训练器
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
        self.stage1_losses = []
        self.stage1_accuracies = []
        self.stage2_losses = []
        self.stage2_maps = []
        
        print(f"两阶段YOLO训练器初始化完成，设备: {self.device}")
    
    def stage1_classification_pretraining(self, 
                                        coco_config: Dict,
                                        epochs: int = 30):
        """
        阶段1：COCO分割目标分类预训练
        Args:
            coco_config: COCO数据配置
            epochs: 训练轮数
        """
        print("="*60)
        print("阶段1：COCO分割目标分类预训练")
        print("="*60)
        
        # 创建COCO分类数据集
        print("正在加载COCO分割分类数据集...")
        try:
            coco_dataset = COCO_Segmentation_Classification_Set(
                imgs_path=coco_config['imgs_path'],
                coco_json=coco_config['coco_json'],
                input_size=self.hyperparameters.get('input_size', 448),
                min_area=coco_config.get('min_area', 1000),
                max_objects_per_image=coco_config.get('max_objects_per_image', 10)
            )
            
            coco_loader = DataLoader(
                coco_dataset,
                batch_size=coco_config.get('batch_size', 16),
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            
            print(f"COCO数据集加载完成，样本数: {len(coco_dataset)}")
            num_classes = coco_dataset.class_num
            
        except Exception as e:
            print(f"COCO数据集加载失败: {e}")
            print("使用虚拟数据进行演示...")
            # 创建虚拟数据集
            from torch.utils.data import TensorDataset
            input_size = self.hyperparameters.get('input_size', 448)
            dummy_images = torch.randn(1000, 3, input_size, input_size)
            dummy_labels = torch.randint(0, 80, (1000,))
            coco_dataset = TensorDataset(dummy_images, dummy_labels)
            coco_loader = DataLoader(coco_dataset, batch_size=16, shuffle=True)
            num_classes = 80
        
        # 创建分类模型
        input_size = self.hyperparameters.get('input_size', 448)
        use_efficient = self.hyperparameters.get('use_efficient_backbone', True)
        model = YOLOv1(
            class_num=num_classes,
            grid_size=input_size // self.hyperparameters.get('grid_size', 64),
            training_mode='classification',
            input_size=input_size,
            use_efficient_backbone=use_efficient
        ).to(self.device)
        
        model.set_training_mode('classification')
        
        # 创建分类损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = create_adam_optimizer(
            model,
            learning_rate=coco_config.get('learning_rate', 0.001),
            weight_decay=self.hyperparameters.get('weight_decay', 0.0005)
        )
        
        # 训练循环
        best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = tqdm(coco_loader, desc=f'阶段1 Epoch {epoch+1}/{epochs}')
            
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
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.2f}%',
                    'lr': f'{optimizer.get_current_lr():.6f}'
                })
            
            # 计算epoch统计
            avg_loss = total_loss / len(coco_loader)
            accuracy = 100.0 * correct / total
            self.stage1_losses.append(avg_loss)
            self.stage1_accuracies.append(accuracy)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            # 保存最佳模型
            if accuracy > best_acc:
                best_acc = accuracy
                stage1_path = os.path.join(self.save_dir, 'stage1_classification_best.pth')
                save_checkpoint(model, optimizer.optimizer, epoch, avg_loss, stage1_path)
                
                # 重要：单独保存backbone权重用于阶段2
                backbone_path = os.path.join(self.save_dir, 'pretrained_backbone.pth')
                model.save_backbone_weights(backbone_path)
                
                print(f"保存阶段1最佳模型，准确率: {best_acc:.2f}%")
                print(f"已保存预训练backbone权重: {backbone_path}")
            
            # 更新学习率
            optimizer.step(avg_loss)
        
        print(f"阶段1训练完成！最佳准确率: {best_acc:.2f}%")
        return os.path.join(self.save_dir, 'stage1_classification_best.pth')
    
    def stage2_detection_finetuning(self,
                                   pretrained_path: str,
                                   voc_config: Dict,
                                   epochs: int = 80):
        """
        阶段2：VOC目标检测微调
        Args:
            pretrained_path: 阶段1预训练模型路径
            voc_config: VOC数据配置
            epochs: 训练轮数
        """
        print("="*60)
        print("阶段2：VOC目标检测微调")
        print("="*60)
        
        # 创建VOC检测数据集
        print("正在加载VOC检测数据集...")
        try:
            train_datasets = []
            
            # VOC2007训练数据
            if 'voc2007_train_imgs' in voc_config:
                voc2007_train = VOC_Detection_Set(
                    imgs_path=voc_config['voc2007_train_imgs'],
                    annotations_path=voc_config['voc2007_train_annotations'],
                    class_file=voc_config['class_file'],
                    input_size=self.hyperparameters.get('input_size', 448),
                    grid_size=self.hyperparameters.get('grid_size', 64)
                )
                train_datasets.append(voc2007_train)
                print(f"VOC2007训练集: {len(voc2007_train)} 样本")
            
            # VOC2012训练数据
            if 'voc2012_train_imgs' in voc_config:
                voc2012_train = VOC_Detection_Set(
                    imgs_path=voc_config['voc2012_train_imgs'],
                    annotations_path=voc_config['voc2012_train_annotations'],
                    class_file=voc_config['class_file'],
                    input_size=self.hyperparameters.get('input_size', 448),
                    grid_size=self.hyperparameters.get('grid_size', 64)
                )
                train_datasets.append(voc2012_train)
                print(f"VOC2012训练集: {len(voc2012_train)} 样本")
            
            # 合并数据集
            if train_datasets:
                combined_dataset = ConcatDataset(train_datasets)
                train_loader = DataLoader(
                    combined_dataset,
                    batch_size=voc_config.get('batch_size', 8),
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                print(f"VOC训练数据集加载完成，总样本数: {len(combined_dataset)}")
            else:
                raise ValueError("未找到有效的VOC训练数据")
                
        except Exception as e:
            print(f"VOC数据集加载失败: {e}")
            print("使用虚拟数据进行演示...")
            # 创建虚拟检测数据集
            from torch.utils.data import TensorDataset
            input_size = self.hyperparameters.get('input_size', 448)
            grid_cells = input_size // self.hyperparameters.get('grid_size', 64)
            dummy_images = torch.randn(500, 3, input_size, input_size)
            dummy_targets = torch.randn(500, grid_cells, grid_cells, 25)
            dummy_masks = torch.ones(500, grid_cells, grid_cells, 1, dtype=torch.bool)
            combined_dataset = TensorDataset(dummy_images, dummy_targets, dummy_masks, dummy_masks)
            train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
        
        # 创建检测模型
        input_size = self.hyperparameters.get('input_size', 448)
        use_efficient = self.hyperparameters.get('use_efficient_backbone', True)
        model = YOLOv1(
            class_num=20,  # VOC有20个类别
            grid_size=input_size // self.hyperparameters.get('grid_size', 64),
            training_mode='detection',
            input_size=input_size,
            use_efficient_backbone=use_efficient
        ).to(self.device)
        
        # 加载预训练权重
        if os.path.exists(pretrained_path):
            print(f"加载预训练权重: {pretrained_path}")
            model.load_pretrained_backbone(pretrained_path)
        else:
            print(f"预训练模型不存在: {pretrained_path}，使用随机初始化")
        
        # 设置为检测模式
        model.set_training_mode('detection')
        
        # 创建检测损失函数和优化器
        criterion = YOLOLoss(
            lambda_coord=self.hyperparameters.get('lambda_coord', 5.0),
            lambda_noobj=self.hyperparameters.get('lambda_noobj', 0.5),
            grid_size=input_size // self.hyperparameters.get('grid_size', 64),
            num_classes=20
        )
        
        optimizer = create_yolo_optimizer(
            model,
            learning_rate=voc_config.get('learning_rate', 0.0001),
            weight_decay=self.hyperparameters.get('weight_decay', 0.0005)
        )
        
        # 训练循环
        best_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f'阶段2 Epoch {epoch+1}/{epochs}')
            
            for batch_idx, batch_data in enumerate(progress_bar):
                if len(batch_data) == 4:  # 虚拟数据
                    images, targets, _, _ = batch_data
                    targets = [targets]
                else:  # 真实数据
                    images, targets = batch_data
                
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]
                
                # 前向传播
                optimizer.zero_grad()
                predictions = model(images)
                
                # 计算损失
                gt = targets[0]  # ground truth
                loss, loss_dict = criterion(predictions, gt)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.get_current_lr():.6f}'
                })
            
            # 计算epoch统计
            avg_loss = total_loss / len(train_loader)
            self.stage2_losses.append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                stage2_path = os.path.join(self.save_dir, 'stage2_detection_best.pth')
                save_checkpoint(model, optimizer.optimizer, epoch, avg_loss, stage2_path)
                print(f"保存阶段2最佳模型，损失: {best_loss:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 20 == 0:
                checkpoint_path = os.path.join(self.save_dir, f'stage2_epoch_{epoch+1}.pth')
                save_checkpoint(model, optimizer.optimizer, epoch, avg_loss, checkpoint_path)
            
            # 更新学习率
            optimizer.step(avg_loss)
        
        print(f"阶段2训练完成！最佳损失: {best_loss:.4f}")
        return os.path.join(self.save_dir, 'stage2_detection_best.pth')
    
    def train_two_stage(self, coco_config: Dict, voc_config: Dict):
        """
        执行完整的两阶段训练
        Args:
            coco_config: COCO数据配置
            voc_config: VOC数据配置
        """
        print("开始两阶段YOLO训练...")
        
        # 阶段1：COCO分类预训练
        stage1_epochs = coco_config.get('epochs', 30)
        stage1_model_path = self.stage1_classification_pretraining(coco_config, stage1_epochs)
        
        # 阶段2：VOC检测微调
        stage2_epochs = voc_config.get('epochs', 80)
        stage2_model_path = self.stage2_detection_finetuning(stage1_model_path, voc_config, stage2_epochs)
        
        print("="*60)
        print("两阶段训练完成!")
        print(f"阶段1模型: {stage1_model_path}")
        print(f"阶段2模型: {stage2_model_path}")
        print("="*60)
        
        return stage2_model_path


def two_stage_train(config_file: str = None):
    """两阶段训练主函数"""
    
    # 加载配置文件
    if config_file and os.path.exists(config_file):
        import json
        print(f"加载配置文件: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 从配置文件中提取超参数
        model_config = config.get('model_config', {})
        training_config = config.get('training_config', {})
        
        hyperparameters = {
            'input_size': model_config.get('input_size', 448),
            'grid_size': model_config.get('grid_size', 64),
            'lambda_coord': model_config.get('lambda_coord', 5.0),
            'lambda_noobj': model_config.get('lambda_noobj', 0.5),
            'weight_decay': model_config.get('weight_decay', 0.0005),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu' if training_config.get('device') == 'auto' else training_config.get('device', 'cpu'),
            'save_dir': training_config.get('save_dir', './two_stage_checkpoints')
        }
        
        # COCO分类预训练配置
        stage1_config = config.get('stage1_coco_classification', {})
        coco_config = {
            'imgs_path': stage1_config.get('coco_train_images', r"D:\COCO\images\train2017"),
            'coco_json': stage1_config.get('coco_train_annotations', r"D:\COCO\annotations\instances_train2017.json"),
            'batch_size': stage1_config.get('batch_size', 16),
            'learning_rate': stage1_config.get('learning_rate', 0.001),
            'epochs': stage1_config.get('epochs', 30),
            'min_area': stage1_config.get('min_object_area', 1000),
            'max_objects_per_image': stage1_config.get('max_objects_per_image', 10)
        }
        
        # VOC检测微调配置
        stage2_config = config.get('stage2_voc_detection', {})
        voc_config = {
            'voc2007_train_imgs': stage2_config.get('voc2007_train_images', r"D:\VOC\VOC2007\train\JPEGImages"),
            'voc2007_train_annotations': stage2_config.get('voc2007_train_annotations', r"D:\VOC\VOC2007\train\Annotations"),
            'voc2012_train_imgs': stage2_config.get('voc2012_train_images', r"D:\VOC\VOC2012\train\JPEGImages"),
            'voc2012_train_annotations': stage2_config.get('voc2012_train_annotations', r"D:\VOC\VOC2012\train\Annotations"),
            'class_file': stage2_config.get('class_file', "voc_classes.txt"),
            'batch_size': stage2_config.get('batch_size', 8),
            'learning_rate': stage2_config.get('learning_rate', 0.0001),
            'epochs': stage2_config.get('epochs', 80)
        }
        
        print("✓ 配置文件加载成功")
        
    else:
        print("使用默认配置...")
        # 加载超参数
        hyperparameters = load_hyperparameters()
        
        # COCO分类预训练配置
        coco_config = {
            'imgs_path': r"D:\COCO\images\train2017",
            'coco_json': r"D:\COCO\annotations\instances_train2017.json",
            'batch_size': 16,
            'learning_rate': 0.001,
            'epochs': 30,
            'min_area': 1000,  # 最小目标面积
            'max_objects_per_image': 10  # 每张图片最大目标数
        }
        
        # VOC检测微调配置
        voc_config = {
            'voc2007_train_imgs': r"D:\VOC\VOC2007\train\JPEGImages",
            'voc2007_train_annotations': r"D:\VOC\VOC2007\train\Annotations",
            'voc2012_train_imgs': r"D:\VOC\VOC2012\train\JPEGImages",
            'voc2012_train_annotations': r"D:\VOC\VOC2012\train\Annotations",
            'class_file': "voc_classes.txt",
            'batch_size': 8,
            'learning_rate': 0.0001,
            'epochs': 80
        }
    
    # 创建两阶段训练器
    trainer = TwoStageYOLOTrainer(
        hyperparameters=hyperparameters,
        save_dir=hyperparameters.get('save_dir', './checkpoints')
    )
    
    # 执行两阶段训练
    final_model_path = trainer.train_two_stage(coco_config, voc_config)
    
    print(f"两阶段训练完成！最终模型保存在: {final_model_path}")
    
    return final_model_path


if __name__ == "__main__":
    two_stage_train()
