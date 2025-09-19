"""
两阶段训练完整流程示例
Complete Two-Stage Training Workflow
"""
import os
import time
from Train_Classification import BackboneClassificationTrainer
from Train_Detection import DetectionTrainer
from Utils import load_hyperparameters


def run_two_stage_training():
    """运行完整的两阶段训练流程"""
    print("="*80)
    print("YOLO 两阶段训练完整流程")
    print("="*80)
    
    # 加载配置
    hyperparameters = load_hyperparameters()
    
    # 配置数据路径
    coco_config = {
        'imgs_path': '../data/COCO/train2017',
        'coco_json': '../data/COCO/annotations/instances_train2017.json',
        'batch_size': 16,
        'learning_rate': 0.001,
        'min_area': 1000,
        'max_objects_per_image': 10,
        'num_classes': 80
    }
    
    voc_config = {
        'voc_data_path': '../data/VOC2012',
        'batch_size': 8,
        'backbone_lr': 0.0001,
        'detection_lr': 0.001,
        'num_classes': 20
    }
    
    # 第一阶段：分类预训练
    print("\\n开始第一阶段：backbone分类预训练")
    classification_trainer = BackboneClassificationTrainer(
        hyperparameters=hyperparameters,
        save_dir='./checkpoints/classification'
    )
    
    backbone_path, classification_acc = classification_trainer.train(
        coco_config=coco_config,
        epochs=30
    )
    
    print(f"第一阶段完成！分类精度: {classification_acc:.2f}%")
    print(f"训练好的backbone保存在: {backbone_path}")
    
    # 等待一下
    time.sleep(2)
    
    # 第二阶段：检测微调
    print("\\n开始第二阶段：检测头训练")
    detection_trainer = DetectionTrainer(
        hyperparameters=hyperparameters,
        save_dir='./checkpoints/detection'
    )
    
    detection_map = detection_trainer.train(
        voc_config=voc_config,
        backbone_path=backbone_path,
        epochs=50,
        freeze_epochs=10
    )
    
    print(f"第二阶段完成！检测mAP: {detection_map:.4f}")
    
    # 总结
    print("\\n" + "="*80)
    print("两阶段训练流程全部完成！")
    print("="*80)
    print(f"阶段1 - 分类预训练精度: {classification_acc:.2f}%")
    print(f"阶段2 - 检测训练mAP: {detection_map:.4f}")
    print(f"最终模型保存在: ./checkpoints/detection/best_detection_model.pth")
    

def run_classification_only():
    """只运行分类训练"""
    print("="*60)
    print("仅运行分类预训练")
    print("="*60)
    
    hyperparameters = load_hyperparameters()
    
    coco_config = {
        'imgs_path': '../data/COCO/train2017',
        'coco_json': '../data/COCO/annotations/instances_train2017.json',
        'batch_size': 16,
        'learning_rate': 0.001,
        'min_area': 1000,
        'max_objects_per_image': 10,
        'num_classes': 80
    }
    
    trainer = BackboneClassificationTrainer(
        hyperparameters=hyperparameters,
        save_dir='./checkpoints/classification'
    )
    
    backbone_path, acc = trainer.train(coco_config, epochs=30)
    print(f"分类训练完成！精度: {acc:.2f}%, Backbone: {backbone_path}")


def run_detection_only():
    """只运行检测训练（需要预训练backbone）"""
    print("="*60)
    print("仅运行检测训练")
    print("="*60)
    
    backbone_path = './checkpoints/classification/trained_backbone.pth'
    
    if not os.path.exists(backbone_path):
        print(f"错误：找不到预训练backbone: {backbone_path}")
        print("请先运行分类训练或完整的两阶段训练")
        return
    
    hyperparameters = load_hyperparameters()
    
    voc_config = {
        'voc_data_path': '../data/VOC2012',
        'batch_size': 8,
        'backbone_lr': 0.0001,
        'detection_lr': 0.001,
        'num_classes': 20
    }
    
    trainer = DetectionTrainer(
        hyperparameters=hyperparameters,
        save_dir='./checkpoints/detection'
    )
    
    map_score = trainer.train(voc_config, backbone_path, epochs=50, freeze_epochs=10)
    print(f"检测训练完成！mAP: {map_score:.4f}")


if __name__ == "__main__":
    print("YOLO 两阶段训练脚本")
    print("1. 完整两阶段训练")
    print("2. 仅分类预训练")
    print("3. 仅检测训练")
    
    choice = input("请选择 (1/2/3): ").strip()
    
    if choice == "1":
        run_two_stage_training()
    elif choice == "2":
        run_classification_only()
    elif choice == "3":
        run_detection_only()
    else:
        print("无效选择！")
        print("默认运行完整两阶段训练...")
        run_two_stage_training()
