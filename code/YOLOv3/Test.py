import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from NetModel import create_yolov3_model
from postprocess import YOLOv3PostProcessor
from Utils import (
    load_hyperparameters, load_checkpoint, decode_yolo_output,
    visualize_detections, create_class_names, calculate_map
)
from dataset import VOC_Detection_Set, COCO_Detection_Set
from compatibility_fixes import apply_yolov3_compatibility_patches

# 应用YOLOv3兼容性补丁
apply_yolov3_compatibility_patches()


class YOLOv3Tester:
    """YOLOv3 测试器"""
    
    def __init__(self, 
                 model = None, 
                 hyperparameters: Dict = None,
                 class_names: List[str] = None):
        self.model = model
        
        # 加载超参数
        if hyperparameters is None:
            hyperparameters = load_hyperparameters()
        self.hyperparameters = hyperparameters
        
        # 设置设备
        self.device = torch.device(hyperparameters.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        self.model.eval()
        
        # 类别名称
        if class_names is None:
            self.class_names = create_class_names('voc')
        else:
            self.class_names = class_names
        
        # 测试参数
        self.conf_threshold = hyperparameters.get('conf_threshold', 0.1)
        self.nms_threshold = hyperparameters.get('nms_threshold', 0.5)
        self.input_size = hyperparameters.get('input_size', 448)
        self.grid_size = hyperparameters.get('grid_size', 7)
        self.num_classes = hyperparameters.get('num_classes', 20)
    
    def load_model(self, checkpoint_path: str):
        """
        加载模型权重
        Args:
            checkpoint_path: 检查点路径
        """
        try:
            print(f"🔄 Attempting to load weights from: {checkpoint_path}")
            
            # 方法1: 尝试直接加载（适用于自己保存的权重）
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                
                # 检查不同的权重键名
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print("📦 Found model_state_dict key")
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print("📦 Found state_dict key")
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    print("📦 Found model key")
                else:
                    # 如果没有特定键，假设整个checkpoint就是state_dict
                    state_dict = checkpoint
                    print("📦 Using entire checkpoint as state_dict")
                
                # 处理ultralytics格式的权重
                if hasattr(state_dict, 'float'):
                    # 这可能是一个模型对象，尝试获取state_dict
                    if hasattr(state_dict, 'state_dict'):
                        state_dict = state_dict.state_dict()
                        print("📦 Extracted state_dict from model object")
                
                # 加载权重，允许部分不匹配
                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                
                print(f"✅ Model loaded from {checkpoint_path}")
                if missing_keys:
                    print(f"⚠️  Missing keys: {len(missing_keys)}")
                if unexpected_keys:
                    print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
                
                return True
                
            except Exception as e1:
                print(f"❌ Standard loading failed: {str(e1)[:100]}...")
                
                # 方法2: 尝试使用pickle模块直接加载
                try:
                    import pickle
                    print("🔄 Trying pickle loading...")
                    
                    with open(checkpoint_path, 'rb') as f:
                        # 跳过不兼容的模块引用
                        class CustomUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                # 重定向不兼容的模块
                                if module.startswith('models.'):
                                    # 尝试忽略models模块的引用
                                    return None
                                return super().find_class(module, name)
                        
                        checkpoint = CustomUnpickler(f).load()
                        
                        # 尝试提取state_dict
                        if isinstance(checkpoint, dict):
                            if 'model' in checkpoint:
                                if hasattr(checkpoint['model'], 'state_dict'):
                                    state_dict = checkpoint['model'].state_dict()
                                else:
                                    state_dict = checkpoint['model']
                            else:
                                state_dict = checkpoint
                        else:
                            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
                        
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                        print(f"✅ Model loaded with pickle method")
                        return True
                        
                except Exception as e2:
                    print(f"❌ Pickle loading failed: {str(e2)[:100]}...")
                    
                    # 方法3: 忽略不兼容的权重，只使用随机初始化
                    print("🎲 All loading methods failed, using random initialization")
                    print("💡 Suggestion: Use weights saved from this codebase or convert external weights")
                    return False
                    
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
        
        return True
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, Tuple]:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = original_image.shape[:2]
        
        # 填充和缩放
        max_edge = max(original_height, original_width)
        top = bottom = left = right = 0
        
        if original_height < max_edge:
            pad = max_edge - original_height
            top = pad // 2
            bottom = pad - top
        else:
            pad = max_edge - original_width
            left = pad // 2
            right = pad - left
        
        # 填充图像
        padded_image = cv2.copyMakeBorder(
            original_image, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # 缩放到输入尺寸
        resized_image = cv2.resize(padded_image, (self.input_size, self.input_size))
        
        # 转换为张量
        tensor_image = torch.from_numpy(resized_image).float().permute(2, 0, 1)
        tensor_image = tensor_image / 255.0
        
        # 归一化
        mean = torch.tensor([0.408, 0.448, 0.471]).view(3, 1, 1)
        std = torch.tensor([0.242, 0.239, 0.234]).view(3, 1, 1)
        tensor_image = (tensor_image - mean) / std
        
        # 添加批次维度
        tensor_image = tensor_image.unsqueeze(0).to(self.device)
        
        return tensor_image, original_image, (original_height, original_width, top, left, max_edge)
    
    def postprocess_detections(self, detections: Dict, image_info: Tuple) -> Dict:
        original_height, original_width, top, left, max_edge = image_info
        scale_factor = max_edge / self.input_size
        
        if len(detections['boxes']) == 0:
            return detections
        
        boxes = detections['boxes'].copy()
        
        # 缩放回填充图像尺寸
        boxes *= scale_factor
        
        # 移除填充
        boxes[:, [0, 2]] -= left  # x 坐标
        boxes[:, [1, 3]] -= top   # y 坐标
        
        # 裁剪到原始图像范围
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
        
        return {
            'boxes': boxes,
            'scores': detections['scores'],
            'class_ids': detections['class_ids']
        }
    
    def detect_single_image(self, image_path: str, save_path: str = None) -> Dict:
        with torch.no_grad():
            # 预处理
            tensor_image, original_image, image_info = self.preprocess_image(image_path)
            
            # 前向传播
            predictions = self.model(tensor_image)
            
            # 解码预测结果
            decoded_predictions = decode_yolo_output(
                predictions.cpu(),
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                input_size=self.input_size,
                grid_size=self.grid_size,
                num_classes=self.num_classes
            )
            
            detections = decoded_predictions[0]
            
            # 后处理坐标
            detections = self.postprocess_detections(detections, image_info)
            
            # 可视化和保存
            if save_path:
                vis_image = visualize_detections(
                    original_image, detections, self.class_names, 
                    conf_threshold=self.conf_threshold
                )
                vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, vis_image_bgr)
                print(f"Detection result saved to {save_path}")
            
            return detections
    
    def evaluate_dataset(self, test_loader: DataLoader) -> Dict:
        all_predictions = []
        all_targets = []
        
        print("Evaluating on test dataset...")
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc='Testing'):
                images = images.to(self.device)
                
                # 前向传播
                predictions = self.model(images)
                
                # 解码预测结果
                decoded_predictions = decode_yolo_output(
                    predictions.cpu(),
                    conf_threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold,
                    input_size=self.input_size,
                    grid_size=self.grid_size,
                    num_classes=self.num_classes
                )
                
                all_predictions.extend(decoded_predictions)
                
                # 处理目标数据（需要根据实际数据格式调整）
                batch_size = images.size(0)
                for i in range(batch_size):
                    # 这里应该将targets转换为标准格式
                    # 简化处理，实际使用时需要具体实现
                    all_targets.append({
                        'boxes': np.array([]),
                        'scores': np.array([]),
                        'class_ids': np.array([])
                    })
        
        # 计算评估指标
        try:
            map_results = calculate_map(
                all_predictions, all_targets,
                num_classes=self.num_classes,
                iou_threshold=0.5
            )
        except Exception as e:
            print(f"Error calculating mAP: {e}")
            map_results = {'mAP': 0.0, 'APs': [], 'num_classes': 0}
        
        # 统计检测结果
        total_detections = sum(len(pred['boxes']) for pred in all_predictions)
        avg_detections_per_image = total_detections / len(all_predictions) if all_predictions else 0
        
        evaluation_results = {
            'mAP': map_results['mAP'],
            'total_images': len(all_predictions),
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections_per_image,
            'class_APs': map_results['APs']
        }
        
        return evaluation_results
    
    def batch_detect_images(self, 
                           image_dir: str, 
                           output_dir: str, 
                           image_extensions: List[str] = None) -> List[Dict]:
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend([
                f for f in os.listdir(image_dir) 
                if f.lower().endswith(ext.lower())
            ])
        
        all_detections = []
        
        print(f"Processing {len(image_files)} images...")
        
        for image_file in tqdm(image_files, desc='Detecting'):
            image_path = os.path.join(image_dir, image_file)
            output_name = os.path.splitext(image_file)[0] + '_detected.jpg'
            output_path = os.path.join(output_dir, output_name)
            
            try:
                detections = self.detect_single_image(image_path, output_path)
                all_detections.append({
                    'image_file': image_file,
                    'detections': detections
                })
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
        
        print(f"Batch detection completed. Results saved to {output_dir}")
        return all_detections


def create_test_data_loader(data_config: Dict) -> DataLoader:
    """创建测试数据加载器"""
    dataset_type = data_config.get('type', 'voc').lower()
    
    if dataset_type == 'voc':
        test_dataset = VOC_Detection_Set(
            imgs_path=data_config['test_imgs_path'],
            annotations_path=data_config['test_annotations_path'],
            class_file=data_config['class_file'],
            input_size=data_config.get('input_size', 448),
            grid_size=data_config.get('grid_size', 64)
        )
    elif dataset_type == 'coco':
        test_dataset = COCO_Detection_Set(
            imgs_path=data_config['test_imgs_path'],
            coco_json=data_config['test_coco_json'],
            input_size=data_config.get('input_size', 448),
            grid_size=data_config.get('grid_size', 64)
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.get('batch_size', 8),
        shuffle=False,
        num_workers=data_config.get('num_workers', 4)
    )
    
    return test_loader


def main():
    """主测试函数"""
    # 超参数配置
    hyperparameters = {
        'input_size': 448,
        'grid_size': 7,
        'num_classes': 20,
        'conf_threshold': 0.3,
        'nms_threshold': 0.5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 创建模型
    model = create_yolov3_model(
        num_classes=hyperparameters['num_classes'],
        input_size=hyperparameters['input_size']
    )
    
    # 创建测试器
    class_names = create_class_names('voc')
    tester = YOLOv3Tester(model, hyperparameters, class_names)
    
    # 加载模型权重
    checkpoint_path = './checkpoints/detection/yolov3-spp.pt'
    if os.path.exists(checkpoint_path):
        success = tester.load_model(checkpoint_path)
        if not success:
            print("⚠️  Failed to load weights, using random initialization")
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("🎲 Using randomly initialized model for testing...")
        # 尝试查找其他可能的权重文件
        alternative_paths = [
            './checkpoints/detection/best_detection_model.pth',
            './yolov3.weights',
            './yolov3-spp.pt',
            '../yolov3-master/yolov3-spp.pt'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"🔍 Found alternative weights: {alt_path}")
                success = tester.load_model(alt_path)
                if success:
                    break
    
    # 测试模式选择
    test_mode = input("Choose test mode (1: single image, 2: batch images, 3: dataset evaluation): ")
    
    if test_mode == '1':
        # 单张图像测试
        image_path = input("Enter image path: ")
        if os.path.exists(image_path):
            output_path = './output_detected.jpg'
            detections = tester.detect_single_image(image_path, output_path)
            print(f"Detected {len(detections['boxes'])} objects")
            for i, (box, score, class_id) in enumerate(zip(
                detections['boxes'], detections['scores'], detections['class_ids']
            )):
                class_name = class_names[int(class_id)]
                print(f"  {i+1}: {class_name} ({score:.3f})")
        else:
            print("Image not found!")
    
    elif test_mode == '2':
        # 批量图像测试
        image_dir = input("Enter image directory: ")
        if os.path.exists(image_dir):
            output_dir = './batch_output'
            all_detections = tester.batch_detect_images(image_dir, output_dir)
            print(f"Processed {len(all_detections)} images")
        else:
            print("Directory not found!")
    
    elif test_mode == '3':
        # 数据集评估
        data_config = {
            'type': 'voc',
            'test_imgs_path': '../../data/VOC2012/VOCdevkit/VOC2012/JPEGImages',
            'test_annotations_path': '../../data/VOC2012/VOCdevkit/VOC2012/Annotations',
            'class_file': './voc_classes.txt',
            'input_size': hyperparameters['input_size'],
            'grid_size': hyperparameters['input_size'] // hyperparameters['grid_size'],
            'batch_size': 8,
            'num_workers': 4
        }
        
        try:
            test_loader = create_test_data_loader(data_config)
            results = tester.evaluate_dataset(test_loader)
            
            print("Evaluation Results:")
            print(f"  mAP: {results['mAP']:.4f}")
            print(f"  Total Images: {results['total_images']}")
            print(f"  Total Detections: {results['total_detections']}")
            print(f"  Avg Detections/Image: {results['avg_detections_per_image']:.2f}")
            
        except Exception as e:
            print(f"Error in dataset evaluation: {e}")
    
    else:
        print("Invalid test mode!")


if __name__ == "__main__":
    main()
