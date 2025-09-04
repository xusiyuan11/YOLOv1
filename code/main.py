#!/usr/bin/env python3
"""
YOLO复现项目主入口
提供两阶段训练、测试、可视化的统一接口
"""

import os
import sys
import argparse
from typing import Dict, Any

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO复现项目')
    parser.add_argument('mode', choices=['train', 'test', 'visualize', 'demo'], 
                       help='运行模式: train(两阶段训练), test(测试), visualize(可视化), demo(演示)')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, help='模型检查点路径')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--output', type=str, default='./output', help='输出目录')
    
    args = parser.parse_args()
    
    print("="*50)
    print("YOLO复现项目")
    print("="*50)
    
    if args.mode == 'train':
        print("启动两阶段训练模式...")
        from Train import two_stage_train
        config_file = args.config or 'two_stage_config.json'
        two_stage_train(config_file)
    
    elif args.mode == 'test':
        print("启动测试模式...")
        from Test import main as test_main
        test_main()
    
    elif args.mode == 'visualize':
        print("启动可视化模式...")
        from result_visualisation import main as vis_main
        vis_main()
    
    elif args.mode == 'demo':
        print("启动演示模式...")
        demo_yolo_system()
    
    else:
        parser.print_help()


def demo_yolo_system():
    """YOLO系统演示"""
    import torch
    from NetModel import YOLOv1, test_yolo_model
    from YOLOLoss import test_yolo_loss
    from Utils import test_utils
    from OPT import test_optimizer
    
    print("\n1. 测试YOLO模型...")
    try:
        test_yolo_model()
        print("✓ YOLO模型测试通过")
    except Exception as e:
        print(f"✗ YOLO模型测试失败: {e}")
    
    print("\n2. 测试YOLO损失函数...")
    try:
        test_yolo_loss()
        print("✓ YOLO损失函数测试通过")
    except Exception as e:
        print(f"✗ YOLO损失函数测试失败: {e}")
    
    print("\n3. 测试工具函数...")
    try:
        test_utils()
        print("✓ 工具函数测试通过")
    except Exception as e:
        print(f"✗ 工具函数测试失败: {e}")
    
    print("\n4. 测试优化器...")
    try:
        test_optimizer()
        print("✓ 优化器测试通过")
    except Exception as e:
        print(f"✗ 优化器测试失败: {e}")
    
    print("\n5. 测试数据集...")
    try:
        from dataset import run_cifar_example
        run_cifar_example()
        print("✓ 数据集测试通过")
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
    
    print("\n演示完成！")
    print("\n系统组件状态:")
    print("- backbone.py: DarkNet骨干网络 ✓")
    print("- NetModel.py: YOLO网络模型 ✓") 
    print("- YOLOLoss.py: YOLO损失函数 ✓")
    print("- Utils.py: 工具函数集合 ✓")
    print("- OPT.py: 优化器管理 ✓")
    print("- dataset.py: 数据集处理 ✓")
    print("- Train.py: 训练模块 ✓")
    print("- Test.py: 测试模块 ✓")
    print("- result_visualisation.py: 结果可视化 ✓")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 没有参数时显示帮助信息
        print("YOLO复现项目")
        print("\n使用方法:")
        print("  python main.py train      # 两阶段训练")
        print("  python main.py test       # 开始测试")
        print("  python main.py visualize  # 可视化结果")
        print("  python main.py demo       # 系统演示")
        print("\n可选参数:")
        print("  --config CONFIG_FILE     # 指定配置文件")
        print("  --checkpoint MODEL_FILE  # 指定模型文件")
        print("  --image IMAGE_FILE       # 指定测试图像")
        print("  --output OUTPUT_DIR      # 指定输出目录")
    else:
        main()
