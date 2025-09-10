# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - 主窗口类
"""
import sys
import os
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import Qt
from aiui import AutonomousDrivingUISetup
from aicamera import CameraVideoHandler
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2
from PyQt5.QtGui import QPixmap, QImage


class AutonomousDrivingUI(QMainWindow, AutonomousDrivingUISetup, CameraVideoHandler):
    """自动驾驶目标检测系统主窗口"""

    def __init__(self):
        super().__init__()
        # 初始化UI
        self.setup_ui()
        # 初始化摄像头和视频处理
        self.setup_camera_video()

        # 加载模型
        self.model = self.load_model()

        # 初始化统计信息
        self.reset_stats()

    def reset_stats(self):
        """重置统计信息"""
        self.lbl_targets.setText("0")
        self.lbl_time.setText("0 ms")
        self.lbl_fps.setText("0 FPS")
        self.classes_info.setPlainText("🚗 车辆: 0\n🚶 行人: 0\n🚦 交通灯: 0\n🚧 标志牌: 0")
        self.progress.setValue(0)

    def connect_signals(self):
        """连接所有信号和槽函数"""
        # 参数控制信号
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        self.conf_slider.valueChanged.connect(self.on_conf_slider_changed)
        self.iou_spin.valueChanged.connect(self.on_iou_changed)
        self.iou_slider.valueChanged.connect(self.on_iou_slider_changed)

        # 输入源选择信号
        self.btn_image.clicked.connect(self.on_image_clicked)
        self.btn_video.clicked.connect(self.on_video_clicked)
        self.btn_camera.clicked.connect(self.on_camera_clicked)

        # 控制信号
        self.btn_play.clicked.connect(self.on_play_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)

    # 参数调整回调函数
    def on_conf_changed(self, value):
        """Confidence值改变"""
        self.conf_slider.setValue(int(value * 100))
        self.console.append(f"📊 Confidence阈值调整为: {value:.2f}")

    def on_conf_slider_changed(self, value):
        """Confidence滑块改变"""
        self.conf_spin.setValue(value / 100.0)

    def on_iou_changed(self, value):
        """IOU值改变"""
        self.iou_slider.setValue(int(value * 100))
        self.console.append(f"📊 IOU阈值调整为: {value:.2f}")

    def on_iou_slider_changed(self, value):
        """IOU滑块改变"""
        self.iou_spin.setValue(value / 100.0)

    def load_model(self):
        """加载训练好的模型"""
        model_path = 'model.pth'
        if not os.path.exists(model_path):
            self.console.append("⚠️ 模型文件不存在，跳过加载")
            return None

        try:
            model = torch.load(model_path)
            model.eval()
            self.console.append("✅ 模型加载成功")
            return model
        except Exception as e:
            self.console.append(f"❌ 模型加载失败: {str(e)}")
            return None

    def detect_image(self, image_path):
        """检测图像并显示结果"""
        # 首先显示图像
        if not self.display_image(image_path):
            return

        if self.model is None:
            self.console.append("⚠️ 未加载模型，无法进行检测")
            return

        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')

            # 图像预处理
            transform = transforms.Compose([
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ])
            input_tensor = transform(image).unsqueeze(0)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(input_tensor)

            # 处理检测结果
            self.process_detection_results(outputs)

            self.console.append("✅ 图像检测完成")

        except Exception as e:
            self.console.append(f"❌ 图像检测失败: {str(e)}")

    def process_detection_results(self, outputs):
        """处理检测结果并显示"""
        # 这里应该根据实际的模型输出格式来解析检测结果
        # 由于不知道具体模型结构，这里只显示基本信息
        self.lbl_targets.setText("检测中...")
        self.lbl_time.setText("计算中...")
        self.lbl_fps.setText("N/A")
        self.classes_info.setPlainText("🚗 车辆: 计算中...\n🚶 行人: 计算中...\n🚦 交通灯: 计算中...\n🚧 标志牌: 计算中...")
        self.progress.setValue(50)

        # 模拟处理完成
        self.console.append("🔍 正在分析检测结果...")

    # 输入源选择回调函数
    def on_image_clicked(self):
        """图像检测按钮点击"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.stop_all()  # 停止其他模式
            self.current_image_path = file_path
            self.console.append(f"📷 已选择图像: {os.path.basename(file_path)}")
            # 显示图像并进行检测
            self.detect_image(file_path)

    def on_video_clicked(self):
        """视频检测按钮点击"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.stop_all()  # 停止其他模式
            self.console.append(f"🎬 已选择视频: {os.path.basename(file_path)}")
            self.start_video_playback(file_path)

    def on_camera_clicked(self):
        """摄像头按钮点击"""
        if not self.camera_running:
            # 停止其他模式
            self.stop_video_playback()

            # 尝试启动真实摄像头
            if self.start_camera():
                self.console.append("📹 启动真实摄像头检测")
                self.btn_camera.setText("📹 停止摄像头")
                self.btn_camera.setStyleSheet(
                    "QPushButton { background-color: #e74c3c; color: white; font-size: 14px; }")
                # 重置统计信息
                self.reset_stats()
            else:
                self.console.append("❌ 无法启动摄像头")
        else:
            # 停止摄像头
            self.stop_camera()
            self.btn_camera.setText("📹 实时摄像头")
            self.btn_camera.setStyleSheet("QPushButton { background-color: #8e44ad; color: white; font-size: 14px; }")

    # 控制回调函数
    def on_play_clicked(self):
        """开始检测按钮点击"""
        if self.camera_running:
            self.console.append("▶️ 摄像头检测已在运行")
        elif self.video_thread and self.video_thread.isRunning():
            self.console.append("▶️ 视频检测已在运行")
        elif self.current_image_path:
            self.console.append("▶️ 重新检测当前图像")
            self.detect_image(self.current_image_path)
        else:
            self.console.append("⚠️ 请先选择输入源")

    def on_stop_clicked(self):
        """停止检测按钮点击"""
        self.stop_all()
        self.console.append("⏹️ 停止所有检测")
        # 重置统计信息
        self.reset_stats()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_all()
        event.accept()
