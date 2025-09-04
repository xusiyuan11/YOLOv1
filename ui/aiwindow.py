# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - 主窗口类
"""
import sys
import os
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QTimer
from aiui import AutonomousDrivingUISetup
from aicamera import CameraSimulation


class AutonomousDrivingUI(QMainWindow, AutonomousDrivingUISetup, CameraSimulation):
    """自动驾驶目标检测系统主窗口"""
    
    def __init__(self):
        super().__init__()
        # 初始化UI
        self.setup_ui()
        # 初始化模拟系统
        self.setup_simulation()
        
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
        
    # 输入源选择回调函数
    def on_image_clicked(self):
        """图像检测按钮点击"""
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.console.append(f"📷 已选择图像: {os.path.basename(file_path)}")
            self.simulate_detection()
            
    def on_video_clicked(self):
        """视频检测按钮点击"""
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.console.append(f"🎬 已选择视频: {os.path.basename(file_path)}")
            self.start_simulation()
            
    def on_camera_clicked(self):
        """摄像头按钮点击"""
        if not self.camera_running:
            # 尝试启动真实摄像头
            if self.start_camera():
                self.console.append("📹 启动真实摄像头检测")
                self.btn_camera.setText("📹 停止摄像头")
                self.btn_camera.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-size: 14px; }")
            else:
                # 如果摄像头不可用，启动模拟
                self.console.append("📹 摄像头不可用，启动模拟模式")
                self.start_simulation()
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
        else:
            self.console.append("▶️ 开始模拟检测")
            self.start_simulation()
        
    def on_stop_clicked(self):
        """停止检测按钮点击"""
        if self.camera_running:
            self.stop_camera()
            self.btn_camera.setText("📹 实时摄像头")
            self.btn_camera.setStyleSheet("QPushButton { background-color: #8e44ad; color: white; font-size: 14px; }")
        
        self.console.append("⏹️ 停止检测")
        self.stop_simulation()
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.camera_running:
            self.stop_camera()
        self.stop_simulation()
        event.accept()
        
