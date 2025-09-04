# -*- coding: UTF-8 -*-
import sys
import os
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QGroupBox, QPushButton, QDoubleSpinBox, QSlider,
                             QCheckBox, QProgressBar, QLineEdit, QFileDialog, QMessageBox,
                             QTextEdit, QComboBox)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon, QPalette, QColor, QPainter


class AutonomousDrivingUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_simulation()

    def setup_ui(self):
        self.setWindowTitle("自动驾驶目标检测系统 v1.0")
        self.setGeometry(100, 100, 1600, 900)

        # 设置应用程序样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QGroupBox {
                font-weight: bold;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background-color: #34495e;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #3498db;
            }
            QPushButton {
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                border: 2px solid #2c3e50;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QLabel {
                color: #ecf0f1;
            }
            QProgressBar {
                border: 2px solid #34495e;
                border-radius: 5px;
                text-align: center;
                background-color: #2c3e50;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #34495e;
                border-radius: 4px;
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QDoubleSpinBox {
                padding: 6px;
                border: 2px solid #34495e;
                border-radius: 4px;
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QCheckBox {
                color: #ecf0f1;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QCheckBox::indicator:checked {
                background-color: #3498db;
                border: 2px solid #2980b9;
            }
        """)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左侧控制面板 (30%)
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)

        # 系统状态组
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)

        status_info = QTextEdit()
        status_info.setPlainText("✅ 系统就绪\n✅ 模型加载完成\n✅ 摄像头可用\n⏹️ 等待输入源")
        status_info.setMaximumHeight(100)
        status_layout.addWidget(status_info)
        left_layout.addWidget(status_group)

        # 模型配置组
        model_group = QGroupBox("模型配置")
        model_layout = QVBoxLayout(model_group)

        # 权重选择
        weight_layout = QHBoxLayout()
        weight_label = QLabel("模型权重:")
        weight_label.setFont(QFont("Arial", 10, QFont.Bold))
        weight_layout.addWidget(weight_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["yolo(自动驾驶)", "yolo (标准版)", "yolo(快速版)"])
        self.model_combo.setCurrentIndex(1)
        weight_layout.addWidget(self.model_combo)
        model_layout.addLayout(weight_layout)

        # 参数设置
        params_layout = QVBoxLayout()

        # Confidence设置
        conf_layout = QVBoxLayout()
        conf_label = QLabel("置信度阈值 (Confidence):")
        conf_label.setFont(QFont("Arial", 9))
        conf_layout.addWidget(conf_label)

        conf_control_layout = QHBoxLayout()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        # Style and link spinbox <-> slider with a compact value label
        self.conf_spin.setDecimals(2)
        self.conf_spin.setFixedWidth(80)
        conf_control_layout.addWidget(self.conf_spin)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        # nicer visual style for the slider
        self.conf_slider.setStyleSheet(
            "QSlider::groove:horizontal{height:8px;background:#34495e;border-radius:4px;}"
            "QSlider::handle:horizontal{background:#3498db;border:2px solid #2980b9;width:18px;margin:-6px 0;}"
        )
        conf_control_layout.addWidget(self.conf_slider)

        self.conf_value_label = QLabel(f"{self.conf_spin.value():.2f}")
        self.conf_value_label.setFixedWidth(50)
        self.conf_value_label.setAlignment(Qt.AlignCenter)
        conf_control_layout.addWidget(self.conf_value_label)

        # Bind slider and spinbox together
        self.conf_spin.valueChanged.connect(lambda v: self.conf_slider.setValue(int(v * 100)))
        self.conf_slider.valueChanged.connect(lambda v: self.conf_spin.setValue(v / 100.0))
        self.conf_spin.valueChanged.connect(lambda v: self.conf_value_label.setText(f"{v:.2f}"))
        conf_layout.addLayout(conf_control_layout)
        params_layout.addLayout(conf_layout)

        # IOU设置
        iou_layout = QVBoxLayout()
        iou_label = QLabel("交并比阈值 (IOU):")
        iou_label.setFont(QFont("Arial", 9))
        iou_layout.addWidget(iou_label)

        iou_control_layout = QHBoxLayout()
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        # Style and link spinbox <-> slider with a compact value label
        self.iou_spin.setDecimals(2)
        self.iou_spin.setFixedWidth(80)
        iou_control_layout.addWidget(self.iou_spin)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(45)
        self.iou_slider.setStyleSheet(
            "QSlider::groove:horizontal{height:8px;background:#34495e;border-radius:4px;}"
            "QSlider::handle:horizontal{background:#3498db;border:2px solid #2980b9;width:18px;margin:-6px 0;}"
        )
        iou_control_layout.addWidget(self.iou_slider)

        self.iou_value_label = QLabel(f"{self.iou_spin.value():.2f}")
        self.iou_value_label.setFixedWidth(50)
        self.iou_value_label.setAlignment(Qt.AlignCenter)
        iou_control_layout.addWidget(self.iou_value_label)

        # Bind slider and spinbox together
        self.iou_spin.valueChanged.connect(lambda v: self.iou_slider.setValue(int(v * 100)))
        self.iou_slider.valueChanged.connect(lambda v: self.iou_spin.setValue(v / 100.0))
        self.iou_spin.valueChanged.connect(lambda v: self.iou_value_label.setText(f"{v:.2f}"))
        iou_layout.addLayout(iou_control_layout)
        params_layout.addLayout(iou_layout)

        model_layout.addLayout(params_layout)
        left_layout.addWidget(model_group)

        # 输入源组
        input_group = QGroupBox("输入源选择")
        input_layout = QVBoxLayout(input_group)

        self.btn_image = QPushButton("📷 图像检测")
        self.btn_image.setStyleSheet("QPushButton { background-color: #27ae60; color: white; font-size: 14px; }")
        input_layout.addWidget(self.btn_image)

        self.btn_video = QPushButton("🎬 视频检测")
        self.btn_video.setStyleSheet("QPushButton { background-color: #f39c12; color: white; font-size: 14px; }")
        input_layout.addWidget(self.btn_video)

        self.btn_camera = QPushButton("📹 实时摄像头")
        self.btn_camera.setStyleSheet("QPushButton { background-color: #8e44ad; color: white; font-size: 14px; }")
        input_layout.addWidget(self.btn_camera)

        left_layout.addWidget(input_group)

        # 控制组
        control_group = QGroupBox("系统控制")
        control_layout = QVBoxLayout(control_group)

        # 播放控制
        play_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶️ 开始检测")
        self.btn_play.setStyleSheet("QPushButton { background-color: #e74c3c; color: white; font-size: 14px; }")
        play_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹️ 停止检测")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #7f8c8d; color: white; font-size: 14px; }")
        play_layout.addWidget(self.btn_stop)
        control_layout.addLayout(play_layout)

        # 保存选项
        save_layout = QHBoxLayout()
        self.cb_save = QCheckBox("自动保存结果")
        self.cb_save.setChecked(True)
        save_layout.addWidget(self.cb_save)
        control_layout.addLayout(save_layout)

        left_layout.addWidget(control_group)

        # 检测信息组
        info_group = QGroupBox("检测信息")
        info_layout = QVBoxLayout(info_group)

        # 统计信息
        stats_layout = QVBoxLayout()

        # 目标数量
        targets_layout = QHBoxLayout()
        targets_label = QLabel("检测目标:")
        targets_label.setFont(QFont("Arial", 10, QFont.Bold))
        targets_layout.addWidget(targets_label)
        self.lbl_targets = QLabel("0")
        self.lbl_targets.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_targets.setStyleSheet("color: #3498db;")
        targets_layout.addWidget(self.lbl_targets)
        targets_layout.addStretch()
        stats_layout.addLayout(targets_layout)

        # 推理时间
        time_layout = QHBoxLayout()
        time_label = QLabel("推理时间:")
        time_label.setFont(QFont("Arial", 10, QFont.Bold))
        time_layout.addWidget(time_label)
        self.lbl_time = QLabel("0 ms")
        self.lbl_time.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_time.setStyleSheet("color: #2ecc71;")
        time_layout.addWidget(self.lbl_time)
        time_layout.addStretch()
        stats_layout.addLayout(time_layout)

        # FPS
        fps_layout = QHBoxLayout()
        fps_label = QLabel("处理帧率:")
        fps_label.setFont(QFont("Arial", 10, QFont.Bold))
        fps_layout.addWidget(fps_label)
        self.lbl_fps = QLabel("0 FPS")
        self.lbl_fps.setFont(QFont("Arial", 10, QFont.Bold))
        self.lbl_fps.setStyleSheet("color: #e74c3c;")
        fps_layout.addWidget(self.lbl_fps)
        fps_layout.addStretch()
        stats_layout.addLayout(fps_layout)

        info_layout.addLayout(stats_layout)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setValue(0)
        info_layout.addWidget(self.progress)

        # 检测类别统计
        classes_layout = QVBoxLayout()
        classes_label = QLabel("检测类别统计:")
        classes_label.setFont(QFont("Arial", 9, QFont.Bold))
        classes_layout.addWidget(classes_label)

        self.classes_info = QTextEdit()
        self.classes_info.setMaximumHeight(120)
        self.classes_info.setPlainText("🚗 车辆: 0\n🚶 行人: 0\n🚦 交通灯: 0\n🚧 标志牌: 0")
        classes_layout.addWidget(self.classes_info)
        info_layout.addLayout(classes_layout)

        left_layout.addWidget(info_group)
        left_layout.addStretch()

        # 右侧显示面板 (70%)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 视频显示组
        video_group = QGroupBox("实时检测画面")
        video_layout = QVBoxLayout(video_group)

        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("""
            QLabel {
                border: 3px solid #34495e;
                background-color: #1a1a1a;
                min-height: 500px;
                color: #95a5a6;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.lbl_video.setText("选择输入源开始检测")
        video_layout.addWidget(self.lbl_video)
        right_layout.addWidget(video_group)

        # 控制台输出组
        console_group = QGroupBox("系统日志")
        console_layout = QVBoxLayout(console_group)

        self.console = QTextEdit()
        self.console.setMaximumHeight(120)
        self.console.setPlainText("🚀 系统启动成功\n📋 初始化完成\n✅ 准备就绪")
        console_layout.addWidget(self.console)
        right_layout.addWidget(console_group)

        # 将左右面板添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # 连接信号槽
        self.connect_signals()

    def connect_signals(self):
        """连接所有信号和槽函数"""
        self.conf_spin.valueChanged.connect(self.on_conf_changed)
        self.conf_slider.valueChanged.connect(self.on_conf_slider_changed)
        self.iou_spin.valueChanged.connect(self.on_iou_changed)
        self.iou_slider.valueChanged.connect(self.on_iou_slider_changed)

        self.btn_image.clicked.connect(self.on_image_clicked)
        self.btn_video.clicked.connect(self.on_video_clicked)
        self.btn_camera.clicked.connect(self.on_camera_clicked)
        self.btn_play.clicked.connect(self.on_play_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)

    def setup_simulation(self):
        """设置模拟定时器"""
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)
        self.sim_timer.setInterval(100)  # 10 FPS
        self.simulation_running = False
        self.frame_count = 0

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

    def on_image_clicked(self):
        """图像检测按钮点击"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.console.append(f"📷 已选择图像: {os.path.basename(file_path)}")
            self.simulate_detection()

    def on_video_clicked(self):
        """视频检测按钮点击"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.console.append(f"🎬 已选择视频: {os.path.basename(file_path)}")
            self.start_simulation()

    def on_camera_clicked(self):
        """摄像头按钮点击"""
        self.console.append("📹 启动摄像头检测")
        self.start_simulation()

    def on_play_clicked(self):
        """开始检测按钮点击"""
        self.console.append("▶️ 开始检测")
        self.start_simulation()

    def on_stop_clicked(self):
        """停止检测按钮点击"""
        self.console.append("⏹️ 停止检测")
        self.stop_simulation()

    def start_simulation(self):
        """开始模拟"""
        if not self.simulation_running:
            self.simulation_running = True
            self.sim_timer.start()
            self.frame_count = 0

    def stop_simulation(self):
        """停止模拟"""
        if self.simulation_running:
            self.simulation_running = False
            self.sim_timer.stop()

    def update_simulation(self):
        """更新模拟数据"""
        self.frame_count += 1

        # 更新统计信息
        targets = random.randint(3, 15)
        inference_time = random.randint(20, 50)
        fps = random.randint(25, 35)

        self.lbl_targets.setText(str(targets))
        self.lbl_time.setText(f"{inference_time} ms")
        self.lbl_fps.setText(f"{fps} FPS")

        # 更新类别统计
        classes_text = f"🚗 车辆: {random.randint(2, 8)}\n🚶 行人: {random.randint(1, 5)}\n🚦 交通灯: {random.randint(0, 3)}\n🚧 标志牌: {random.randint(0, 4)}"
        self.classes_info.setPlainText(classes_text)

        # 更新进度条
        if self.frame_count % 5 == 0:
            self.progress.setValue(random.randint(0, 100))

        # 创建模拟图像
        if self.frame_count % 2 == 0:
            self.update_display_image()

    def update_display_image(self):
        """更新显示图像（模拟）"""
        # 创建一个模拟的检测结果图像
        width, height = 640, 480
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(QColor(30, 30, 30))

        painter = QPainter(image)
        painter.setPen(QColor(0, 255, 0))
        painter.drawText(150, 200, "自动驾驶目标检测模拟")
        painter.drawText(180, 230, f"帧数: {self.frame_count}")

        # 画一些模拟的检测框
        for i in range(random.randint(3, 8)):
            x = random.randint(50, width - 100)
            y = random.randint(50, height - 100)
            w = random.randint(50, 150)
            h = random.randint(50, 150)

            color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            painter.setPen(color)
            painter.drawRect(x, y, w, h)
            painter.drawText(x, y - 5, f"目标 {i + 1}")

        painter.end()

        # 显示图像
        pixmap = QPixmap.fromImage(image)
        self.lbl_video.setPixmap(pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def simulate_detection(self):
        """模拟单次检测"""
        # 更新统计信息
        targets = random.randint(1, 10)
        inference_time = random.randint(15, 80)

        self.lbl_targets.setText(str(targets))
        self.lbl_time.setText(f"{inference_time} ms")
        self.lbl_fps.setText("N/A")

        # 更新类别统计
        classes_text = f"🚗 车辆: {random.randint(1, 6)}\n🚶 行人: {random.randint(0, 3)}\n🚦 交通灯: {random.randint(0, 2)}\n🚧 标志牌: {random.randint(0, 2)}"
        self.classes_info.setPlainText(classes_text)

        # 更新进度条
        self.progress.setValue(100)

        # 创建模拟图像
        self.update_display_image()

        self.console.append("✅ 图像检测完成")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序图标和样式
    app.setStyle('Fusion')

    window = AutonomousDrivingUI()
    window.show()

    sys.exit(app.exec_())