# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - UI界面设置类
负责界面的布局和样式设置
"""
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QLabel, QGroupBox, 
                            QPushButton, QDoubleSpinBox, QSlider, QCheckBox, QProgressBar, 
                            QLineEdit, QTextEdit, QComboBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class AutonomousDrivingUISetup:
    """UI界面设置基类"""
    
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("自动驾驶目标检测系统 v1.0")
        self.setGeometry(100, 100, 1600, 900)
        
        # 设置应用程序样式
        self.set_app_style()
        
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # 创建左侧面板
        left_panel = self.create_left_panel()
        
        # 创建右侧面板
        right_panel = self.create_right_panel()
        
        # 将左右面板添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # 连接信号槽
        self.connect_signals()
        
    def set_app_style(self):
        """设置应用程序样式表"""
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
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        
        # 系统状态组
        left_layout.addWidget(self.create_status_group())
        
        # 模型配置组
        left_layout.addWidget(self.create_model_group())
        
        # 输入源组
        left_layout.addWidget(self.create_input_group())
        
        # 控制组
        left_layout.addWidget(self.create_control_group())
        
        # 检测信息组
        left_layout.addWidget(self.create_info_group())
        
        left_layout.addStretch()
        return left_panel
        
    def create_status_group(self):
        """创建系统状态组"""
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        
        status_info = QTextEdit()
        status_info.setPlainText("✅ 系统就绪\n✅ 模型加载完成\n✅ 摄像头可用\n⏹️ 等待输入源")
        status_info.setMaximumHeight(100)
        status_layout.addWidget(status_info)
        
        return status_group
        
    def create_model_group(self):
        """创建模型配置组"""
        model_group = QGroupBox("模型配置")
        model_layout = QVBoxLayout(model_group)
        
        # 权重选择
        weight_layout = QHBoxLayout()
        weight_label = QLabel("模型权重:")
        weight_label.setFont(QFont("Arial", 10, QFont.Bold))
        weight_layout.addWidget(weight_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolo(自动驾驶)", "yolo (标准版)", "yolo(快速版)"])
        self.model_combo.setCurrentIndex(1)
        weight_layout.addWidget(self.model_combo)
        model_layout.addLayout(weight_layout)
        
        # 参数设置
        model_layout.addLayout(self.create_params_layout())
        
        return model_group
        
    def create_params_layout(self):
        """创建参数设置布局"""
        params_layout = QVBoxLayout()
        
        # Confidence设置
        params_layout.addLayout(self.create_confidence_layout())
        
        # IOU设置
        params_layout.addLayout(self.create_iou_layout())
        
        return params_layout
        
    def create_confidence_layout(self):
        """创建置信度设置布局"""
        conf_layout = QVBoxLayout()
        conf_label = QLabel("置信度阈值 (Confidence):")
        conf_label.setFont(QFont("Arial", 9))
        conf_layout.addWidget(conf_label)
        
        conf_control_layout = QHBoxLayout()
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setFixedWidth(80)
        conf_control_layout.addWidget(self.conf_spin)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.setStyleSheet(
            "QSlider::groove:horizontal{height:8px;background:#34495e;border-radius:4px;}"
            "QSlider::handle:horizontal{background:#3498db;border:2px solid #2980b9;width:18px;margin:-6px 0;}"
        )
        conf_control_layout.addWidget(self.conf_slider)
        
        self.conf_value_label = QLabel(f"{self.conf_spin.value():.2f}")
        self.conf_value_label.setFixedWidth(50)
        self.conf_value_label.setAlignment(Qt.AlignCenter)
        conf_control_layout.addWidget(self.conf_value_label)
        
        # 绑定信号
        self.conf_spin.valueChanged.connect(lambda v: self.conf_slider.setValue(int(v * 100)))
        self.conf_slider.valueChanged.connect(lambda v: self.conf_spin.setValue(v / 100.0))
        self.conf_spin.valueChanged.connect(lambda v: self.conf_value_label.setText(f"{v:.2f}"))
        
        conf_layout.addLayout(conf_control_layout)
        return conf_layout
        
    def create_iou_layout(self):
        """创建IOU设置布局"""
        iou_layout = QVBoxLayout()
        iou_label = QLabel("交并比阈值 (IOU):")
        iou_label.setFont(QFont("Arial", 9))
        iou_layout.addWidget(iou_label)
        
        iou_control_layout = QHBoxLayout()
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
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
        
        # 绑定信号
        self.iou_spin.valueChanged.connect(lambda v: self.iou_slider.setValue(int(v * 100)))
        self.iou_slider.valueChanged.connect(lambda v: self.iou_spin.setValue(v / 100.0))
        self.iou_spin.valueChanged.connect(lambda v: self.iou_value_label.setText(f"{v:.2f}"))
        
        iou_layout.addLayout(iou_control_layout)
        return iou_layout
        
    def create_input_group(self):
        """创建输入源组"""
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
        
        return input_group
        
    def create_control_group(self):
        """创建控制组"""
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
        
        return control_group
        
    def create_info_group(self):
        """创建检测信息组"""
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
        
        return info_group
        
    def create_right_panel(self):
        """创建右侧显示面板"""
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
        
        return right_panel
