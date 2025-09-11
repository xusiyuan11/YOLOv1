# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - UI界面设置类
负责界面的布局和样式设置
"""
from PyQt5.QtWidgets import (QVBoxLayout, QHBoxLayout, QWidget, QLabel, QGroupBox,
                             QPushButton, QDoubleSpinBox, QSlider, QCheckBox, QProgressBar,
                             QTextEdit, QComboBox, QFrame, QScrollArea, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor, QLinearGradient, QGradient


class AutonomousDrivingUISetup:
    """UI界面设置基类"""

    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("🚗 自动驾驶目标检测系统 v2.0")
        self.setGeometry(100, 100, 1200, 900)  # 增加窗口高度确保内容显示完整
        self.setMinimumSize(1000, 800)  # 设置最小尺寸，允许用户调整

        # 设置应用程序样式
        self.set_app_style()

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)  # 适中间距
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 创建左侧面板
        left_panel = self.create_left_panel()

        # 创建右侧面板
        right_panel = self.create_right_panel()

        # 将左右面板添加到主布局，允许调整大小
        main_layout.addWidget(left_panel, 1)  # 左侧面板可调整
        main_layout.addWidget(right_panel, 2)  # 右侧面板占更多空间

        # 连接信号槽
        self.connect_signals()

    def set_app_style(self):
        """设置应用程序样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background: #ffffff;
            }

            /* 组框样式 */
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                border: 2px solid #dee2e6;
                border-radius: 15px;
                margin-top: 1.5ex;
                padding-top: 25px;
                background: #f8f9fa;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 8px 25px;
                background: #007bff;
                color: white;
                border-radius: 12px;
                font-size: 13px;
                font-weight: bold;
            }

            /* 按钮样式 - 蓝色（默认） */
            QPushButton {
                padding: 15px 25px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 13px;
                border: none;
                color: white;
                margin: 4px;
                background: #007bff;
            }

            QPushButton:hover {
                background: #0056b3;
            }

            QPushButton:pressed {
                background: #004085;
            }

            /* 绿色按钮 */
            QPushButton[buttonType="success"] {
                background: #28a745;
            }

            QPushButton[buttonType="success"]:hover {
                background: #1e7e34;
            }

            QPushButton[buttonType="success"]:pressed {
                background: #155724;
            }

            /* 红色按钮 */
            QPushButton[buttonType="danger"] {
                background: #dc3545;
            }

            QPushButton[buttonType="danger"]:hover {
                background: #c82333;
            }

            QPushButton[buttonType="danger"]:pressed {
                background: #bd2130;
            }

            /* 标签样式 */
            QLabel {
                color: #2c3e50;
                font-size: 12px;
                font-weight: 500;
            }

            /* 进度条样式 */
            QProgressBar {
                border: 2px solid #dee2e6;
                border-radius: 12px;
                text-align: center;
                background-color: #f8f9fa;
                height: 25px;
                font-weight: bold;
                color: #2c3e50;
            }

            QProgressBar::chunk {
                background: #007bff;
                border-radius: 10px;
            }

            /* 滑动条样式 */
            QSlider::groove:horizontal {
                border: 2px solid #dee2e6;
                height: 12px;
                background: #f8f9fa;
                border-radius: 8px;
            }

            QSlider::handle:horizontal {
                background: #007bff;
                border: 3px solid #ffffff;
                width: 24px;
                margin: -6px 0;
                border-radius: 12px;
            }

            QSlider::handle:horizontal:hover {
                background: #0056b3;
                border: 3px solid #ffffff;
            }

            /* 下拉框样式 */
            QComboBox {
                padding: 10px 15px;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                background: #ffffff;
                color: #2c3e50;
                selection-background-color: #007bff;
                font-size: 12px;
                font-weight: 500;
            }

            QComboBox:hover {
                border: 2px solid #007bff;
            }

            QComboBox::drop-down {
                border: none;
                background: #007bff;
                border-radius: 0 8px 8px 0;
                width: 30px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 10px solid white;
            }

            /* 复选框样式 */
            QCheckBox {
                color: #2c3e50;
                spacing: 12px;
                font-size: 12px;
                font-weight: 500;
            }

            QCheckBox::indicator {
                width: 22px;
                height: 22px;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                background: #ffffff;
            }

            QCheckBox::indicator:hover {
                border: 2px solid #007bff;
            }

            QCheckBox::indicator:checked {
                background: #007bff;
                border: 2px solid #0056b3;
            }

            QCheckBox::indicator:checked:hover {
                background: #0056b3;
                border: 2px solid #004085;
            }

            /* 双精度旋钮框样式 */
            QDoubleSpinBox {
                padding: 10px 15px;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                background: #ffffff;
                color: #2c3e50;
                selection-background-color: #007bff;
                font-size: 12px;
                font-weight: 500;
            }

            QDoubleSpinBox:hover {
                border: 2px solid #007bff;
            }

            QDoubleSpinBox:focus {
                border: 2px solid #007bff;
            }

            /* 文本编辑框样式 */
            QTextEdit {
                background: #ffffff;
                border: 2px solid #dee2e6;
                border-radius: 12px;
                color: #2c3e50;
                font-size: 12px;
                padding: 12px;
                font-weight: 500;
            }

            QTextEdit:hover {
                border: 2px solid #007bff;
            }

            QTextEdit:focus {
                border: 2px solid #007bff;
            }
        """)

    def create_left_panel(self):
        """创建左侧控制面板"""
        left_panel = QWidget()
        left_panel.setMinimumWidth(400)  # 设置最小宽度
        left_panel.setMaximumWidth(500)  # 设置最大宽度
        left_panel.setMinimumHeight(600)  # 调整最小高度适应窗口
        left_panel.setStyleSheet("""
            QWidget {
                background: #f8f9fa;
                border-radius: 15px;
                border: 1px solid #dee2e6;
            }
        """)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)  # 恢复合理间距
        left_layout.setContentsMargins(8, 8, 8, 8)

        # 模型配置组
        model_group = self.create_model_group()
        left_layout.addWidget(model_group)

        # 输入源组
        input_group = self.create_input_group()
        left_layout.addWidget(input_group)

        # 控制组
        control_group = self.create_control_group()
        left_layout.addWidget(control_group)

        # 检测信息组
        info_group = self.create_info_group()
        left_layout.addWidget(info_group)

        # 添加弹性空间，确保组件不会重叠
        left_layout.addStretch()
        return left_panel


    def create_model_group(self):
        """创建模型配置组"""
        model_group = QGroupBox("🤖 模型配置")
        model_group.setMinimumHeight(200)  # 大幅减少高度
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(10)
        model_layout.setContentsMargins(10, 10, 10, 10)

        # 权重选择
        weight_layout = QHBoxLayout()
        weight_label = QLabel("模型权重:")
        weight_label.setFont(QFont("Arial", 11, QFont.Bold))
        weight_label.setStyleSheet("color: #3498db;")
        weight_label.setFixedWidth(80)  # 固定标签宽度
        weight_layout.addWidget(weight_label)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8 (自动驾驶优化版)", "YOLOv5 (标准版)", "YOLOv7 (快速版)", "自定义模型"])
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setFixedHeight(35)  # 固定高度
        weight_layout.addWidget(self.model_combo)
        model_layout.addLayout(weight_layout)

        # 参数设置按钮布局
        params_button_layout = QHBoxLayout()
        params_button_layout.setSpacing(15)
        
        # 置信度阈值按钮
        self.conf_button = QPushButton("🎯 置信度阈值\n0.50")
        self.conf_button.setFixedSize(120, 60)
        self.conf_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2ecc71, stop: 1 #27ae60);
                color: white;
                font-size: 11px;
                font-weight: bold;
                border-radius: 10px;
                padding: 8px;
                text-align: center;
                border: 2px solid #1e8449;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #58d68d, stop: 1 #2ecc71);
                border: 2px solid #27ae60;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #27ae60, stop: 1 #1e8449);
                border: 2px solid #1e8449;
            }
        """)
        self.conf_button.clicked.connect(self.open_confidence_dialog)
        params_button_layout.addWidget(self.conf_button)
        
        # 交并比阈值按钮
        self.iou_button = QPushButton("📐 交并比阈值\n0.45")
        self.iou_button.setFixedSize(120, 60)
        self.iou_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #e74c3c, stop: 1 #c0392b);
                color: white;
                font-size: 11px;
                font-weight: bold;
                border-radius: 10px;
                padding: 8px;
                text-align: center;
                border: 2px solid #a93226;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ec7063, stop: 1 #e74c3c);
                border: 2px solid #c0392b;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #c0392b, stop: 1 #a93226);
                border: 2px solid #a93226;
            }
        """)
        self.iou_button.clicked.connect(self.open_iou_dialog)
        params_button_layout.addWidget(self.iou_button)
        
        model_layout.addLayout(params_button_layout)

        # 初始化参数值
        self.confidence_value = 0.50
        self.iou_value = 0.45

        return model_group

    def open_confidence_dialog(self):
        """打开置信度阈值调整弹窗"""
        dialog = QDialog(self)
        dialog.setWindowTitle("🎯 置信度阈值调整")
        dialog.setFixedSize(400, 250)
        dialog.setStyleSheet("""
            QDialog {
                background: #ffffff;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title_label = QLabel("🎯 置信度阈值 (Confidence)")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #2ecc71; text-align: center;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 当前值显示
        current_label = QLabel(f"当前值: {self.confidence_value:.2f}")
        current_label.setFont(QFont("Arial", 12))
        current_label.setStyleSheet("color: #2c3e50; text-align: center;")
        current_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(current_label)
        
        # 数值输入
        spin_layout = QHBoxLayout()
        spin_label = QLabel("数值:")
        spin_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        spin_label.setFixedWidth(50)
        spin_layout.addWidget(spin_label)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.confidence_value)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setFixedSize(100, 35)
        self.conf_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: #ffffff;
                color: #2c3e50;
                border: 2px solid #28a745;
                border-radius: 8px;
                padding: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QDoubleSpinBox:focus {
                border-color: #1e7e34;
            }
        """)
        spin_layout.addWidget(self.conf_spin)
        spin_layout.addStretch()
        layout.addLayout(spin_layout)
        
        # 滑块
        slider_layout = QHBoxLayout()
        slider_label = QLabel("滑块:")
        slider_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        slider_label.setFixedWidth(50)
        slider_layout.addWidget(slider_label)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.confidence_value * 100))
        self.conf_slider.setFixedHeight(30)
        self.conf_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #dee2e6;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #28a745;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #1e7e34;
            }
        """)
        slider_layout.addWidget(self.conf_slider)
        layout.addLayout(slider_layout)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #3498db, stop: 1 #2980b9);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5dade2, stop: 1 #3498db);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2980b9, stop: 1 #1f618d);
            }
        """)
        
        # 绑定信号
        self.conf_spin.valueChanged.connect(lambda v: self.conf_slider.setValue(int(v * 100)))
        self.conf_slider.valueChanged.connect(lambda v: self.conf_spin.setValue(v / 100.0))
        self.conf_spin.valueChanged.connect(lambda v: current_label.setText(f"当前值: {v:.2f}"))
        
        button_box.accepted.connect(lambda: self.apply_confidence_value(dialog))
        button_box.rejected.connect(dialog.reject)
        
        layout.addWidget(button_box)
        
        dialog.exec_()

    def apply_confidence_value(self, dialog):
        """应用置信度阈值"""
        self.confidence_value = self.conf_spin.value()
        self.conf_button.setText(f"🎯 置信度阈值\n{self.confidence_value:.2f}")
        dialog.accept()

    def open_iou_dialog(self):
        """打开交并比阈值调整弹窗"""
        dialog = QDialog(self)
        dialog.setWindowTitle("📐 交并比阈值调整")
        dialog.setFixedSize(400, 250)
        dialog.setStyleSheet("""
            QDialog {
                background: #ffffff;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title_label = QLabel("📐 交并比阈值 (IOU)")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #e74c3c; text-align: center;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 当前值显示
        current_label = QLabel(f"当前值: {self.iou_value:.2f}")
        current_label.setFont(QFont("Arial", 12))
        current_label.setStyleSheet("color: #2c3e50; text-align: center;")
        current_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(current_label)
        
        # 数值输入
        spin_layout = QHBoxLayout()
        spin_label = QLabel("数值:")
        spin_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        spin_label.setFixedWidth(50)
        spin_layout.addWidget(spin_label)
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(self.iou_value)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setFixedSize(100, 35)
        self.iou_spin.setStyleSheet("""
            QDoubleSpinBox {
                background: #ffffff;
                color: #2c3e50;
                border: 2px solid #dc3545;
                border-radius: 8px;
                padding: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QDoubleSpinBox:focus {
                border-color: #c82333;
            }
        """)
        spin_layout.addWidget(self.iou_spin)
        spin_layout.addStretch()
        layout.addLayout(spin_layout)
        
        # 滑块
        slider_layout = QHBoxLayout()
        slider_label = QLabel("滑块:")
        slider_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        slider_label.setFixedWidth(50)
        slider_layout.addWidget(slider_label)
        
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(0, 100)
        self.iou_slider.setValue(int(self.iou_value * 100))
        self.iou_slider.setFixedHeight(30)
        self.iou_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #dee2e6;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #dc3545;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                margin: -6px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #c82333;
            }
        """)
        slider_layout.addWidget(self.iou_slider)
        layout.addLayout(slider_layout)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #3498db, stop: 1 #2980b9);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5dade2, stop: 1 #3498db);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2980b9, stop: 1 #1f618d);
            }
        """)
        
        # 绑定信号
        self.iou_spin.valueChanged.connect(lambda v: self.iou_slider.setValue(int(v * 100)))
        self.iou_slider.valueChanged.connect(lambda v: self.iou_spin.setValue(v / 100.0))
        self.iou_spin.valueChanged.connect(lambda v: current_label.setText(f"当前值: {v:.2f}"))
        
        button_box.accepted.connect(lambda: self.apply_iou_value(dialog))
        button_box.rejected.connect(dialog.reject)
        
        layout.addWidget(button_box)
        
        dialog.exec_()

    def apply_iou_value(self, dialog):
        """应用交并比阈值"""
        self.iou_value = self.iou_spin.value()
        self.iou_button.setText(f"📐 交并比阈值\n{self.iou_value:.2f}")
        dialog.accept()


    def create_input_group(self):
        """创建输入源组"""
        input_group = QGroupBox("📥 输入源选择")
        input_group.setMinimumHeight(120)  # 恢复合理高度
        input_layout = QHBoxLayout(input_group)  # 改为水平布局
        input_layout.setSpacing(8)
        input_layout.setContentsMargins(8, 8, 8, 8)

        # 图像检测按钮
        self.btn_image = QPushButton("🖼️\n图像检测")
        self.btn_image.setFixedSize(100, 80)  # 设置固定方框尺寸
        self.btn_image.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #27ae60, stop: 1 #229954);
                font-size: 11px;
                font-weight: bold;
                border-radius: 12px;
                border: 2px solid #34495e;
                padding: 8px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2ecc71, stop: 1 #27ae60);
                border: 2px solid #3498db;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #229954, stop: 1 #1e8449);
                border: 2px solid #34495e;
            }
        """)
        input_layout.addWidget(self.btn_image)

        # 视频检测按钮
        self.btn_video = QPushButton("🎬\n视频检测")
        self.btn_video.setFixedSize(100, 80)  # 设置固定方框尺寸
        self.btn_video.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f39c12, stop: 1 #e67e22);
                font-size: 11px;
                font-weight: bold;
                border-radius: 12px;
                border: 2px solid #34495e;
                padding: 8px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f4d03f, stop: 1 #f39c12);
                border: 2px solid #3498db;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #e67e22, stop: 1 #d35400);
                border: 2px solid #34495e;
            }
        """)
        input_layout.addWidget(self.btn_video)

        # 实时摄像头按钮
        self.btn_camera = QPushButton("📹\n实时摄像头")
        self.btn_camera.setFixedSize(100, 80)  # 设置固定方框尺寸
        self.btn_camera.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #8e44ad, stop: 1 #6c3483);
                font-size: 11px;
                font-weight: bold;
                border-radius: 12px;
                border: 2px solid #34495e;
                padding: 8px;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #9b59b6, stop: 1 #8e44ad);
                border: 2px solid #3498db;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #6c3483, stop: 1 #5b2c6f);
                border: 2px solid #34495e;
            }
        """)
        input_layout.addWidget(self.btn_camera)

        return input_group

    def create_control_group(self):
        """创建控制组"""
        control_group = QGroupBox("🎮 系统控制")
        control_group.setMinimumHeight(140)  # 增加高度，有更多空间
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(8, 8, 8, 8)

        # 播放控制
        play_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶️ 开始检测")
        self.btn_play.setFixedHeight(50)  # 增加高度确保文字显示完整
        self.btn_play.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #e74c3c, stop: 1 #c0392b);
                font-size: 13px;
                font-weight: bold;
                border-radius: 12px;
                border: 2px solid #34495e;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #ec7063, stop: 1 #e74c3c);
                border: 2px solid #3498db;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #c0392b, stop: 1 #a93226);
                border: 2px solid #34495e;
            }
        """)
        play_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹️ 停止检测")
        self.btn_stop.setFixedHeight(50)  # 增加高度确保文字显示完整
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #7f8c8d, stop: 1 #616a6b);
                font-size: 13px;
                font-weight: bold;
                border-radius: 12px;
                border: 2px solid #34495e;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #95a5a6, stop: 1 #7f8c8d);
                border: 2px solid #3498db;
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #616a6b, stop: 1 #515a5a);
                border: 2px solid #34495e;
            }
        """)
        play_layout.addWidget(self.btn_stop)
        control_layout.addLayout(play_layout)

        # 保存选项
        save_layout = QHBoxLayout()
        self.cb_save = QCheckBox("💾 自动保存检测结果")
        self.cb_save.setChecked(True)
        save_layout.addWidget(self.cb_save)
        control_layout.addLayout(save_layout)

        return control_group

    def create_info_group(self):
        """创建检测信息组"""
        info_group = QGroupBox("📈 检测信息")
        info_group.setMinimumHeight(300)  # 增加高度以容纳更多信息
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(10, 10, 10, 10)

        # 检测结果文本显示区域
        self.detection_info = QTextEdit()
        self.detection_info.setMinimumHeight(250)
        self.detection_info.setStyleSheet("""
            QTextEdit {
                background: #ffffff;
                color: #2c3e50;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border-color: #007bff;
            }
        """)
        
        # 设置初始检测信息
        initial_text = """
🔍 自动驾驶目标检测系统 v2.0
═══════════════════════════════════════

📊 系统状态: 待机中
🎯 检测目标: 0 个
⏱️ 推理时间: 0 ms
📈 处理帧率: 0 FPS
📶 处理进度: 0%

🏷️ 检测类别统计:
   🚗 车辆: 0
   🚶 行人: 0  
   🚦 交通灯: 0
   🚧 标志牌: 0
   🚌 公交车: 0
   🚛 卡车: 0
   🏍️ 摩托车: 0
   🚲 自行车: 0

📋 最近检测结果:
   暂无检测数据

💡 系统提示:
   请选择输入源开始检测
        """
        
        self.detection_info.setPlainText(initial_text.strip())
        self.detection_info.setReadOnly(True)  # 设为只读
        info_layout.addWidget(self.detection_info)

        return info_group

    def create_right_panel(self):
        """创建右侧显示面板"""
        right_panel = QWidget()
        right_panel.setMinimumSize(600, 600)  # 调整最小高度适应窗口
        right_panel.setStyleSheet("""
            QWidget {
                background: #f8f9fa;
                border-radius: 15px;
                border: 1px solid #dee2e6;
            }
        """)

        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)  # 减少间距防止重叠
        right_layout.setContentsMargins(8, 8, 8, 8)

        # 视频显示组
        video_group = QGroupBox("👁️ 实时检测画面")
        video_group.setMinimumHeight(500)  # 增加高度，有更大显示区域
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(8, 8, 8, 8)
        video_layout.setSpacing(6)

        self.lbl_video = QLabel()
        self.lbl_video.setAlignment(Qt.AlignCenter)
        # 调整视频显示区域大小，适应窗口大小
        self.lbl_video.setMinimumSize(400, 300)  # 设置最小尺寸
        self.lbl_video.setScaledContents(False)  # 不自动缩放内容
        self.lbl_video.setStyleSheet("""
            QLabel {
                border: 3px solid #34495e;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #2c3e50, stop: 1 #34495e);
                color: #95a5a6;
                font-size: 16px;
                font-weight: bold;
                border-radius: 15px;
                padding: 15px;
            }
        """)
        self.lbl_video.setText("🎯 请选择输入源开始检测\n\n📸 支持图像、视频和实时摄像头\n🔍 点击下方按钮开始检测")
        video_layout.addWidget(self.lbl_video)
        right_layout.addWidget(video_group)

        # 系统状态和日志组
        status_console_group = self.create_status_console_group()
        right_layout.addWidget(status_console_group)

        return right_panel

    def create_status_console_group(self):
        """创建系统状态和日志合并组"""
        status_console_group = QGroupBox("📊 系统状态 & 📋 日志")
        status_console_group.setMinimumHeight(150)  # 设置最小高度
        main_layout = QVBoxLayout(status_console_group)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # 系统状态部分
        status_layout = QHBoxLayout()
        status_layout.setSpacing(8)
        
        # 状态指示器
        status_indicators = QVBoxLayout()
        status_indicators.setSpacing(4)
        
        # 系统状态标签
        self.lbl_system_status = QLabel("✅ 系统就绪")
        self.lbl_system_status.setStyleSheet("""
            QLabel {
                color: #2ecc71;
                background: rgba(46, 204, 113, 0.2);
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        status_indicators.addWidget(self.lbl_system_status)
        
        # 模型状态标签
        self.lbl_model_status = QLabel("📋 模型就绪")
        self.lbl_model_status.setStyleSheet("""
            QLabel {
                color: #3498db;
                background: rgba(52, 152, 219, 0.2);
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        status_indicators.addWidget(self.lbl_model_status)
        
        # 硬件状态标签
        self.lbl_hardware_status = QLabel("🔧 硬件检测通过")
        self.lbl_hardware_status.setStyleSheet("""
            QLabel {
                color: #f39c12;
                background: rgba(243, 156, 18, 0.2);
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
            }
        """)
        status_indicators.addWidget(self.lbl_hardware_status)
        
        status_layout.addLayout(status_indicators)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #34495e; margin: 0 8px;")
        status_layout.addWidget(separator)
        
        # 日志部分
        log_layout = QVBoxLayout()
        log_layout.setSpacing(4)
        
        log_label = QLabel("📋 系统日志:")
        log_label.setStyleSheet("color: #ecf0f1; font-size: 10px; font-weight: bold;")
        log_layout.addWidget(log_label)
        
        self.console = QTextEdit()
        self.console.setMinimumHeight(60)  # 设置最小高度
        self.console.setPlainText("🚀 系统启动成功\n📋 初始化完成\n✅ 准备就绪\n💡 请选择检测模式")
        self.console.setStyleSheet("font-size: 9px;")
        log_layout.addWidget(self.console)
        
        status_layout.addLayout(log_layout)
        main_layout.addLayout(status_layout)

        return status_console_group
