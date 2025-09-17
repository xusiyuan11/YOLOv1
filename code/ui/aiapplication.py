# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - 应用程序类
负责应用程序的初始化和全局设置
"""
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QCoreApplication
from aiwindow import AutonomousDrivingUI


class AIApp(QApplication):
    """自动驾驶AI应用程序类"""

    def __init__(self, argv=None):
        if argv is None:
            argv = sys.argv
        super().__init__(argv)

        # 启用高DPI缩放
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

        # 设置应用程序样式
        self.setStyle('Fusion')

        # 设置应用程序属性
        self.setApplicationName("目标检测系统")
        self.setApplicationVersion("1.0")
        self.setOrganizationName("AI Detection Team")

        # 创建主窗口
        self.main_window = AutonomousDrivingUI()

    def setup_global_style(self):
        """设置全局样式表"""
        self.setStyleSheet("""
            QApplication {
                font-family: "Microsoft YaHei", Arial, sans-serif;
                font-size: 9pt;
            }
        """)

    def show_main_window(self):
        """显示主窗口"""
        self.main_window.show()