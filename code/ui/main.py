# -*- coding: UTF-8 -*-
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPalette, QColor, QFont
from aiwindow import AutonomousDrivingUI

if __name__ == "__main__":
    # 启用高DPI缩放
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    # 全局字体（优先中文显示效果，自动回退）
    app.setFont(QFont("Microsoft YaHei UI", 10))

    # 现代深色调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(35, 35, 35))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, Qt.black)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    # 全局样式（圆角、边距、交互反馈）
    app.setStyleSheet(
        """
        QWidget { background-color: #1e1e1e; color: #f0f0f0; }

        /* 文字与标题 */
        QLabel { font-size: 12px; }
        QGroupBox { border: 1px solid #333; border-radius: 8px; margin-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #d0d0d0; }

        /* 输入与下拉 */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit, QComboBox {
            background-color: #2b2b2b;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            padding: 6px 8px;
            selection-background-color: #2a82da;
        }
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QTimeEdit:focus, QDateTimeEdit:focus, QComboBox:focus {
            border-color: #2a82da;
        }
        QComboBox QAbstractItemView {
            background-color: #2b2b2b;
            selection-background-color: #2a82da;
            border: 1px solid #3a3a3a;
        }

        /* 按钮 */
        QPushButton {
            background-color: #2d2d2d;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            padding: 6px 12px;
        }
        QPushButton:hover { background-color: #343434; }
        QPushButton:pressed { background-color: #262626; }
        QPushButton:disabled { color: #8a8a8a; border-color: #2e2e2e; }

        /* 选项卡 */
        QTabWidget::pane { border: 1px solid #323232; border-radius: 8px; }
        QTabBar::tab {
            background: #2b2b2b;
            border: 1px solid #3a3a3a;
            padding: 6px 12px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
        }
        QTabBar::tab:selected { background: #323232; border-bottom-color: #323232; }
        QTabBar::tab:hover { background: #343434; }

        /* 表格 */
        QTableView, QTreeView {
            gridline-color: #3a3a3a;
            selection-background-color: #2a82da;
            selection-color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
        }
        QHeaderView::section {
            background-color: #2b2b2b;
            color: #dcdcdc;
            padding: 6px 8px;
            border: none;
            border-right: 1px solid #3a3a3a;
        }

        /* 滚动条（简洁版） */
        QScrollBar:vertical {
            background: #1e1e1e;
            width: 10px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #3a3a3a;
            border-radius: 5px;
            min-height: 24px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            background: #1e1e1e;
            height: 10px;
            margin: 0;
        }
        QScrollBar::handle:horizontal {
            background: #3a3a3a;
            border-radius: 5px;
            min-width: 24px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        """
    )

    window = AutonomousDrivingUI()
    window.show()

    sys.exit(app.exec_())