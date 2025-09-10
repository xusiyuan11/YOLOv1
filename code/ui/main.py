# -*- coding: UTF-8 -*-
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QCoreApplication
from aiwindow import AutonomousDrivingUI

if __name__ == "__main__":
    # 启用高DPI缩放
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle('Fusion')

    window = AutonomousDrivingUI()
    window.show()

    sys.exit(app.exec_())
