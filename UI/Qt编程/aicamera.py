# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - 摄像头和模拟功能类
负责摄像头操作和检测结果模拟
"""
import random
import cv2
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt5.QtCore import Qt


class AICamera(QThread):
    """真实摄像头线程类"""
    sigvideo = pyqtSignal(int, int, int, bytes)  
    
    def __init__(self):
        super(AICamera, self).__init__()
        self.camera = None
        self.isrunning = False
        self.init_camera()
        
    def init_camera(self):
        """初始化摄像头"""
            
        try:
            # 确保之前的摄像头已释放
            if self.camera and self.camera.isOpened():
                self.camera.release()
                
            # 创建新的摄像头对象
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            # 设置摄像头参数
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                print("摄像头初始化成功")
            else:
                print("摄像头初始化失败")
        except Exception as e:
            print(f"摄像头初始化异常: {e}")
            self.camera = None
    
    def run(self):
        """摄像头工作线程"""
        self.isrunning = True
        while self.isrunning and self.camera and self.camera.isOpened():
            # 抓屏
            status, img = self.camera.read()
            if status:
                h, w, c = img.shape
                self.sigvideo.emit(h, w, c, img.tobytes())  # 发送信号
                # print(f"摄像头读取成功({h:02d},{w:d},{c:d})")
            else:
                print("摄像头读取错误")
                break
            
            # 控制帧率
            QThread.usleep(33000)  # 约30fps
        
        print("摄像头线程结束")
        
    def close(self):
        """关闭摄像头"""
        print("正在关闭摄像头...")
        self.isrunning = False
        
        # 等待线程结束
        if self.isRunning():
            self.quit()
            self.wait(3000)  # 等待最多3秒
            
        # 释放摄像头资源
        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None
            print("摄像头资源已释放")
            
    def is_camera_available(self):
        """检查摄像头是否可用"""
        return self.camera is not None and self.camera.isOpened()


class CameraSimulation:
    """摄像头和模拟功能基类"""
    
    def setup_simulation(self):
        """设置模拟定时器和摄像头"""
        # 模拟定时器
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)
        self.sim_timer.setInterval(100)  # 10 FPS
        self.simulation_running = False
        self.frame_count = 0
        
        # 真实摄像头（初始化为None，每次使用时创建新对象）
        self.real_camera = None
        self.camera_running = False
        
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
            
    def start_camera(self):
        """启动真实摄像头"""
    
        if self.camera_running:
            self.console.append("📹 摄像头已在运行中")
            return True
            
        # 每次启动时创建新的摄像头对象
        self.real_camera = AICamera()
        
        if self.real_camera.is_camera_available():
            self.real_camera.sigvideo.connect(self.show_real_video)
            self.camera_running = True
            self.real_camera.start()
            self.console.append("📹 摄像头启动成功")
            return True
        else:
            self.console.append("❌ 摄像头不可用，启动模拟模式")
            self.real_camera = None
            # 启动模拟模式作为替代
            self.start_simulation()
            return False
            
    def stop_camera(self):
        """停止真实摄像头"""
        if self.camera_running and self.real_camera:
            self.camera_running = False
            self.real_camera.close()
            # 等待线程完全结束
            self.real_camera.wait()
            self.real_camera = None
            self.console.append("📹 摄像头已停止")
            
    def show_real_video(self, h, w, c, data):
        """显示真实摄像头画面"""
        # 处理图像
        # 1. 把二进制编码成Qt的QImage对象
        qt_img = QImage(data, w, h, c * w, QImage.Format_BGR888)
        # 2. 把QImage转成像素图 PixelMap格式
        qt_pm = QPixmap.fromImage(qt_img)
        # 3. 显示QPixelMap到QLabel
        scaled_pixmap = qt_pm.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_video.setPixmap(scaled_pixmap)
        
        # 更新实时统计信息
        self.update_real_camera_stats()
        
    def update_real_camera_stats(self):
        """更新真实摄像头统计信息"""
        # 这里可以添加真实的目标检测逻辑
        # 目前使用模拟数据
        targets = random.randint(1, 8)
        inference_time = random.randint(25, 60)
        fps = random.randint(28, 32)
        
        self.lbl_targets.setText(str(targets))
        self.lbl_time.setText(f"{inference_time} ms")
        self.lbl_fps.setText(f"{fps} FPS")
        
        # 更新类别统计
        classes_text = f"🚗 车辆: {random.randint(0, 5)}\n🚶 行人: {random.randint(0, 3)}\n🚦 交通灯: {random.randint(0, 2)}\n🚧 标志牌: {random.randint(0, 2)}"
        self.classes_info.setPlainText(classes_text)
        
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