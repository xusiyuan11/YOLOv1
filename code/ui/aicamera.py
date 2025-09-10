# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - 摄像头和视频功能类
负责摄像头操作和视频播放
"""
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage


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
            status, img = self.camera.read()
            if status:
                h, w, c = img.shape
                self.sigvideo.emit(h, w, c, img.tobytes())
            else:
                print("摄像头读取错误")
                break
            QThread.usleep(33000)  # 约30fps

        print("摄像头线程结束")

    def close(self):
        """关闭摄像头"""
        print("正在关闭摄像头...")
        self.isrunning = False

        if self.isRunning():
            self.quit()
            self.wait(3000)

        if self.camera and self.camera.isOpened():
            self.camera.release()
            self.camera = None
            print("摄像头资源已释放")

    def is_camera_available(self):
        """检查摄像头是否可用"""
        return self.camera is not None and self.camera.isOpened()


class VideoThread(QThread):
    """视频播放线程"""
    frame_signal = pyqtSignal(QImage)
    finished_signal = pyqtSignal()

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_path)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.frame_signal.emit(qt_image)
            QThread.msleep(33)  # 约30fps

        cap.release()
        self.finished_signal.emit()

    def stop(self):
        self.running = False


class CameraVideoHandler:
    """摄像头和视频处理基类"""

    def setup_camera_video(self):
        self.real_camera = None
        self.camera_running = False
        self.video_thread = None
        self.current_image_path = None

    def start_camera(self):
        if self.camera_running:
            self.console.append("📹 摄像头已在运行中")
            return True

        self.real_camera = AICamera()

        if self.real_camera.is_camera_available():
            self.real_camera.sigvideo.connect(self.show_real_video)
            self.camera_running = True
            self.real_camera.start()
            self.console.append("📹 摄像头启动成功")
            return True
        else:
            self.console.append("❌ 摄像头不可用")
            self.real_camera = None
            return False

    def stop_camera(self):
        if self.camera_running and self.real_camera:
            self.camera_running = False
            self.real_camera.close()
            self.real_camera.wait()
            self.real_camera = None
            self.console.append("📹 摄像头已停止")

    def show_real_video(self, h, w, c, data):
        """显示真实摄像头画面"""
        try:
            # data 是 BGR，需要转 RGB
            frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            qt_img = QImage(rgb_image.data, w, h, c * w, QImage.Format_RGB888)
            qt_pm = QPixmap.fromImage(qt_img)
            scaled_pixmap = qt_pm.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_video.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"显示摄像头画面错误: {e}")

    def display_image(self, image_path):
        """显示图像（支持中文路径）"""
        try:
            image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self.console.append(f"❌ 无法读取图像文件: {image_path}")
                return False

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_video.setPixmap(scaled_pixmap)
            self.console.append("✅ 图像显示成功")
            return True

        except Exception as e:
            self.console.append(f"❌ 显示图像失败: {str(e)}")
            return False

    def display_video_frame(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_video.setPixmap(scaled_pixmap)

    def start_video_playback(self, video_path):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()

        self.video_thread = VideoThread(video_path)
        self.video_thread.frame_signal.connect(self.display_video_frame)
        self.video_thread.finished_signal.connect(self.on_video_finished)
        self.video_thread.start()
        self.console.append("▶️ 开始视频播放")

    def on_video_finished(self):
        self.console.append("⏹️ 视频播放完成")

    def stop_video_playback(self):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
            self.video_thread = None
            self.console.append("⏹️ 视频播放已停止")

    def stop_all(self):
        if self.camera_running:
            self.stop_camera()
        self.stop_video_playback()
