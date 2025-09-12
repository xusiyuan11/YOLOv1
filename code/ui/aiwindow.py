# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - 主窗口类
"""
import sys
import os
import time
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from aiui import AutonomousDrivingUISetup
from aicamera import CameraVideoHandler
import torch
import cv2
from PyQt5.QtGui import QPixmap, QImage


class InferenceThread(QThread):
    """后台推理线程：接收最新一帧，按设置节流后进行YOLOv5推理并发回叠加后的QImage"""

    frame_ready = pyqtSignal(QImage)
    meta_ready = pyqtSignal(str)

    def __init__(self, model, infer_size: int, frame_skip: int, interval_ms: int):
        super().__init__()
        self.model = model
        self.infer_size = infer_size
        self.frame_skip = max(1, int(frame_skip))
        self.interval_ms = max(0, int(interval_ms))
        self._running = True
        self._busy = False
        self._last_infer_ms = 0
        self._frame_index = 0
        self._latest_frame = None  # BGR np.ndarray

    def update_params(self, infer_size: int = None, frame_skip: int = None, interval_ms: int = None):
        if infer_size is not None:
            self.infer_size = int(infer_size)
        if frame_skip is not None:
            self.frame_skip = max(1, int(frame_skip))
        if interval_ms is not None:
            self.interval_ms = max(0, int(interval_ms))

    def submit_frame(self, frame_bgr):
        # 确保持有独立副本，避免引用临时缓冲导致崩溃
        try:
            self._latest_frame = frame_bgr.copy()
        except Exception:
            self._latest_frame = frame_bgr

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            try:
                frame = self._latest_frame
                if frame is None:
                    self.msleep(5)
                    continue

                # 帧采样
                self._frame_index += 1
                if (self._frame_index % self.frame_skip) != 0:
                    self.msleep(1)
                    continue

                now_ms = int(time.time() * 1000)
                if self._busy or (now_ms - self._last_infer_ms < self.interval_ms):
                    self.msleep(1)
                    continue

                self._busy = True
                try:
                    with torch.no_grad():
                        results = self.model(frame, size=self.infer_size)
                    annotated_bgr = results.render()[0]
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = annotated_rgb.shape
                    # 构造独立QImage，防止指针悬挂
                    qt_img = QImage(annotated_rgb.copy().data, w, h, ch * w, QImage.Format_RGB888).copy()
                    self.frame_ready.emit(qt_img)
                    # 生成检测文本（类别/时间/位置/置信度）
                    try:
                        pred = results.xyxy[0]
                        names = getattr(results, 'names', getattr(self.model, 'names', {}))
                        if pred is not None and pred.shape[0] > 0:
                            ts_ms = int(time.time() * 1000)
                            ts = time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000)) + f".{ts_ms % 1000:03d}"
                            lines = []
                            for i in range(int(pred.shape[0])):
                                x1, y1, x2, y2 = [int(v) for v in pred[i, :4].detach().cpu().numpy().tolist()]
                                conf = float(pred[i, 4])
                                cls = int(pred[i, 5])
                                label = names[cls] if isinstance(names, (list, dict)) else f'class{cls}'
                                lines.append(
                                    f"{i + 1}. 时间 {ts} | 类别 {label} | 位置 ({x1},{y1},{x2},{y2}) | 置信度 {conf:.2f}"
                                )
                            self.meta_ready.emit("\n".join(lines))
                        # 无检测时不输出
                    except Exception:
                        pass
                    self._last_infer_ms = now_ms
                finally:
                    self._busy = False
            except Exception:
                # 出错时短暂休眠，避免busy loop
                self.msleep(5)


class AutonomousDrivingUI(QMainWindow, AutonomousDrivingUISetup, CameraVideoHandler):
    """目标检测系统主窗口"""

    def __init__(self):
        super().__init__()
        # 初始化UI
        self.setup_ui()
        # 初始化摄像头和视频处理
        self.setup_camera_video()

        # 检测开关（开始/停止）
        self.detecting = False
        # 推理节流与分辨率
        self.infer_interval_ms = 150  # 最小推理间隔，防止频繁卡顿
        self.last_infer_ts = 0
        self.infer_size = 512  # CPU 默认使用较低分辨率，GPU 会在加载后自动提升
        self._infer_busy = False
        # 按帧检测（确保连续多帧动态检测）
        self.frame_skip = 2  # CPU 默认每2帧检测一次；GPU下会改为每帧
        self.video_frame_index = 0
        self.camera_frame_index = 0
        # 后台推理线程
        self.infer_thread = None
        # 最近一次叠加后的结果帧（用于持续显示，避免闪烁）
        self._last_annotated_qimage = None
        # UI 刷新节流（按目标帧率限制标签刷新频率）
        self._last_ui_update_ms = 0
        self.ui_min_update_interval_ms = 1000 // 30

        # 加载YOLOv5模型
        self.model = self.load_model()

        # 初始化统计信息
        self.recent_detections = []  # 存储最近检测结果
        self.reset_stats()

    def reset_stats(self):
        """重置统计信息"""
        self.update_detection_info(0, 0, 0, 0, {})

    def update_detection_info(self, targets, inference_time, fps, progress, class_counts, status="待机中"):
        """不再构建固定信息文本，保持与用户要求一致，由推理线程追加日志"""
        return

    def get_recent_detections(self):
        """获取最近检测结果"""
        if hasattr(self, 'recent_detections') and self.recent_detections:
            return "\n   ".join(self.recent_detections[-3:])  # 显示最近3条
        return "暂无检测数据"

    def get_system_tip(self, status):
        """获取系统提示"""
        tips = {
            "待机中": "请选择输入源开始检测",
            "检测中": "正在处理输入数据，请稍候...",
            "完成": "检测完成，可查看结果",
            "错误": "检测过程中出现错误，请检查输入"
        }
        return tips.get(status, "系统运行正常")

    def connect_signals(self):
        """连接所有信号和槽函数"""
        # 参数控制信号 - 现在通过按钮弹窗处理，不需要直接连接
        # self.conf_spin.valueChanged.connect(self.on_conf_changed)
        # self.conf_slider.valueChanged.connect(self.on_conf_slider_changed)
        # self.iou_spin.valueChanged.connect(self.on_iou_changed)
        # self.iou_slider.valueChanged.connect(self.on_iou_slider_changed)

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
        self.console.append(f"🎯 Confidence阈值调整为: {value:.2f}")

    def on_conf_slider_changed(self, value):
        """Confidence滑块改变"""
        self.conf_spin.setValue(value / 100.0)

    def on_iou_changed(self, value):
        """IOU值改变"""
        self.iou_slider.setValue(int(value * 100))
        self.console.append(f"📐 IOU阈值调整为: {value:.2f}")

    def on_iou_slider_changed(self, value):
        """IOU滑块改变"""
        self.iou_spin.setValue(value / 100.0)

    def load_model(self):
        """加载YOLOv5模型（优先本地仓库），支持自定义权重"""
        # 推测本地YOLOv5仓库位置：.../YOLOv5/yolov5-master
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        default_repo_paths = [
            os.path.join(base_dir, 'YOLOv5', 'yolov5-master'),
            os.path.join(base_dir, 'yolov5'),
        ]

        yolo_repo = next((p for p in default_repo_paths if os.path.isdir(p)), None)
        # 避免 torch.hub 缓存导致加载旧代码，强制本地优先时可关闭缓存
        try:
            torch.hub.set_dir(os.path.join(base_dir, '.torchhub'))
        except Exception:
            pass
        if yolo_repo is None:
            self.console.append("⚠️ 未找到本地YOLOv5仓库，将尝试联网加载（可能较慢）")

        # 可能的默认权重文件候选（优先使用用户提供的 yolov5x.pt）
        default_weight_candidates = [
            os.path.join(base_dir, 'YOLOv5', 'yolov5-master', 'yolov5x.pt'),
            os.path.join(base_dir, 'best.pt'),
            os.path.join(base_dir, 'weights', 'best.pt'),
            os.path.join(base_dir, 'yolov5s.pt'),
            os.path.join(base_dir, 'YOLOv5', 'yolov5-master', 'yolov5s.pt'),
        ]
        weights_path = next((w for w in default_weight_candidates if os.path.isfile(w)), None)

        if weights_path is None:
            # 让用户选择权重
            self.console.append("📂 请选择YOLOv5权重文件 (*.pt)")
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择YOLOv5权重文件", base_dir, "PyTorch 权重 (*.pt)"
            )
            weights_path = file_path if file_path else None

        try:
            self.console.append(f"🧩 Torch 版本: {getattr(torch, '__version__', 'unknown')}")
            self.console.append(f"📁 仓库路径: {yolo_repo if yolo_repo else 'remote: ultralytics/yolov5'}")
            self.console.append(f"📦 权重路径: {weights_path if weights_path else '预训练 yolov5s'}")

            if weights_path:
                if yolo_repo and os.path.isdir(yolo_repo):
                    try:
                        model = torch.hub.load(yolo_repo, 'custom', path=weights_path, source='local', trust_repo=True)
                    except TypeError:
                        model = torch.hub.load(yolo_repo, 'custom', path=weights_path, source='local')
                else:
                    try:
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, trust_repo=True)
                    except TypeError:
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
                self.console.append(f"✅ YOLOv5 模型加载成功: {os.path.basename(weights_path)}")
            else:
                # 无自定义权重，回退到yolov5s
                if yolo_repo and os.path.isdir(yolo_repo):
                    try:
                        model = torch.hub.load(yolo_repo, 'yolov5s', source='local', trust_repo=True)
                    except TypeError:
                        model = torch.hub.load(yolo_repo, 'yolov5s', source='local')
                else:
                    try:
                        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
                    except TypeError:
                        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                self.console.append("✅ 已加载预训练模型 yolov5s")

            # 设置阈值（与UI保持一致），兼容不同版本API
            try:
                if hasattr(self, 'confidence_value') and hasattr(model, 'conf'):
                    model.conf = float(self.confidence_value)
                if hasattr(self, 'iou_value') and hasattr(model, 'iou'):
                    model.iou = float(self.iou_value)
            except Exception:
                pass

            # 推理设备
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model.to(device)
                self.console.append(f"🖥️ 推理设备: {device}")
                # GPU 用高分辨率，CPU 用较低分辨率
                self.infer_size = 640 if device == 'cuda' else 512
                self.frame_skip = 1 if device == 'cuda' else 2
            except Exception:
                pass

            # 模型预热：避免首帧推理卡顿
            try:
                import numpy as _np
                dummy = _np.zeros((self.infer_size, self.infer_size, 3), dtype=_np.uint8)
                with torch.no_grad():
                    _ = model(dummy, size=self.infer_size)
            except Exception:
                pass

            # 更新模型状态标签
            if hasattr(self, 'lbl_model_status'):
                self.lbl_model_status.setText("📋 模型就绪 (YOLOv5)")

            return model
        except Exception as e:
            self.console.append(f"❌ YOLOv5 模型加载失败: {str(e)}")
            return None

    def detect_image(self, image_path):
        """使用YOLOv5检测图像并显示标注结果（支持中文路径）"""
        try:
            if self.model is None:
                self.console.append("⚠️ 未加载模型，无法进行检测")
                return

            # 读取原始图像（BGR），支持中文路径
            bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                self.console.append(f"❌ 无法读取图像文件: {image_path}")
                return

            start_ts = time.time()
            # 设置阈值热更新（防止用户调整后未生效）
            try:
                if hasattr(self, 'confidence_value') and hasattr(self.model, 'conf'):
                    self.model.conf = float(self.confidence_value)
                if hasattr(self, 'iou_value') and hasattr(self.model, 'iou'):
                    self.model.iou = float(self.iou_value)
            except Exception:
                pass

            # 推理（节流不需要，因为单次图像检测）
            with torch.no_grad():
                results = self.model(bgr, size=self.infer_size)
            inference_ms = int((time.time() - start_ts) * 1000)

            # 渲染标注（BGR）
            rendered = results.render()[0]

            # 显示到UI（转RGB再显示）
            rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_video.setPixmap(scaled_pixmap)

            # 解析结果并更新统计信息
            targets, class_counts, recent_text = self._parse_yolo_results(results)
            self.recent_detections.append(recent_text)
            self.update_detection_info(targets, inference_ms, 0, 100, class_counts, status="完成")

            # 自动保存结果
            if hasattr(self, 'cb_save') and self.cb_save.isChecked():
                save_path = os.path.splitext(image_path)[0] + "_det.jpg"
                cv2.imencode('.jpg', rendered)[1].tofile(save_path)
                self.console.append(f"💾 结果已保存: {os.path.basename(save_path)}")

            self.console.append("✅ 图像检测完成")

        except Exception as e:
            self.console.append(f"❌ 图像检测失败: {str(e)}")

    def _parse_yolo_results(self, results):
        """从YOLOv5结果中统计目标数与类别分布，返回(总数, 类别统计, 最近结果文本)"""
        try:
            # 轻量解析，避免首帧引入 pandas 的开销
            pred = results.xyxy[0]  # tensor: [N, 6] -> x1,y1,x2,y2,conf,cls
            total = int(pred.shape[0]) if pred is not None else 0

            name_map = {
                'car': '车辆',
                'person': '行人',
                'traffic light': '交通灯',
                'stop sign': '标志牌',
                'bus': '公交车',
                'truck': '卡车',
                'motorcycle': '摩托车',
                'bicycle': '自行车',
            }

            class_counts = {}
            recent_text = "无检测"
            if total > 0:
                conf = pred[:, 4].detach().cpu().numpy()
                cls = pred[:, 5].detach().cpu().numpy().astype(int)
                names = getattr(results, 'names', getattr(self.model, 'names', {}))

                import numpy as _np
                unique_cls = _np.unique(cls)
                for ci in unique_cls:
                    en = names[int(ci)] if isinstance(names, (list, dict)) else f'class{int(ci)}'
                    cn = name_map.get(en, en)
                    class_counts[cn] = int((_np.array(cls) == ci).sum())

                # 取置信度最高的前3个
                top_idx = _np.argsort(-conf)[:3]
                labels = []
                for i in top_idx:
                    en = names[int(cls[i])] if isinstance(names, (list, dict)) else f'class{int(cls[i])}'
                    cn = name_map.get(en, en)
                    labels.append(f"{cn} {conf[i]:.2f}")
                if labels:
                    recent_text = ", ".join(labels)

            return total, class_counts, recent_text
        except Exception:
            return 0, {}, "解析失败"

    # 覆盖摄像头帧显示以加入实时检测（可由开始/停止按钮控制）
    def show_real_video(self, h, w, c, data):
        try:
            frame_bgr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))
            # 将帧送入后台线程进行推理
            if self.detecting and self.model is not None and self.infer_thread is not None:
                self.infer_thread.submit_frame(frame_bgr)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_qimage = QImage(frame_rgb.data, w, h, c * w, QImage.Format_RGB888)

            # 检测中优先显示最近的叠加结果，避免原始帧覆盖造成“闪烁”；并限制刷新频率
            now_ms = int(time.time() * 1000)
            if now_ms - self._last_ui_update_ms >= self.ui_min_update_interval_ms:
                if self.detecting and self._last_annotated_qimage is not None:
                    pixmap = QPixmap.fromImage(self._last_annotated_qimage)
                else:
                    pixmap = QPixmap.fromImage(raw_qimage)
                scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
                self.lbl_video.setPixmap(scaled_pixmap)
                self._last_ui_update_ms = now_ms
        except Exception as e:
            print(f"显示摄像头画面错误: {e}")

    def display_video_frame(self, qt_image):
        """视频播放帧回调：在实时检测开启时执行YOLOv5推理并叠加结果"""
        try:
            w = qt_image.width()
            h = qt_image.height()
            ch = 3

            # 将帧送入后台线程进行推理
            if self.detecting and self.model is not None and self.infer_thread is not None:
                bits = qt_image.bits()
                bits.setsize(h * w * ch)
                rgb = np.frombuffer(bits, dtype=np.uint8).reshape((h, w, ch))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self.infer_thread.submit_frame(bgr)

            # 检测中优先显示最近的叠加结果，避免原始帧覆盖造成“闪烁”；并限制刷新频率
            now_ms = int(time.time() * 1000)
            if now_ms - self._last_ui_update_ms >= self.ui_min_update_interval_ms:
                if self.detecting and self._last_annotated_qimage is not None:
                    pixmap = QPixmap.fromImage(self._last_annotated_qimage)
                else:
                    pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
                self.lbl_video.setPixmap(scaled_pixmap)
                self._last_ui_update_ms = now_ms
        except Exception as e:
            print(f"显示视频帧错误: {e}")

    # 输入源选择回调函数
    def on_image_clicked(self):
        """图像检测按钮点击"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.stop_all()  # 停止其他模式
            self.current_image_path = file_path
            self.console.append(f"🖼️ 已选择图像: {os.path.basename(file_path)}")
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
            # 开始后台推理线程
            self._start_infer_thread()

    def on_camera_clicked(self):
        """摄像头按钮点击"""
        if not self.camera_running:
            # 停止其他模式
            self.stop_video_playback()

            # 尝试启动真实摄像头
            if self.start_camera():
                self.console.append("📹 启动真实摄像头检测")
                self.btn_camera.setText("📹 停止摄像头")
                self.btn_camera.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                            stop: 0 #e74c3c, stop: 1 #c0392b);
                        font-size: 13px;
                    }
                """)
                # 重置统计信息
                self.reset_stats()
            else:
                self.console.append("❌ 无法启动摄像头")
        else:
            # 停止摄像头
            self.stop_camera()
            self.btn_camera.setText("📹 实时摄像头")
            self.btn_camera.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #8e44ad, stop: 1 #6c3483);
                    font-size: 13px;
                }
            """)

    # 控制回调函数
    def on_play_clicked(self):
        """开始检测按钮点击"""
        # 开启检测模式（用于摄像头/视频）
        self.detecting = True
        self._start_infer_thread()
        # 如勾选保存结果，询问保存路径（视频/摄像头）
        if hasattr(self, 'cb_save') and self.cb_save.isChecked():
            if (self.camera_running or (self.video_thread and self.video_thread.isRunning())):
                self._prompt_save_video_path()
        if self.camera_running:
            self.console.append("▶️ 已启用摄像头实时检测")
        elif self.video_thread and self.video_thread.isRunning():
            self.console.append("▶️ 已启用视频实时检测")
        elif self.current_image_path:
            self.console.append("▶️ 开始图像检测")
            self.detect_image(self.current_image_path)
        else:
            self.console.append("⚠️ 请先选择输入源")

    def on_stop_clicked(self):
        """停止检测按钮点击"""
        # 关闭实时检测
        self.detecting = False
        self.stop_all()
        self._stop_infer_thread()
        self.console.append("⏹️ 停止所有检测")
        # 重置统计信息
        self.reset_stats()
        # 停止后不再保留历史输出（根据需要可注释掉）
        self.detection_info.setPlainText("")
        # 清空叠加帧缓存
        self._last_annotated_qimage = None
        # 关闭视频写入
        self._close_video_writer()

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_all()
        self._stop_infer_thread()
        self._last_annotated_qimage = None
        event.accept()

    def _start_infer_thread(self):
        """启动或更新后台推理线程参数"""
        if self.model is None:
            return
        if self.infer_thread is None:
            self.infer_thread = InferenceThread(self.model, self.infer_size, self.frame_skip, self.infer_interval_ms)
            self.infer_thread.frame_ready.connect(self._on_infer_frame_ready)
            self.infer_thread.meta_ready.connect(self._on_infer_meta_ready)
            # 设为守护线程，避免进程退出阻塞
            try:
                self.infer_thread.setObjectName("InferenceThread")
            except Exception:
                pass
            self.infer_thread.start()
        else:
            self.infer_thread.update_params(self.infer_size, self.frame_skip, self.infer_interval_ms)

    def _stop_infer_thread(self):
        if self.infer_thread is not None:
            try:
                self.infer_thread.stop()
                self.infer_thread.wait(2000)
            except Exception:
                pass
            self.infer_thread = None

    def _on_infer_frame_ready(self, qt_image: QImage):
        """收到后台推理结果帧，非阻塞更新UI"""
        try:
            # 缓存最新的叠加结果，供原始帧到达时也优先显示
            self._last_annotated_qimage = qt_image
            now_ms = int(time.time() * 1000)
            if now_ms - self._last_ui_update_ms >= self.ui_min_update_interval_ms:
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(self.lbl_video.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
                self.lbl_video.setPixmap(scaled_pixmap)
                self._last_ui_update_ms = now_ms

            # 写入视频结果
            if getattr(self, 'saving_video', False) and getattr(self, 'save_video_path', None):
                w = qt_image.width()
                h = qt_image.height()
                ch = 3
                bits = qt_image.bits()
                bits.setsize(h * w * ch)
                rgb = np.frombuffer(bits, dtype=np.uint8).reshape((h, w, ch))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if getattr(self, 'video_writer', None) is None or getattr(self, '_video_writer_size', None) != (w, h):
                    self._open_video_writer(w, h)
                if getattr(self, 'video_writer', None) is not None:
                    self.video_writer.write(bgr)
        except Exception:
            pass

    def _prompt_save_video_path(self):
        try:
            default_name = time.strftime("%Y%m%d_%H%M%S") + ".mp4"
            path, _ = QFileDialog.getSaveFileName(self, "保存检测视频为", default_name, "视频文件 (*.mp4 *.avi)")
            if path:
                self.save_video_path = path
                self.saving_video = True
                self.console.append(f"💾 保存路径: {os.path.basename(path)}")
            else:
                self.saving_video = False
                self.save_video_path = None
        except Exception as e:
            self.console.append(f"❌ 选择保存路径失败: {str(e)}")
            self.saving_video = False
            self.save_video_path = None

    def _open_video_writer(self, w: int, h: int):
        try:
            if not getattr(self, 'saving_video', False) or not getattr(self, 'save_video_path', None):
                return
            # 关闭旧的
            self._close_video_writer()
            # fourcc 根据扩展名选择
            ext = os.path.splitext(self.save_video_path)[1].lower()
            if ext == ".avi":
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(getattr(self, 'target_fps', 20))
            self.video_writer = cv2.VideoWriter(self.save_video_path, fourcc, max(1, fps), (w, h))
            self._video_writer_size = (w, h)
            if not self.video_writer.isOpened():
                self.console.append("❌ 无法创建视频写入器")
                self.video_writer = None
        except Exception as e:
            self.console.append(f"❌ 创建视频写入器失败: {str(e)}")
            self.video_writer = None

    def _close_video_writer(self):
        try:
            if getattr(self, 'video_writer', None) is not None:
                self.video_writer.release()
            if getattr(self, 'save_video_path', None) and getattr(self, 'saving_video', False):
                self.console.append(f"✅ 已保存视频: {os.path.basename(self.save_video_path)}")
        except Exception:
            pass
        finally:
            self.video_writer = None
            self._video_writer_size = None
            self.saving_video = False

    def _on_infer_meta_ready(self, text: str):
        try:
            # 逐条追加所有检测记录，不清空历史
            curr = self.detection_info.toPlainText()
            new_text = (curr + "\n" + text).strip() if curr else text
            self.detection_info.setPlainText(new_text)
            # 滚动到末尾
            self.detection_info.moveCursor(self.detection_info.textCursor().End)
        except Exception:
            pass