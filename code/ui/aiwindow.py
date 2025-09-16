# -*- coding: UTF-8 -*-
"""
自动驾驶目标检测系统 - 主窗口类
"""
import sys
import os
import time
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QInputDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from aiui import AutonomousDrivingUISetup
from aicamera import CameraVideoHandler
import torch
import cv2
from PyQt5.QtGui import QPixmap, QImage


class InferenceThread(QThread):
    """后台推理线程：接收最新一帧，按设置节流后进行YOLO推理并发回叠加后的QImage（兼容v5/v8）"""

    frame_ready = pyqtSignal(QImage)
    meta_ready = pyqtSignal(str)

    def __init__(self, model, infer_size: int, frame_skip: int, interval_ms: int, model_type: str = 'v5', conf: float = 0.5, iou: float = 0.45):
        super().__init__()
        self.model = model
        self.infer_size = infer_size
        self.frame_skip = max(1, int(frame_skip))
        self.interval_ms = max(0, int(interval_ms))
        self.model_type = 'v8' if (str(model_type).lower().find('v8') != -1 or str(model_type).lower().find('8') != -1) else 'v5'
        self.confidence = float(conf)
        self.iou = float(iou)
        self._running = True
        self._busy = False
        self._last_infer_ms = 0
        self._frame_index = 0
        self._latest_frame = None  # BGR np.ndarray

    def update_params(self, infer_size: int = None, frame_skip: int = None, interval_ms: int = None, conf: float = None, iou: float = None, model_type: str = None):
        if infer_size is not None:
            self.infer_size = int(infer_size)
        if frame_skip is not None:
            self.frame_skip = max(1, int(frame_skip))
        if interval_ms is not None:
            self.interval_ms = max(0, int(interval_ms))
        if conf is not None:
            self.confidence = float(conf)
        if iou is not None:
            self.iou = float(iou)
        if model_type is not None:
            self.model_type = 'v8' if (str(model_type).lower().find('v8') != -1 or str(model_type).lower().find('8') != -1) else 'v5'

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
                        if self.model_type == 'v8':
                            # YOLOv8 推理
                            results = self.model(frame, imgsz=int(self.infer_size), conf=float(self.confidence), iou=float(self.iou))
                            annotated_bgr = results[0].plot()
                            names = getattr(results[0], 'names', getattr(self.model, 'names', {}))
                            # 元信息
                            try:
                                boxes = getattr(results[0], 'boxes', None)
                                if boxes is not None and hasattr(boxes, 'xyxy') and boxes.xyxy is not None and boxes.xyxy.shape[0] > 0:
                                    ts_ms = int(time.time() * 1000)
                                    ts = time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000)) + f".{ts_ms % 1000:03d}"
                                    lines = []
                                    count = boxes.xyxy.shape[0]
                                    for i in range(count):
                                        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].detach().cpu().numpy().tolist()]
                                        conf = float(boxes.conf[i].detach().cpu().numpy().tolist())
                                        cls = int(boxes.cls[i].detach().cpu().numpy().tolist())
                                        label = names[cls] if isinstance(names, (list, dict)) else (names[cls] if isinstance(names, list) else f'class{cls}')
                                        lines.append(f"{i + 1}. 时间 {ts} | 类别 {label} | 位置 ({x1},{y1},{x2},{y2}) | 置信度 {conf:.2f}")
                                    self.meta_ready.emit("\n".join(lines))
                            except Exception:
                                pass
                        else:
                            # YOLOv5 推理
                            results = self.model(frame, size=int(self.infer_size))
                            annotated_bgr = results.render()[0]
                            names = getattr(results, 'names', getattr(self.model, 'names', {}))
                            # 元信息
                            try:
                                pred = results.xyxy[0]
                                if pred is not None and pred.shape[0] > 0:
                                    ts_ms = int(time.time() * 1000)
                                    ts = time.strftime("%H:%M:%S", time.localtime(ts_ms / 1000)) + f".{ts_ms % 1000:03d}"
                                    lines = []
                                    for i in range(int(pred.shape[0])):
                                        x1, y1, x2, y2 = [int(v) for v in pred[i, :4].detach().cpu().numpy().tolist()]
                                        conf = float(pred[i, 4])
                                        cls = int(pred[i, 5])
                                        label = names[cls] if isinstance(names, (list, dict)) else f'class{cls}'
                                        lines.append(f"{i + 1}. 时间 {ts} | 类别 {label} | 位置 ({x1},{y1},{x2},{y2}) | 置信度 {conf:.2f}")
                                    self.meta_ready.emit("\n".join(lines))
                            except Exception:
                                pass

                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = annotated_rgb.shape
                    # 构造独立QImage，防止指针悬挂
                    qt_img = QImage(annotated_rgb.copy().data, w, h, ch * w, QImage.Format_RGB888).copy()
                    self.frame_ready.emit(qt_img)
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

        # 加载YOLO模型（v5/v8）
        # 启动时先刷新一次本地权重列表，避免弹窗
        try:
            self.refresh_local_weights()
        except Exception:
            pass
        # 默认不自动加载模型，等待用户选择具体权重
        self.model = None
        self.model_type = 'v5'
        # 通过按钮选择的权重路径
        self.selected_weight_path = None

        # 初始化统计信息
        self.recent_detections = []  # 存储最近检测结果
        self.reset_stats()
        # 初始禁用开始检测，待模型成功加载后启用
        try:
            if hasattr(self, 'btn_play'):
                self.btn_play.setEnabled(False)
        except Exception:
            pass

    def reset_stats(self):
        """重置统计信息"""
        self.update_detection_info(0, 0, 0, 0, {})

    def _get_base_dir(self):
        """获取应用的基础目录：开发环境返回项目上级目录，打包环境返回可执行文件目录"""
        try:
            if getattr(sys, 'frozen', False):
                return os.path.dirname(sys.executable)
            return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        except Exception:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

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

        # 模型选择信号
        try:
            if hasattr(self, 'btn_select_weight'):
                self.btn_select_weight.clicked.connect(self.on_select_weight_clicked)
        except Exception:
            pass

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
        """加载YOLO模型（v5 或 v8，优先本地仓库），支持自定义权重"""
        # 推测本地仓库根：开发环境=项目上级，打包环境=exe目录
        base_dir = self._get_base_dir()
        # v5 仓库候选：优先 F:/.../YOLOv5 目录，其次 YOLOv5/yolov5-master，再次 yolo v5 同级
        v5_repo_candidates = [
            os.path.join(base_dir, 'YOLOv5'),
            os.path.join(base_dir, 'YOLOv5', 'yolov5-master'),
            os.path.join(base_dir, 'yolov5'),
            # 打包环境常见内置路径
            os.path.join(base_dir, '_internal', 'YOLOv5', 'yolov5-master'),
            os.path.join(base_dir, '_internal', 'YOLOv5'),
        ]
        # 仅当目录内含 hubconf.py 才视为有效 YOLOv5 本地仓库
        yolo_repo = next((p for p in v5_repo_candidates if os.path.isfile(os.path.join(p, 'hubconf.py'))), None)
        # 避免 torch.hub 缓存导致加载旧代码，强制本地优先时可关闭缓存
        try:
            torch.hub.set_dir(os.path.join(base_dir, '.torchhub'))
        except Exception:
            pass
        if yolo_repo is None:
            self.console.append("⚠️ 未找到本地YOLOv5仓库，将尝试联网加载（可能较慢）")

        # 使用按钮选择的权重路径
        weights_path = getattr(self, 'selected_weight_path', None)
        sp = (weights_path or '').strip()
        sp_lower = os.path.basename(sp).lower()
        path_lower = (sp or '').lower()
        # 先基于路径名做偏好判断；若不确定则自动尝试
        prefer_v8 = ('yolov8' in path_lower) or ('v8' in sp_lower)
        prefer_v5 = ('yolov5' in path_lower) or ('v5' in sp_lower)

        try:
            self.console.append(f"🧩 Torch 版本: {getattr(torch, '__version__', 'unknown')}")
            self.console.append(f"📦 权重路径: {weights_path if (weights_path and os.path.isfile(weights_path)) else '未选择'}")

            if not weights_path or not os.path.isfile(weights_path):
                self.console.append("ℹ️ 未选择权重，跳过模型加载")
                if hasattr(self, 'lbl_model_status'):
                    self.lbl_model_status.setText("❌ 模型未加载")
                if hasattr(self, 'btn_play'):
                    self.btn_play.setEnabled(False)
                return None

            model = None

            # 定义四种加载器（v1/v3 先按 v8 尝试，失败再按 v5）
            def _try_load_v8(p):
                from ultralytics import YOLO
                return YOLO(p)

            def _try_load_v5(p):
                if yolo_repo and os.path.isdir(yolo_repo):
                    try:
                        return torch.hub.load(yolo_repo, 'custom', path=p, source='local', trust_repo=True)
                    except TypeError:
                        return torch.hub.load(yolo_repo, 'custom', path=p, source='local')
                else:
                    try:
                        return torch.hub.load('ultralytics/yolov5', 'custom', path=p, trust_repo=True)
                    except TypeError:
                        return torch.hub.load('ultralytics/yolov5', 'custom', path=p)

            # 取消自动识别：默认强制按 YOLOv5 加载，如需 YOLOv8 可手动设置 forced_model_type='v8'
            forced = getattr(self, 'forced_model_type', None)
            if forced == 'v8':
                model = _try_load_v8(weights_path)
                self.model_type = 'v8'
                self.console.append(f"✅ 已按 YOLOv8 加载: {os.path.basename(weights_path)}")
            else:
                model = _try_load_v5(weights_path)
                self.model_type = 'v5'
                self.console.append(f"✅ 已按 YOLOv5 加载: {os.path.basename(weights_path)}")

            # 设置阈值（与UI保持一致），兼容不同版本API
            try:
                if self.model_type == 'v8':
                    # YOLOv8 使用推理参数控制，无需在模型上设置
                    pass
                else:
                    if hasattr(self, 'confidence_value') and hasattr(model, 'conf'):
                        model.conf = float(self.confidence_value)
                    if hasattr(self, 'iou_value') and hasattr(model, 'iou'):
                        model.iou = float(self.iou_value)
            except Exception:
                pass

            # 推理设备
            try:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if self.model_type == 'v5':
                    model.to(device)
                else:
                    try:
                        model.to(device)
                    except Exception:
                        pass
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
                    if self.model_type == 'v8':
                        _ = model(dummy, imgsz=int(self.infer_size), conf=float(getattr(self, 'confidence_value', 0.5)), iou=float(getattr(self, 'iou_value', 0.45)))
                    else:
                        _ = model(dummy, size=int(self.infer_size))
            except Exception:
                pass

            # 更新模型状态标签
            if hasattr(self, 'lbl_model_status'):
                label_type = 'v8' if getattr(self, 'model_type', 'v5') == 'v8' else 'v5'
                self.lbl_model_status.setText(f"📋 模型就绪 (YOLO{label_type.upper()})")

            return model
        except Exception as e:
            self.console.append(f"❌ 模型加载失败: {str(e)}")
            return None

    def detect_image(self, image_path):
        """使用YOLOv5/YOLOv8检测图像并显示标注结果（支持中文路径）"""
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
            # 设置阈值热更新（对于v5设置到模型，对于v8使用推理参数）
            try:
                if getattr(self, 'model_type', 'v5') == 'v5':
                    if hasattr(self, 'confidence_value') and hasattr(self.model, 'conf'):
                        self.model.conf = float(self.confidence_value)
                    if hasattr(self, 'iou_value') and hasattr(self.model, 'iou'):
                        self.model.iou = float(self.iou_value)
            except Exception:
                pass

            # 推理（节流不需要，因为单次图像检测）
            with torch.no_grad():
                if getattr(self, 'model_type', 'v5') == 'v8':
                    results = self.model(bgr, imgsz=int(self.infer_size), conf=float(getattr(self, 'confidence_value', 0.5)), iou=float(getattr(self, 'iou_value', 0.45)))
                    rendered = results[0].plot()
                else:
                    results = self.model(bgr, size=int(self.infer_size))
                    rendered = results.render()[0]
            inference_ms = int((time.time() - start_ts) * 1000)

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
        """从YOLO(v5/v8)结果中统计目标数与类别分布，返回(总数, 类别统计, 最近结果文本)"""
        try:
            if getattr(self, 'model_type', 'v5') == 'v8':
                res0 = results[0] if isinstance(results, (list, tuple)) else results
                boxes = getattr(res0, 'boxes', None)
                names = getattr(res0, 'names', getattr(self.model, 'names', {}))
                if boxes is not None and hasattr(boxes, 'cls') and boxes.cls is not None:
                    total = int(boxes.cls.shape[0])
                else:
                    total = 0
            else:
                # v5
                pred = results.xyxy[0]  # tensor: [N, 6]
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
                import numpy as _np
                if getattr(self, 'model_type', 'v5') == 'v8':
                    boxes = results[0].boxes
                    conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes, 'conf') else _np.array([])
                    cls = boxes.cls.detach().cpu().numpy().astype(int) if hasattr(boxes, 'cls') else _np.array([], dtype=int)
                    names = getattr(results[0], 'names', getattr(self.model, 'names', {}))
                else:
                    pred = results.xyxy[0]
                    conf = pred[:, 4].detach().cpu().numpy()
                    cls = pred[:, 5].detach().cpu().numpy().astype(int)
                    names = getattr(results, 'names', getattr(self.model, 'names', {}))

                unique_cls = _np.unique(cls)
                for ci in unique_cls:
                    en = names[int(ci)] if isinstance(names, (list, dict)) else f'class{int(ci)}'
                    cn = name_map.get(en, en)
                    class_counts[cn] = int((_np.array(cls) == ci).sum())

                # 取置信度最高的前3个
                if conf.size > 0:
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
        # 如勾选保存结果，询问图片序列保存文件夹
        if hasattr(self, 'cb_save') and self.cb_save.isChecked():
            if (self.camera_running or (self.video_thread and self.video_thread.isRunning())):
                self._prompt_save_image_dir()
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
        # 清理保存状态
        self._clear_save_state()

    # 已移除模型系列选择

    # 旧的下拉切换逻辑已废弃

    def refresh_local_weights(self):
        """兼容旧接口：不再刷新下拉；改为按钮选择时打开目录浏览。"""
        return

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
            # 根据权重文件名推断线程的模型类型
            model_type = getattr(self, 'model_type', 'v5')
            self.infer_thread = InferenceThread(self.model, self.infer_size, self.frame_skip, self.infer_interval_ms, model_type=model_type, conf=getattr(self, 'confidence_value', 0.5), iou=getattr(self, 'iou_value', 0.45))
            # 使用QueuedConnection确保跨线程UI信号安全
            try:
                from PyQt5.QtCore import Qt as _Qt
                self.infer_thread.frame_ready.connect(self._on_infer_frame_ready, type=_Qt.QueuedConnection)
                self.infer_thread.meta_ready.connect(self._on_infer_meta_ready, type=_Qt.QueuedConnection)
            except Exception:
                self.infer_thread.frame_ready.connect(self._on_infer_frame_ready)
                self.infer_thread.meta_ready.connect(self._on_infer_meta_ready)
            try:
                self.infer_thread.setObjectName("InferenceThread")
            except Exception:
                pass
            self.infer_thread.start()
        else:
            model_type = getattr(self, 'model_type', 'v5')
            self.infer_thread.update_params(self.infer_size, self.frame_skip, self.infer_interval_ms, conf=getattr(self, 'confidence_value', 0.5), iou=getattr(self, 'iou_value', 0.45), model_type=model_type)

    def _stop_infer_thread(self):
        if self.infer_thread is not None:
            try:
                self.infer_thread.stop()
                self.infer_thread.wait(2000)
            except Exception:
                pass
            self.infer_thread = None
    
    def _clear_save_state(self):
        try:
            if hasattr(self, 'save_image_dir'):
                del self.save_image_dir
            if hasattr(self, 'save_image_index'):
                del self.save_image_index
        except Exception:
            pass

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

            # 写入图片序列
            if getattr(self, 'save_image_dir', None):
                try:
                    w = qt_image.width()
                    h = qt_image.height()
                    ch = 3
                    tmp_img = qt_image.copy()
                    bits = tmp_img.bits()
                    bits.setsize(h * w * ch)
                    rgb = np.frombuffer(bits, dtype=np.uint8).reshape((h, w, ch))
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    name = f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{getattr(self, 'save_image_index', 0):06d}.jpg"
                    cv2.imwrite(os.path.join(self.save_image_dir, name), bgr)
                    self.save_image_index = getattr(self, 'save_image_index', 0) + 1
                except Exception:
                    pass
        except Exception:
            pass

    def _prompt_save_image_dir(self):
        try:
            default_dir = os.path.join(os.path.expanduser("~"), "Detections")
            if not os.path.isdir(default_dir):
                try:
                    os.makedirs(default_dir, exist_ok=True)
                except Exception:
                    default_dir = os.path.expanduser("~")

            folder = QFileDialog.getExistingDirectory(self, "选择保存图片序列的文件夹", default_dir)
            if folder:
                self.save_image_dir = folder
                self.save_image_index = 0
                self.console.append(f"💾 保存图片序列到: {folder}")
            else:
                self._clear_save_state()
        except Exception as e:
            self.console.append(f"❌ 选择保存路径失败: {str(e)}")
            self._clear_save_state()

    def on_select_weight_clicked(self):
        """点击‘选择权重’：先选 YOLO 类型(v3/v5/v8)，再在固定目录中选 .pt 文件。"""
        try:
            types = ["YOLOv3", "YOLOv5", "YOLOv8"]
            model_type, ok = QInputDialog.getItem(self, "选择模型类型", "模型类型:", types, 1, False)
            if not ok or not model_type:
                self.console.append("ℹ️ 已取消选择模型类型")
                return

            fixed_dirs = {
                'YOLOv3': r"F:\\desktop\\SEU\\卓工\\YOLOv3",
                'YOLOv5': r"F:\\desktop\\SEU\\卓工\\YOLOv5",
                'YOLOv8': r"F:\\desktop\\SEU\\卓工\\YOLOv8",
            }
            start_dir = fixed_dirs.get(model_type, r"F:\\desktop\\SEU\\卓工")
            if not os.path.isdir(start_dir):
                self.console.append(f"⚠️ 目录不存在: {start_dir}，回退到用户目录")
                start_dir = os.path.expanduser("~")

            file_path, _ = QFileDialog.getOpenFileName(self, f"选择 {model_type} 权重(.pt)", start_dir, "PyTorch 权重 (*.pt)")
            if not file_path or not os.path.isfile(file_path):
                self.console.append("ℹ️ 已取消选择权重")
                return

            self.selected_weight_path = file_path
            self.forced_model_type = 'v8' if model_type == 'YOLOv8' else 'v5'
            if hasattr(self, 'lbl_weight_path'):
                self.lbl_weight_path.setText(file_path)

            self.console.append("🔁 加载所选权重...")
            self.model = self.load_model()
            if self.model is None:
                self.console.append("❌ 模型加载失败")
                return

            if self.infer_thread is not None:
                model_type_flag = getattr(self, 'model_type', 'v5')
                self.infer_thread.update_params(self.infer_size, self.frame_skip, self.infer_interval_ms, conf=getattr(self, 'confidence_value', 0.5), iou=getattr(self, 'iou_value', 0.45), model_type=model_type_flag)
            self.console.append("✅ 权重加载完成")
            if hasattr(self, 'btn_play'):
                self.btn_play.setEnabled(True)
        except Exception as e:
            self.console.append(f"❌ 选择权重失败: {str(e)}")

    def _open_video_writer(self, w: int, h: int):
        return

    def _close_video_writer(self):
        return

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