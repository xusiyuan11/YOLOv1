import os
import json
import xml.etree.ElementTree as ET
import pickle
from typing import List, Tuple, Optional, Dict

import cv2  # pylint: disable=no-member
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _pad_and_resize_image(image: np.ndarray, input_size: int) -> np.ndarray:
    """Pad image to square and resize to input_size."""
    h, w, _ = image.shape
    max_edge = max(h, w)
    top = bottom = left = right = 0
    if h < max_edge:
        pad = max_edge - h
        top = pad // 2
        bottom = pad - top
    else:
        pad = max_edge - w
        left = pad // 2
        right = pad - left

    img_pad = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_resized = cv2.resize(img_pad, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    return img_resized, top, left, max_edge


def _transform_coords(coords: List[List], top: int, left: int, max_edge: int, input_size: int) -> List[List]:
    """Transform coordinates after padding and resizing."""
    new_coords = []
    scale_factor = input_size / max_edge
    for c in coords:
        xmin = (c[0] + left) * scale_factor
        ymin = (c[1] + top) * scale_factor
        xmax = (c[2] + left) * scale_factor
        ymax = (c[3] + top) * scale_factor
        new_coords.append([xmin / input_size, ymin / input_size, xmax / input_size, ymax / input_size, c[4]])
    return new_coords


def _encode_ground_truth(coords: List[List], input_size: int, grid_size: int, class_num: int, class_smooth_value: float):
    """Encode YOLOv1 ground truth - 修正版本."""
    feature_size = grid_size  # grid_size就是网格数量，例如7
    cell_size = input_size / grid_size  # 每个网格的像素大小，例如448/7=64
    # 格式：2个边界框(每个5维) + 类别概率 = 2*5 + class_num
    gt = np.zeros([feature_size, feature_size, 2*5 + class_num], dtype=float)
    mask_pos = np.zeros((feature_size, feature_size, 1), dtype=bool)
    mask_neg = np.ones((feature_size, feature_size, 1), dtype=bool)

    for coord in coords:
        xmin, ymin, xmax, ymax, class_id = coord
        w = xmax - xmin
        h = ymax - ymin
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        
        # 修正：直接用归一化坐标计算网格索引
        idx_col = int(cx * grid_size)  # 列索引 (0-6)
        idx_row = int(cy * grid_size)  # 行索引 (0-6)
        idx_row = min(max(idx_row, 0), feature_size - 1)
        idx_col = min(max(idx_col, 0), feature_size - 1)

        class_list = np.full(shape=class_num, fill_value=class_smooth_value / max(class_num - 1, 1), dtype=float)
        class_list[class_id] = 1.0 - class_smooth_value

        # 修正：YOLO格式编码
        tx = (cx * grid_size) - idx_col  # 网格内x偏移 (0-1)
        ty = (cy * grid_size) - idx_row  # 网格内y偏移 (0-1)
        tw = w  # 相对于整个图像的宽度 (已归一化)
        th = h  # 相对于整个图像的高度 (已归一化)
        
        # 修正：只设置第一个边界框，让第二个保持初始状态
        # 这样损失函数可以学习到两个不同的边界框预测
        gt[idx_row, idx_col, 0:5] = [tx, ty, tw, th, 1.0]   # 第一个边界框
        # 第二个边界框保持为0，让网络自己学习
        gt[idx_row, idx_col, 10:10+class_num] = class_list  # 类别概率
        mask_pos[idx_row, idx_col, 0] = True
        mask_neg[idx_row, idx_col, 0] = False

    return gt, torch.BoolTensor(mask_pos), torch.BoolTensor(mask_neg)


class VOC_Detection_Set(Dataset):

    def __init__(self,
                 voc2012_jpeg_dir: str,
                 voc2012_anno_dir: str,
                 class_file: str,
                 class_smooth_value: float = 0.01,
                 input_size: int = 448,
                 grid_size: int = 7,  # 修正：7x7网格，不是64
                 is_train: bool = True):
        super().__init__()

        self.class_smooth_value = class_smooth_value
        self.input_size = input_size
        self.grid_size = grid_size

        # ---------- 读类别 ----------
        self.class_num = 0
        self.class_dict: Dict[str, int] = {}
        with open(class_file, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                self.class_dict[name] = self.class_num
                self.class_num += 1

        # ---------- 聚合所有样本 ----------
        self.samples: List[Tuple[str, str]] = []  # (img_path, xml_path)

        # VOC2007
        # self._add_dir_pair(voc2007_jpeg_dir, voc2007_anno_dir)
        # VOC2012
        self._add_dir_pair(voc2012_jpeg_dir, voc2012_anno_dir)

        if len(self.samples) == 0:
            raise RuntimeError('未找到任何图片/标注，请检查四个目录是否为空或路径错误')

        # ---------- 预处理 ----------
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.408, 0.448, 0.471),
                        std=(0.242, 0.239, 0.234))
        ])

    # ----------------------------- 内部工具 -----------------------------
    def _add_dir_pair(self, jpeg_dir: str, anno_dir: str):
        if not os.path.isdir(jpeg_dir) or not os.path.isdir(anno_dir):
            return
        for img_name in sorted(os.listdir(jpeg_dir)):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            base = os.path.splitext(img_name)[0]
            xml_path = os.path.join(anno_dir, base + '.xml')
            if os.path.isfile(xml_path):
                img_path = os.path.join(jpeg_dir, img_name)
                self.samples.append((img_path, xml_path))

    # ----------------------------- Dataset 接口 -----------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]

        img = self._read_image(img_path)
        coords = self._read_coords(xml_path)
        img, coords = self._process_image_and_coords(img, coords)

        gt, mask_pos, mask_neg = self._encode_ground_truth(
            coords,
            self.input_size,
            self.grid_size,
            self.class_num,
            self.class_smooth_value
        )
        return img, [gt, mask_pos, mask_neg]

    # ----------------------------- 私有方法 -----------------------------
    def _read_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_coords(self, xml_path: str) -> List[List]:
        if not os.path.exists(xml_path):
            return []
        tree = ET.parse(xml_path)
        root = tree.getroot()

        objs = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_dict:
                continue
            cls_id = self.class_dict[name]
            bnd = obj.find('bndbox')
            xmin = float(bnd.find('xmin').text)
            ymin = float(bnd.find('ymin').text)
            xmax = float(bnd.find('xmax').text)
            ymax = float(bnd.find('ymax').text)
            objs.append([xmin, ymin, xmax, ymax, cls_id])

        # 按面积从小到大排序（与之前保持一致）
        objs.sort(key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
        return objs

    def _process_image_and_coords(self,
                                  image: np.ndarray,
                                  coords: List[List]) -> Tuple[torch.Tensor, List[List]]:
        """
        与原版完全一致：先 pad+resize 图片，再同步变换坐标
        """
        # 这里直接调用你之前实现的 utils
        img_resized, top, left, max_edge = self._pad_and_resize_image(image, self.input_size)
        img_tensor = self.transform(img_resized)
        new_coords = self._transform_coords(coords, top, left, max_edge, self.input_size)
        return img_tensor, new_coords

    # ----------------------------- 静态工具 -----------------------------
    @staticmethod
    def _pad_and_resize_image(image: np.ndarray, target_size: int):
        """保持长宽比缩放并填充"""
        h, w, _ = image.shape
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top, left = pad_h // 2, pad_w // 2
        bottom = pad_h - top
        right = pad_w - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(128, 128, 128))
        return padded, top, left, max(h, w)

    @staticmethod
    def _transform_coords(coords: List[List],
                          top: int, left: int,
                          max_edge: float,
                          target_size: int) -> List[List]:
        """把原图坐标映射到 pad+resize 后的图上"""
        scale = target_size / max_edge
        new_coords = []
        for x1, y1, x2, y2, cls_id in coords:
            nx1 = (x1 * scale) + left
            ny1 = (y1 * scale) + top
            nx2 = (x2 * scale) + left
            ny2 = (y2 * scale) + top
            new_coords.append([nx1, ny1, nx2, ny2, cls_id])
        return new_coords

    @staticmethod
    def _encode_ground_truth(coords: List[List],
                             input_size: int,
                             grid_size: int,
                             class_num: int,
                             smooth: float):
        """
        把坐标编码成 YOLO 格式 (S,S,5B+C)
        这里给出简化示例，请替换为你之前的实现
        """
        S = grid_size
        gt = torch.zeros((S, S, 2*5 + class_num))  # 2个边界框 + 类别
        mask_pos = torch.zeros((S, S), dtype=torch.bool)
        mask_neg = torch.ones((S, S), dtype=torch.bool)

        cell_size = input_size / S
        for x1, y1, x2, y2, cls_id in coords:
            # 归一化坐标到[0,1]
            cx = (x1 + x2) / 2.0 / input_size
            cy = (y1 + y2) / 2.0 / input_size
            w  = (x2 - x1) / input_size
            h  = (y2 - y1) / input_size

            # 计算目标中心点落在哪个网格
            grid_x = int(cx * S)
            grid_y = int(cy * S)
            if 0 <= grid_x < S and 0 <= grid_y < S:
                mask_pos[grid_y, grid_x] = True
                mask_neg[grid_y, grid_x] = False

                # YOLO格式编码
                tx = (cx * S) - grid_x  # 网格内偏移[0,1)
                ty = (cy * S) - grid_y  # 网格内偏移[0,1)
                
                # 两个边界框都设置为相同目标
                gt[grid_y, grid_x, 0:5] = torch.tensor([tx, ty, w, h, 1.0])    # 第一个边界框
                gt[grid_y, grid_x, 5:10] = torch.tensor([tx, ty, w, h, 1.0])   # 第二个边界框
                one_hot = torch.zeros(class_num)
                one_hot[cls_id] = 1.0 - smooth
                one_hot += smooth / class_num
                gt[grid_y, grid_x, 10:10+class_num] = one_hot  # 类别概率放在10位置开始

        return gt, mask_pos, mask_neg

class COCO_Detection_Set(Dataset):
    def __init__(self, imgs_path: str, coco_json: str, input_size: int = 448, grid_size: int = 64,
                 class_smooth_value: float = 0.01):
        self.imgs_path = imgs_path
        self.coco_json = coco_json
        self.input_size = input_size
        self.grid_size = grid_size
        self.class_smooth_value = class_smooth_value

        with open(coco_json, 'r', encoding='utf-8') as f:
            jd = json.load(f)

        cats = jd.get('categories', [])
        self.catid2cont = {c['id']: i for i, c in enumerate(cats)}
        self.class_num = len(self.catid2cont)

        images = {img['id']: img for img in jd.get('images', [])}
        anns = jd.get('annotations', [])
        self.annotations_map: Dict[str, List] = {}
        for ann in anns:
            img_info = images.get(ann['image_id'])
            if img_info is None: continue
            fname = img_info['file_name']
            x, y, w, h = ann['bbox']
            cid = self.catid2cont.get(ann['category_id'], 0)
            self.annotations_map.setdefault(fname, []).append([x, y, x + w, y + h, cid])

        self.imgs_name = sorted([f for f in os.listdir(self.imgs_path) if f in self.annotations_map])

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))
        ])

    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, idx):
        img = self._read_image(idx)
        coords = self._read_coords(idx)
        img, coords = self._process_image_and_coords(img, coords)
        gt, mask_pos, mask_neg = _encode_ground_truth(coords, self.input_size, self.grid_size, self.class_num, self.class_smooth_value)
        return img, [gt, mask_pos, mask_neg]

    def _read_image(self, idx: int) -> np.ndarray:
        path = os.path.join(self.imgs_path, self.imgs_name[idx])
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_coords(self, idx: int) -> List[List]:
        fname = self.imgs_name[idx]
        return self.annotations_map.get(fname, [])

    def _process_image_and_coords(self, image: np.ndarray, coords: List[List]):
        img_resized, top, left, max_edge = _pad_and_resize_image(image, self.input_size)
        img_tensor = self.transform(img_resized)
        new_coords = _transform_coords(coords, top, left, max_edge, self.input_size)
        return img_tensor, new_coords


# 示例：使用 CIFAR10 数据集（合成全图 bbox）

class COCO_Segmentation_Classification_Set(Dataset):
    """
    COCO分割目标分类数据集
    将COCO数据集中的每个分割目标作为单独的分类样本
    用于预训练卷积网络
    """
    def __init__(self, 
                 imgs_path: str, 
                 coco_json: str, 
                 input_size: int = 448,
                 min_area: int = 1000,
                 max_objects_per_image: int = 10):
        """
        初始化COCO分割分类数据集
        Args:
            imgs_path: COCO图像路径
            coco_json: COCO标注JSON文件路径
            input_size: 输入图像尺寸
            min_area: 最小目标面积阈值
            max_objects_per_image: 每张图像最大目标数量
        """
        self.imgs_path = imgs_path
        self.input_size = input_size
        self.min_area = min_area
        self.max_objects_per_image = max_objects_per_image
        
        # 加载COCO标注
        with open(coco_json, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        # 构建类别映射
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        self.category_id_to_class_id = {cat['id']: i for i, cat in enumerate(self.coco_data['categories'])}
        self.class_num = len(self.categories)
        
        # 构建图像映射
        self.images = {img['id']: img for img in self.coco_data['images']}
        
        # 提取所有有效的分割目标
        self.segmentation_objects = self._extract_segmentation_objects()
        
        # 数据增强
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))
        ])
        
        print(f"COCO分割分类数据集初始化完成:")
        print(f"  - 类别数: {self.class_num}")
        print(f"  - 分割目标数: {len(self.segmentation_objects)}")
    
    def _extract_segmentation_objects(self):
        """提取所有有效的分割目标"""
        objects = []
        skipped_count = 0
        
        for ann in self.coco_data['annotations']:
            # 检查是否有分割信息 - 简化检查，允许没有分割信息
            # if 'segmentation' not in ann or not ann['segmentation']:
            #     skipped_count += 1
            #     continue
            
            # 检查面积
            if ann['area'] < self.min_area:
                skipped_count += 1
                continue
            
            # 检查bbox有效性
            if 'bbox' not in ann or len(ann['bbox']) != 4:
                skipped_count += 1
                continue
                
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                skipped_count += 1
                continue
            
            # 检查图像是否存在
            image_id = ann['image_id']
            if image_id not in self.images:
                skipped_count += 1
                continue
            
            image_info = self.images[image_id]
            image_path = os.path.join(self.imgs_path, image_info['file_name'])
            if not os.path.exists(image_path):
                continue
            
            # 获取类别ID
            if ann['category_id'] not in self.category_id_to_class_id:
                continue
            
            class_id = self.category_id_to_class_id[ann['category_id']]
            
            objects.append({
                'image_path': image_path,
                'image_id': image_id,
                'annotation': ann,
                'class_id': class_id,
                'bbox': ann['bbox'],  # [x, y, width, height]
                'segmentation': ann.get('segmentation', [])  # 可能为空
            })
        
        print(f"数据加载统计: 有效={len(objects)}, 跳过={skipped_count}")
        return objects
    
    def _resize_and_pad(self, image):
        """调整大小并填充到正方形"""
        h, w = image.shape[:2]
        max_edge = max(h, w)
        
        # 计算填充
        top = (max_edge - h) // 2
        bottom = max_edge - h - top
        left = (max_edge - w) // 2
        right = max_edge - w - left
        
        # 填充图像
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        # 调整到目标尺寸
        resized_image = cv2.resize(padded_image, (self.input_size, self.input_size))
        
        return resized_image
    
    def __len__(self):
        return len(self.segmentation_objects)
    
    def __getitem__(self, idx):
        obj_info = self.segmentation_objects[idx]
        
        try:
            # 读取图像
            image = cv2.imread(obj_info['image_path'])
            if image is None:
                raise FileNotFoundError(f"无法读取图像: {obj_info['image_path']}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 简化目标提取：直接使用bbox而不处理复杂的分割
            bbox = obj_info['bbox']
            x, y, w, h = [int(v) for v in bbox]
            
            # 确保bbox在图像范围内
            img_h, img_w = image.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            # 如果bbox太小，扩展一些边距
            if w < 32 or h < 32:
                margin = 16
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(img_w - x, w + 2 * margin)
                h = min(img_h - y, h + 2 * margin)
            
            # 裁剪目标区域
            object_image = image[y:y+h, x:x+w]
            
            # 检查裁剪结果
            if object_image.size == 0:
                raise ValueError("裁剪区域为空")
            
            # 调整尺寸
            processed_image = self._resize_and_pad(object_image)
            
            # 转换为张量
            image_tensor = self.transform(processed_image)
            
            # 类别标签
            class_id = obj_info['class_id']
            
            return image_tensor, class_id
            
        except Exception as e:
            print(f"Error processing object {idx}: {str(e)}")
            # 尝试使用原始类别ID，如果没有则使用1（避免0类别）
            fallback_class_id = obj_info.get('class_id', 1)
            # 返回一个默认样本
            default_image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            default_tensor = self.transform(default_image)
            return default_tensor, fallback_class_id

