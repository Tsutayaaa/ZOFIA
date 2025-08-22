import os
from typing import Optional, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


class FishmouthDetector:
    """
    使用两个独立的YOLO模型，分别检测鱼眼和鱼嘴。
    """
    def __init__(self, model_path_mouth: str, model_path_eyes: str, 
                 mouth_conf: float = 0.3, eye_conf: float = 0.01, imgsz: int = 640):
        """
        初始化检测器，加载嘴部和眼部两个模型，并分别设置置信度。
        """
        print(f"正在从以下路径初始化嘴部模型: {model_path_mouth}")
        self.model_mouth = YOLO(model_path_mouth)
        print(f"正在从以下路径初始化眼部模型: {model_path_eyes}")
        self.model_eyes = YOLO(model_path_eyes)
        
        self.mouth_conf = mouth_conf
        self.eye_conf = eye_conf
        self.imgsz = imgsz
        print(f"带有眼部追踪功能的 FishmouthDetector 初始化完成。嘴部Conf: {self.mouth_conf}, 眼部Conf: {self.eye_conf}")

    def detect_features(self, frame: np.ndarray, frame_idx: int) -> dict:
        """
        在单帧中同时检测嘴部和眼部特征，用于数据提取。
        返回一个包含计算后数值的字典。
        """
        features = {'mouth_area': 0, 'mouth_height': 0, 'eye_box_diagonal': 0.0}
        device = 0 if torch.cuda.is_available() else 'cpu'

        # 1. 检测嘴部 (使用 self.mouth_conf)
        mouth_detected = False
        try:
            mouth_preds = self.model_mouth.predict(frame, conf=self.mouth_conf, imgsz=self.imgsz, verbose=False, device=device)
            mouth_masks = mouth_preds[0].masks
            if mouth_masks and hasattr(mouth_masks, 'data') and mouth_masks.data.shape[0] > 0:
                best_mask_tensor = mouth_masks.data[0]
                mask_np = (best_mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                features['mouth_area'] = int(np.sum(mask_np))
                rows = np.any(mask_np, axis=1)
                if np.any(rows):
                    ymin, ymax = np.where(rows)[0][[0, -1]]
                    features['mouth_height'] = int(ymax - ymin)
                mouth_detected = True
        except Exception as e:
            print(f"警告：在帧 {frame_idx} 的嘴部检测过程中出错: {e}")
        
        # 2. 检测眼部区域 (使用 self.eye_conf)
        eye_detected = False
        try:
            eye_preds = self.model_eyes.predict(frame, conf=self.eye_conf, imgsz=self.imgsz, verbose=False, device=device)
            eye_boxes = eye_preds[0].boxes
            if eye_boxes and hasattr(eye_boxes, 'data') and eye_boxes.data.shape[0] > 0:
                best_box = eye_boxes.data[0].cpu().numpy()
                x1, y1, x2, y2, _, _ = best_box
                w, h = x2 - x1, y2 - y1
                features['eye_box_diagonal'] = np.sqrt(w**2 + h**2)
                eye_detected = True
        except Exception as e:
            print(f"警告：在帧 {frame_idx} 的眼部检测过程中出错: {e}")

        # --- 增强的诊断日志 ---
        if frame_idx > 0 and frame_idx % 200 == 0: # 每200帧报告一次状态
            if not eye_detected:
                print(f"诊断信息 (帧 {frame_idx}): 未能检测到眼部区域。")
            if not mouth_detected:
                print(f"诊断信息 (帧 {frame_idx}): 未能检测到嘴部区域。")

        return features

    def detect_for_visualization(self, frame: np.ndarray) -> Dict:
        """
        在单帧中进行检测，并返回用于可视化的原始对象。
        """
        device = 0 if torch.cuda.is_available() else 'cpu'
        results = {'mouth_masks': None, 'eye_boxes': None}
        try:
            mouth_preds = self.model_mouth.predict(frame, conf=self.mouth_conf, imgsz=self.imgsz, verbose=False, device=device)
            results['mouth_masks'] = mouth_preds[0].masks
            
            eye_preds = self.model_eyes.predict(frame, conf=self.eye_conf, imgsz=self.imgsz, verbose=False, device=device)
            results['eye_boxes'] = eye_preds[0].boxes
        except Exception as e:
            print(f"警告：可视化检测过程中出错: {e}")
        return results

def process_video_and_extract_data(
    video_source_path: str,
    yolo_model_path_eyes: str,
    yolo_model_path_mouth: str,
    **kwargs
    ) -> Optional[pd.DataFrame]:
    """
    处理视频，提取数据，并直接返回一个包含眼部尺寸的DataFrame。
    """
    if not all(os.path.exists(p) for p in [video_source_path, yolo_model_path_eyes, yolo_model_path_mouth]):
        print(f"错误: 一个或多个输入文件未找到。")
        return None

    try:
        feature_detector = FishmouthDetector(
            model_path_mouth=yolo_model_path_mouth,
            model_path_eyes=yolo_model_path_eyes,
            mouth_conf=kwargs.get('mouth_conf_threshold', 0.3), # 获取嘴部置信度
            eye_conf=kwargs.get('eye_conf_threshold', 0.01),     # 获取眼部置信度
            imgsz=kwargs.get('mouth_imgsz', 640)
        )
        cap = cv2.VideoCapture(video_source_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频: {video_source_path}")
            return None
    except Exception as e:
        print(f"错误：初始化模型或视频捕获时出错: {e}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    all_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        detected_features = feature_detector.detect_features(frame.copy(), frame_idx)
        
        frame_data = {
            'frame_number': frame_idx + 1,
            'timestamp_sec': timestamp,
            **detected_features
        }
        all_data.append(frame_data)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"已处理 {frame_idx} 帧...")

    cap.release()
    if not all_data:
        print("警告: 未从视频中提取到任何数据。")
        return None
        
    df = pd.DataFrame(all_data)
    print(f"检测器处理完成。返回包含 {len(df)} 行数据的DataFrame。")
    return df
