import os
from typing import List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class VideoFishHeadCropper:
    """
    一个用于检测视频中的鱼头、裁剪并生成新视频的类。
    现在作为视频预处理的一个步骤。
    (此代码源于 cuthead2_module.py)
    """
    def __init__(self,
                 video_path: str,
                 output_video_path: str,
                 yolo_model_path: str,
                 fish_head_class_id: int = 0,
                 output_img_size: int = 640,
                 padding_factor: float = 0.1,
                 confidence_thresh: float = 0.5,
                 iou_thresh: float = 0.5,
                 interpolation: int = cv2.INTER_LINEAR,
                 fourcc_codec: str = 'mp4v',
                 enable_adaptive_damping: bool = True,
                 damping_factor_strong: float = 0.15,
                 damping_factor_weak: float = 0.85,
                 damping_threshold_slow_px: float = 5.0,
                 damping_threshold_fast_px: float = 60.0,
                 enable_fixed_crop_size: bool = True
                ):
        """
        初始化处理器。
        """
        self.video_path = os.path.normpath(video_path)
        self.output_video_path = os.path.normpath(output_video_path)
        self.yolo_model_path = os.path.normpath(yolo_model_path)
        self.fish_head_class_id = fish_head_class_id
        self.output_img_size = output_img_size
        self.padding_factor = padding_factor
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.interpolation = interpolation
        self.fourcc_codec = fourcc_codec
        self.enable_adaptive_damping = enable_adaptive_damping
        self.damping_factor_strong = damping_factor_strong
        self.damping_factor_weak = damping_factor_weak
        if damping_threshold_fast_px <= damping_threshold_slow_px:
            self.damping_threshold_fast_px = damping_threshold_slow_px + 1.0
        else:
            self.damping_threshold_fast_px = damping_threshold_fast_px
        self.damping_threshold_slow_px = damping_threshold_slow_px
        self.enable_fixed_crop_size = enable_fixed_crop_size
        self.fixed_crop_square_side: Optional[int] = None
        self.yolo_model = None
        self.video_capture = None
        self.video_writer = None
        self.video_fps = 30.0
        self.black_frame = np.zeros((self.output_img_size, self.output_img_size, 3), dtype=np.uint8)
        self.smoothed_crop_center_x: Optional[float] = None
        self.smoothed_crop_center_y: Optional[float] = None

    def _initialize_yolo(self) -> bool:
        if not os.path.exists(self.yolo_model_path):
            print(f"错误：YOLO 模型文件未找到: {self.yolo_model_path}")
            return False
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"YOLO 模型加载成功: {self.yolo_model_path}")
            return True
        except Exception as e:
            print(f"加载 YOLO 模型时出错: {e}")
            return False

    def _initialize_video_capture(self) -> bool:
        if not os.path.exists(self.video_path):
            print(f"错误：视频文件未找到: {self.video_path}")
            return False
        try:
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                print(f"错误：无法打开视频文件: {self.video_path}")
                return False
            width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.video_fps = fps if fps > 0 else 30.0
            print(f"输入视频信息: {width}x{height} @ {self.video_fps:.2f} FPS")
            return True
        except Exception as e:
            print(f"初始化视频捕获时出错: {e}")
            return False

    def _initialize_video_writer(self) -> bool:
        try:
            output_dir = os.path.dirname(self.output_video_path)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            frame_size = (self.output_img_size, self.output_img_size)
            self.video_writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*self.fourcc_codec), self.video_fps, frame_size)
            if not self.video_writer.isOpened():
                 print(f"错误：无法打开视频写入器。路径: {self.output_video_path}")
                 return False
            print(f"视频写入器初始化成功，输出至: {self.output_video_path}")
            return True
        except Exception as e:
            print(f"初始化视频写入器时发生异常: {e}")
            return False

    def _perform_detection_and_get_best_box(self, frame: np.ndarray) -> Optional[List[int]]:
        best_detection = None
        if self.yolo_model is None: return None
        try:
            results = self.yolo_model.predict(frame, conf=self.confidence_thresh, iou=self.iou_thresh, classes=[self.fish_head_class_id], verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, class_id = box
                    if int(class_id) == self.fish_head_class_id:
                        w, h = x2 - x1, y2 - y1
                        bbox_ltwh = [int(x1), int(y1), int(max(0, w)), int(max(0, h))]
                        if best_detection is None or conf > best_detection[0]:
                            best_detection = (float(conf), bbox_ltwh)
            return best_detection[1] if best_detection else None
        except Exception as e:
            print(f"YOLO 检测过程中出错: {e}")
            return None

    def _crop_and_resize_centered(self, original_frame: np.ndarray, center_x: float, center_y: float, crop_square_side: int) -> Optional[np.ndarray]:
        if original_frame is None or crop_square_side <= 0: return None
        frame_h, frame_w = original_frame.shape[:2]
        half_side = crop_square_side / 2.0
        x_start, y_start = max(0, int(round(center_x - half_side))), max(0, int(round(center_y - half_side)))
        x_end, y_end = min(frame_w, x_start + crop_square_side), min(frame_h, y_start + crop_square_side)
        
        if x_start < x_end and y_start < y_end:
            try:
                cropped_img = original_frame[y_start:y_end, x_start:x_end]
                if cropped_img.size == 0: return None
                return cv2.resize(cropped_img, (self.output_img_size, self.output_img_size), interpolation=self.interpolation)
            except (IndexError, cv2.error) as e:
                 print(f"裁剪或缩放图像时出错: {e}")
                 return None
        return None

    def _cleanup(self):
        print("正在清理资源...")
        if self.video_capture: self.video_capture.release()
        if self.video_writer: self.video_writer.release()
        cv2.destroyAllWindows()
        print("资源清理完毕。")

    def process_video(self):
        print("开始处理视频...")
        if not all([self._initialize_yolo(), self._initialize_video_capture(), self._initialize_video_writer()]):
            self._cleanup()
            return

        frame_count = processed_frame_count = 0
        try:
            while True:
                ret, frame = self.video_capture.read()
                if not ret: break

                best_bbox = self._perform_detection_and_get_best_box(frame)
                processed_img = None
                
                if best_bbox:
                    detected_cx, detected_cy = best_bbox[0] + best_bbox[2] / 2.0, best_bbox[1] + best_bbox[3] / 2.0
                    if self.enable_fixed_crop_size and self.fixed_crop_square_side is None:
                        self.fixed_crop_square_side = int(max(best_bbox[2], best_bbox[3]) * (1 + self.padding_factor))
                        print(f"信息：已固定裁剪框边长为: {self.fixed_crop_square_side} 像素")
                    
                    crop_side = self.fixed_crop_square_side or int(max(best_bbox[2], best_bbox[3]) * (1 + self.padding_factor))

                    if self.enable_adaptive_damping:
                        if self.smoothed_crop_center_x is None:
                            self.smoothed_crop_center_x, self.smoothed_crop_center_y = detected_cx, detected_cy
                        else:
                            delta_x, delta_y = detected_cx - self.smoothed_crop_center_x, detected_cy - self.smoothed_crop_center_y
                            dist = np.sqrt(delta_x**2 + delta_y**2)
                            ratio = np.clip((dist - self.damping_threshold_slow_px) / (self.damping_threshold_fast_px - self.damping_threshold_slow_px + 1e-6), 0, 1)
                            damping = self.damping_factor_strong + (self.damping_factor_weak - self.damping_factor_strong) * ratio
                            self.smoothed_crop_center_x += delta_x * damping
                            self.smoothed_crop_center_y += delta_y * damping
                    else:
                        self.smoothed_crop_center_x, self.smoothed_crop_center_y = detected_cx, detected_cy
                
                if self.smoothed_crop_center_x is not None:
                    crop_side_to_use = self.fixed_crop_square_side or 0
                    processed_img = self._crop_and_resize_centered(frame, self.smoothed_crop_center_x, self.smoothed_crop_center_y, crop_side_to_use)

                if self.video_writer: self.video_writer.write(processed_img if processed_img is not None else self.black_frame)
                if processed_img is not None: processed_frame_count += 1
                
                frame_count += 1
                if frame_count % 100 == 0: print(f"已处理 {frame_count} 帧...")
        finally:
            self._cleanup()
            print(f"\n处理完成。总共处理了 {frame_count} 帧，生成了 {processed_frame_count} 有效裁剪帧。")
            if os.path.exists(self.output_video_path): print(f"输出视频已保存至: {self.output_video_path}")

def run_clipping(video_path: str, output_video_path: str, yolo_model_path: str, **kwargs):
    """
    可调用的函数，用于运行视频裁剪预处理。
    """
    processor = VideoFishHeadCropper(
        video_path=video_path,
        output_video_path=output_video_path,
        yolo_model_path=yolo_model_path,
        **kwargs
    )
    processor.process_video()
    return output_video_path
