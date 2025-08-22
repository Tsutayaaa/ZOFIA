"""
fish_head_cropper.py

模块功能：
    使用已训练的 YOLO 模型检测视频中鱼头位置，并进行居中裁剪与输出视频生成。
    支持启用自适应阻尼平滑跟踪、固定裁剪框尺寸等选项。

使用方式：
    1. 直接调用 crop_fish_head_video(cfg: dict) 接口；
    2. cfg 配置字典由外部 loader.py 读取并传入；
    3. 默认参数定义为模块级常量，可统一管理和更改。

作者：Gan
日期：2025.6.14
"""

import os
import cv2
import numpy as np
import traceback
from ultralytics import YOLO
from typing import Optional, List

# ==== 模块默认设置 ====

DEFAULT_CLASS_ID = 0                      # YOLO 模型中表示“鱼头”的类别编号
DEFAULT_OUTPUT_SIZE = 640                 # 输出图像的边长（单位：像素），最终输出视频将为该尺寸的正方形图像
DEFAULT_PADDING_FACTOR = 0.1              # 对检测到的边界框按比例增加边缘（例如 0.1 表示扩大10%）
DEFAULT_CONFIDENCE_THRESH = 0.2           # YOLO 模型的置信度阈值，低于该值的检测框将被忽略
DEFAULT_IOU_THRESH = 0.5                  # YOLO 模型中用于非极大值抑制的 IOU 阈值
DEFAULT_INTERPOLATION = cv2.INTER_LINEAR  # 图像缩放插值方式（OpenCV 插值方法），如 INTER_LINEAR 或 INTER_CUBIC
DEFAULT_FOURCC = "mp4v"                   # 输出视频编码器，常用如 "mp4v"（MP4）、"XVID"（AVI）
DEFAULT_DAMPING = True                    # 是否启用自适应阻尼平滑追踪中心点（推荐开启）
DEFAULT_DAMPING_STRONG = 0.15             # 缓慢移动时的阻尼系数，越小越平滑（强平滑）
DEFAULT_DAMPING_WEAK = 0.85               # 快速跳变时的阻尼系数，越大越快速响应（弱平滑）
DEFAULT_SLOW_PX = 5.0                     # 定义缓慢移动的距离阈值（单位：像素），小于该值将使用 strong damping
DEFAULT_FAST_PX = 60.0                    # 定义快速跳变的距离阈值（单位：像素），超过该值将使用 weak damping
DEFAULT_FIXED_CROP = True                 # 是否启用固定裁剪框尺寸（以第一帧为准），避免图像缩放比例频繁变化
# ===================

import cv2
import numpy as np
import os
from ultralytics import YOLO
from typing import List, Tuple, Optional
import traceback


class VideoFishHeadCropper:
    """
    一个用于检测视频中的鱼头、裁剪并生成新视频的类。
    包含自适应阻尼功能以平滑镜头移动，并可选择固定裁剪框尺寸。
    (Class structure and core logic aligned with uploaded cuthead2.py)
    """

    def __init__(self,
                 video_path: str,
                 output_video_path: str,
                 yolo_model_path: str,
                 fish_head_class_id: int = 0,  # Default from uploaded cuthead2.py __init__
                 output_img_size: int = 640,  # Default from uploaded cuthead2.py __init__
                 padding_factor: float = 0.1,  # Default from uploaded cuthead2.py __init__
                 confidence_thresh: float = 0.5,  # Default from uploaded cuthead2.py __init__
                 iou_thresh: float = 0.5,  # Default from uploaded cuthead2.py __init__
                 interpolation: int = cv2.INTER_LINEAR,  # Default from uploaded cuthead2.py __init__
                 fourcc_codec: str = 'mp4v',  # Default from uploaded cuthead2.py __init__
                 enable_adaptive_damping: bool = True,  # Default from uploaded cuthead2.py __init__
                 damping_factor_strong: float = 0.15,  # Default from uploaded cuthead2.py __init__
                 damping_factor_weak: float = 0.85,  # Default from uploaded cuthead2.py __init__
                 damping_threshold_slow_px: float = 5.0,  # Default from uploaded cuthead2.py __init__
                 damping_threshold_fast_px: float = 60.0,  # Default from uploaded cuthead2.py __init__
                 enable_fixed_crop_size: bool = True  # Default from uploaded cuthead2.py __init__
                 ):
        """
        初始化处理器。
        (Defaults in signature match uploaded cuthead2.py's VideoFishHeadCropper.__init__)
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
            print(
                f"警告: damping_threshold_fast_px ({damping_threshold_fast_px}) 应大于 damping_threshold_slow_px ({damping_threshold_slow_px}). 将调整 fast_px。")
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
        self.last_frame_width: Optional[int] = None
        self.last_frame_height: Optional[int] = None

    def _initialize_yolo(self) -> bool:
        # Initializes the YOLO model.
        if not os.path.exists(self.yolo_model_path):
            print(f"错误：YOLO 模型文件未找到: {self.yolo_model_path}")
            return False
        try:
            self.yolo_model = YOLO(self.yolo_model_path)
            print(f"YOLO 模型加载成功: {self.yolo_model_path}")
            return True
        except Exception as e:
            print(f"加载 YOLO 模型时出错: {e}")
            traceback.print_exc()
            return False

    def _initialize_video_capture(self) -> bool:
        # Initializes video capture.
        if not os.path.exists(self.video_path):
            print(f"错误：视频文件未找到: {self.video_path}")
            return False
        try:
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                print(f"错误：无法打开视频文件: {self.video_path}")
                return False

            self.last_frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.last_frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                print("警告：无法获取有效的视频帧率，将默认为 30 FPS。")
                self.video_fps = 30.0
            else:
                self.video_fps = fps
            print(f"输入视频信息: {self.last_frame_width}x{self.last_frame_height} @ {self.video_fps:.2f} FPS")
            return True
        except Exception as e:
            print(f"初始化视频捕获时出错: {e}")
            traceback.print_exc()
            return False

    def _initialize_video_writer(self) -> bool:
        # Initializes video writer.
        try:
            output_dir = os.path.dirname(self.output_video_path)
            if output_dir:  # Check if output_dir is not empty (e.g. for current directory)
                os.makedirs(output_dir, exist_ok=True)
            frame_size = (self.output_img_size, self.output_img_size)
            print(
                f"调试：尝试创建 VideoWriter - 路径: '{self.output_video_path}', 编码器: {self.fourcc_codec}, FPS: {self.video_fps}, 尺寸: {frame_size}")
            self.video_writer = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*self.fourcc_codec),
                                                self.video_fps, frame_size)
            if not self.video_writer.isOpened():
                print(f"错误：无法打开视频写入器。路径: {self.output_video_path}")
                return False
            print(f"视频写入器初始化成功，输出至: {self.output_video_path}")
            return True
        except Exception as e:
            print(f"初始化视频写入器时发生异常: {e}")
            traceback.print_exc()
            return False

    def _perform_detection_and_get_best_box(self, frame: np.ndarray) -> Optional[List[int]]:
        # Performs fish head detection and returns the best bounding box.
        best_detection = None
        if self.yolo_model is None: return None
        try:
            results = self.yolo_model.predict(frame, conf=self.confidence_thresh, iou=self.iou_thresh,
                                              classes=[self.fish_head_class_id], verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, class_id = box
                    if int(class_id) == self.fish_head_class_id:
                        w = x2 - x1
                        h = y2 - y1
                        bbox_ltwh = [int(x1), int(y1), int(max(0, w)), int(max(0, h))]
                        if best_detection is None or conf > best_detection[0]:
                            best_detection = (float(conf), bbox_ltwh)
            return best_detection[1] if best_detection else None
        except Exception as e:
            print(f"YOLO 检测过程中出错: {e}")
            return None

    def _crop_and_resize_centered(self, original_frame: np.ndarray, center_x: float, center_y: float,
                                  crop_square_side: int) -> Optional[np.ndarray]:
        # Crops and resizes the frame based on center point and square side.
        if original_frame is None or crop_square_side <= 0:
            return None
        frame_height, frame_width = original_frame.shape[:2]

        half_side = crop_square_side / 2.0
        x_start = max(0, int(round(center_x - half_side)))
        y_start = max(0, int(round(center_y - half_side)))
        x_end = min(frame_width, x_start + crop_square_side)
        y_end = min(frame_height, y_start + crop_square_side)

        if x_start < x_end and y_start < y_end:
            try:
                cropped_img = original_frame[y_start:y_end, x_start:x_end]
                if cropped_img.size == 0: return None
                resized_img = cv2.resize(cropped_img, (self.output_img_size, self.output_img_size),
                                         interpolation=self.interpolation)
                return resized_img
            except IndexError as e:
                print(
                    f"错误：裁剪图像时发生索引错误。Coords: y[{y_start}:{y_end}], x[{x_start}:{x_end}], Frame shape: {original_frame.shape}. Error: {e}")
                return None
            except cv2.error as e:
                print(f"  错误: 缩放图像时出错: {e}")
                print(f"  裁剪图像形状: {cropped_img.shape if 'cropped_img' in locals() else '未知'}")
                return None
        else:
            return None

    def _cleanup(self):
        # Releases resources.
        print("正在清理资源...")
        if self.video_capture is not None: self.video_capture.release()
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception as e:
                print(f"释放 VideoWriter 时出错: {e}")
        cv2.destroyAllWindows()
        print("资源清理完毕。")

    def process_video(self):
        # Main video processing loop.
        print("开始处理视频...")
        frame_count = 0
        processed_frame_count = 0

        if not self._initialize_yolo(): return
        if not self._initialize_video_capture(): return
        if not self._initialize_video_writer():
            self._cleanup()
            return

        try:
            while True:
                if self.video_capture is None: break
                ret, frame = self.video_capture.read()
                if not ret:
                    if frame_count == 0:
                        print("错误：无法读取视频的第一帧。")
                    else:
                        print(f"视频结束或在帧 {frame_count} 读取时出错。")
                    break

                best_bbox_ltwh = self._perform_detection_and_get_best_box(frame)
                processed_img = None
                current_crop_center_x = self.smoothed_crop_center_x
                current_crop_center_y = self.smoothed_crop_center_y
                crop_square_side_to_use = 0

                if best_bbox_ltwh:
                    detected_cx = best_bbox_ltwh[0] + best_bbox_ltwh[2] / 2.0
                    detected_cy = best_bbox_ltwh[1] + best_bbox_ltwh[3] / 2.0
                    detected_w = best_bbox_ltwh[2]
                    detected_h = best_bbox_ltwh[3]

                    if self.enable_fixed_crop_size:
                        if self.fixed_crop_square_side is None:
                            max_dim_for_first_detection = max(detected_w, detected_h)
                            self.fixed_crop_square_side = int(max_dim_for_first_detection * (1 + self.padding_factor))
                            print(f"信息：已固定裁剪框边长为: {self.fixed_crop_square_side} 像素 (基于首次检测)")
                        crop_square_side_to_use = self.fixed_crop_square_side if self.fixed_crop_square_side is not None else \
                            int(max(detected_w, detected_h) * (1 + self.padding_factor))
                    else:
                        crop_square_side_to_use = int(max(detected_w, detected_h) * (1 + self.padding_factor))

                    if self.enable_adaptive_damping:
                        if self.smoothed_crop_center_x is None or self.smoothed_crop_center_y is None:
                            self.smoothed_crop_center_x = detected_cx
                            self.smoothed_crop_center_y = detected_cy
                        else:
                            delta_x = detected_cx - self.smoothed_crop_center_x
                            delta_y = detected_cy - self.smoothed_crop_center_y
                            distance = np.sqrt(delta_x ** 2 + delta_y ** 2)

                            damping_factor_to_use = self.damping_factor_strong
                            if distance > self.damping_threshold_fast_px:
                                damping_factor_to_use = self.damping_factor_weak
                            elif distance > self.damping_threshold_slow_px:
                                if self.damping_threshold_fast_px > self.damping_threshold_slow_px:
                                    ratio = (distance - self.damping_threshold_slow_px) / \
                                            (self.damping_threshold_fast_px - self.damping_threshold_slow_px)
                                    damping_factor_to_use = self.damping_factor_strong + \
                                                            (
                                                                        self.damping_factor_weak - self.damping_factor_strong) * ratio
                                    damping_factor_to_use = np.clip(damping_factor_to_use,
                                                                    min(self.damping_factor_strong,
                                                                        self.damping_factor_weak),
                                                                    max(self.damping_factor_strong,
                                                                        self.damping_factor_weak))
                                else:
                                    damping_factor_to_use = self.damping_factor_weak
                            self.smoothed_crop_center_x += delta_x * damping_factor_to_use
                            self.smoothed_crop_center_y += delta_y * damping_factor_to_use
                        current_crop_center_x = self.smoothed_crop_center_x
                        current_crop_center_y = self.smoothed_crop_center_y
                    else:
                        current_crop_center_x = detected_cx
                        current_crop_center_y = detected_cy

                    if current_crop_center_x is not None and current_crop_center_y is not None and crop_square_side_to_use > 0:
                        processed_img = self._crop_and_resize_centered(frame, current_crop_center_x,
                                                                       current_crop_center_y, crop_square_side_to_use)
                else:
                    if self.enable_adaptive_damping and \
                            self.smoothed_crop_center_x is not None and \
                            self.smoothed_crop_center_y is not None and \
                            self.enable_fixed_crop_size and \
                            self.fixed_crop_square_side is not None:
                        processed_img = self._crop_and_resize_centered(frame,
                                                                       self.smoothed_crop_center_x,
                                                                       self.smoothed_crop_center_y,
                                                                       self.fixed_crop_square_side)

                if self.video_writer is None: break
                try:
                    if processed_img is not None:
                        self.video_writer.write(processed_img)
                        processed_frame_count += 1
                    else:
                        self.video_writer.write(self.black_frame)
                except Exception as e:
                    print(f"错误：在帧 {frame_count} 写入视频帧时出错: {e}")
                    break

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧 (生成有效裁剪帧: {processed_frame_count})...")

        finally:
            self._cleanup()
            print(f"\n处理完成。")
            print(f"总共处理了 {frame_count} 帧。")
            print(f"其中 {processed_frame_count} 帧生成了有效裁剪。")
            if os.path.exists(self.output_video_path) and processed_frame_count > 0:
                print(f"输出视频已保存至: {self.output_video_path}")
            elif processed_frame_count == 0 and frame_count > 0:
                print(f"未生成有效裁剪帧，输出视频可能只包含黑帧: {self.output_video_path}")
            elif not os.path.exists(self.output_video_path) and frame_count > 0:
                print(f"输出视频文件似乎未成功创建: {self.output_video_path}")

def run_cropping(video_path: str,
                 output_video_path: str,
                 yolo_model_path: str,
                 fish_head_class_id: int = 0,
                 output_img_size: int = 640,
                 padding_factor: float = 0.1,  # Default matches class for consistency in callable
                 confidence_thresh: float = 0.5,
                 iou_thresh: float = 0.5,
                 interpolation: int = cv2.INTER_LINEAR,
                 fourcc_codec: str = 'mp4v',
                 enable_adaptive_damping: bool = True,
                 damping_factor_strong: float = 0.15,
                 damping_factor_weak: float = 0.85,
                 damping_threshold_slow_px: float = 5.0,
                 damping_threshold_fast_px: float = 60.0,
                 enable_fixed_crop_size: bool = True):
    """
    Callable function to run the video fish head cropping process.
    (Parameters match VideoFishHeadCropper constructor defaults from uploaded cuthead2.py)
    """
    processor = VideoFishHeadCropper(
        video_path=video_path,
        output_video_path=output_video_path,
        yolo_model_path=yolo_model_path,
        fish_head_class_id=fish_head_class_id,
        output_img_size=output_img_size,
        padding_factor=padding_factor,
        confidence_thresh=confidence_thresh,
        iou_thresh=iou_thresh,
        interpolation=interpolation,
        fourcc_codec=fourcc_codec,
        enable_adaptive_damping=enable_adaptive_damping,
        damping_factor_strong=damping_factor_strong,
        damping_factor_weak=damping_factor_weak,
        damping_threshold_slow_px=damping_threshold_slow_px,
        damping_threshold_fast_px=damping_threshold_fast_px,
        enable_fixed_crop_size=enable_fixed_crop_size
    )
    processor.process_video()
    return output_video_path

def crop_fish_head_from_config(input_path: str, output_path: str, config: dict) -> str:
    """
    从配置字典执行鱼头裁剪流程，兼容 loader.py。

    参数:
        input_path (str): 输入视频路径
        output_path (str): 输出视频路径
        config (dict): 配置字典，包含 fish_head_crop 字段或顶层参数

    返回:
        str: 实际输出视频路径
    """
    crop_cfg = config.get("fish_head_crop", config)  # 支持嵌套或扁平配置
    return run_cropping(
        video_path=input_path,
        output_video_path=output_path,
        yolo_model_path=crop_cfg["yolo_model_path"],  # 必须提供
        fish_head_class_id=crop_cfg.get("fish_head_class_id", 0),
        output_img_size=crop_cfg.get("output_img_size", 640),
        padding_factor=crop_cfg.get("padding_factor", 0.1),
        confidence_thresh=crop_cfg.get("confidence_thresh", 0.5),
        iou_thresh=crop_cfg.get("iou_thresh", 0.5),
        interpolation=crop_cfg.get("interpolation", cv2.INTER_LINEAR),
        fourcc_codec=crop_cfg.get("fourcc_codec", "mp4v"),
        enable_adaptive_damping=crop_cfg.get("enable_adaptive_damping", True),
        damping_factor_strong=crop_cfg.get("damping_factor_strong", 0.15),
        damping_factor_weak=crop_cfg.get("damping_factor_weak", 0.85),
        damping_threshold_slow_px=crop_cfg.get("damping_threshold_slow_px", 5.0),
        damping_threshold_fast_px=crop_cfg.get("damping_threshold_fast_px", 60.0),
        enable_fixed_crop_size=crop_cfg.get("enable_fixed_crop_size", True),
    )