"""
video_standardizer.py

该模块用于视频的标准化处理，
包括剪切指定时长、转码为目标格式、帧率与分辨率调整。
"""

import os
import cv2
from typing import Optional, Tuple

# ===== 默认设置 =====
DEFAULT_DURATION_SEC = 60              # 可设为任意正整数（单位：秒）
DEFAULT_OUTPUT_FORMAT = "mp4"          # 支持如 "avi", "mov" 等 OpenCV 支持格式
DEFAULT_COMPRESS = False               # 暂不启用压缩，仅保留接口
DEFAULT_TARGET_FPS = None              # 可设为 25, 30, 60 等常见帧率
DEFAULT_TARGET_RESOLUTION = None       # 可设为 (640, 480)、(1280, 720) 等常用分辨率
# ==========================

def standardize_video(
    input_path: str,
    output_path: str,
    duration_sec: int = DEFAULT_DURATION_SEC,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    compress: bool = DEFAULT_COMPRESS,
    target_fps: Optional[float] = DEFAULT_TARGET_FPS,
    target_resolution: Optional[Tuple[int, int]] = DEFAULT_TARGET_RESOLUTION
) -> str:
    """
    精确保留时间码的视频裁剪与转码（逐帧写入）。

    参数:
        input_path (str): 原始视频路径
        output_path (str): 输出视频路径
        duration_sec (int): 保留前 N 秒（默认 60 秒）
        output_format (str): 输出格式（默认 mp4）
        compress (bool): 保留参数（暂未启用）
        target_fps (float, 可选): 指定输出帧率
        target_resolution (tuple, 可选): 目标分辨率 (宽, 高)

    返回:
        str: 实际写入的视频路径
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入视频: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {input_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    used_fps = target_fps or original_fps
    max_frames = int(duration_sec * used_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_res = target_resolution or (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') if output_format == 'mp4' else cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, used_fps, out_res)

    written = 0
    while cap.isOpened() and written < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"[警告] 视频帧不足，已提前结束 ({written} 帧)")
            break
        if target_resolution:
            frame = cv2.resize(frame, out_res)
        out.write(frame)
        written += 1

    cap.release()
    out.release()
    print(f"✅ 视频标准化完成，保存至: {output_path}")
    return output_path

def standardize_video_from_config(input_path: str, output_path: str, config: dict) -> str:
    """
    从配置文件中读取参数并执行视频标准化处理。

    参数:
        input_path (str): 输入路径
        output_path (str): 输出路径
        config (dict): 总配置字典（应包含 preprocess 字段）

    返回:
        str: 实际输出路径
    """
    preprocess_cfg = config.get("preprocess", {})

    return standardize_video(
        input_path=input_path,
        output_path=output_path,
        duration_sec=preprocess_cfg.get("duration_sec", DEFAULT_DURATION_SEC),
        output_format=preprocess_cfg.get("output_format", DEFAULT_OUTPUT_FORMAT),
        compress=preprocess_cfg.get("compress", DEFAULT_COMPRESS),
        target_fps=preprocess_cfg.get("target_fps", DEFAULT_TARGET_FPS),
        target_resolution=tuple(preprocess_cfg["target_resolution"]) if "target_resolution" in preprocess_cfg else DEFAULT_TARGET_RESOLUTION
    )