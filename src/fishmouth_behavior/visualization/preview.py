from typing import Dict

import cv2
import numpy as np
# 导入我们的检测器类
from ..mouth_detect.detector import FishmouthDetector


def create_detection_preview_video(
    input_video_path: str,
    output_video_path: str,
    config: Dict
):
    """
    创建一个带有检测结果（嘴部掩码和眼部框）的预览视频。

    Args:
        input_video_path (str): 输入视频（通常是预处理后的）的路径。
        output_video_path (str): 输出预览视频的保存路径。
        config (Dict): 包含模型路径等配置的字典。
    """
    print(f"\n[Step 4: Creating Preview Video]")
    print(f"  Input: {input_video_path}")
    print(f"  Output: {output_video_path}")

    # 1. 初始化视频读取和写入器
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file for preview creation: {input_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 2. 初始化用于可视化的检测器
    try:
        detector = FishmouthDetector(
            model_path_mouth=config['detect_mouth_model'],
            model_path_eyes=config['detect_eye_model']
        )
    except Exception as e:
        print(f"Error initializing detector for preview: {e}")
        cap.release()
        writer.release()
        return
        
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. 获取可视化检测结果
        visual_results = detector.detect_for_visualization(frame)
        
        # 4. 在帧上绘制结果
        annotated_frame = frame.copy()
        
        # 绘制眼部边界框 (绿色)
        if visual_results.get('eye_boxes') is not None:
            for box in visual_results['eye_boxes'].xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制嘴部掩码 (蓝色半透明)
        if visual_results.get('mouth_masks') is not None and visual_results['mouth_masks'].data.shape[0] > 0:
            mask_tensor = visual_results['mouth_masks'].data[0]
            mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
            color_mask = np.zeros_like(annotated_frame)
            color_mask[mask_np == 1] = [255, 0, 0] # 蓝色 BGR
            annotated_frame = cv2.addWeighted(annotated_frame, 1.0, color_mask, 0.5, 0)
        
        # 5. 写入新帧
        writer.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  ... 已为预览渲染 {frame_count} 帧。")

    # 6. 释放资源
    cap.release()
    writer.release()
    print(f"预览视频已成功创建并保存至: {output_video_path}")

