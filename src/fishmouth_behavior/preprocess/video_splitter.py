import cv2
import os
import math
from typing import List

def split_long_video(
    video_path: str,
    output_dir: str,
    split_duration_sec: int = 60,
    min_duration_for_split_sec: int = 120
) -> List[str]:
    """
    检查视频的时长。如果超过阈值，则将其切分成多个片段。

    Args:
        video_path (str): 输入视频文件的路径。
        output_dir (str): 保存视频片段的目录。
        split_duration_sec (int): 每个片段的时长（秒）。
        min_duration_for_split_sec (int): 视频需要被切分的最小总时长（秒）。

    Returns:
        List[str]: 一个包含所有视频片段路径的列表。如果视频未被切分，
                   则返回一个只包含原始视频路径的列表。
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  错误: 无法打开视频文件进行时长检查: {video_path}")
            return [video_path]

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        if duration < min_duration_for_split_sec:
            print(f"  视频时长 ({duration:.2f}s) 小于 {min_duration_for_split_sec}s，无需切分。")
            return [video_path]

        print(f"  视频时长 ({duration:.2f}s) 超过 {min_duration_for_split_sec}s，开始切分...")
        os.makedirs(output_dir, exist_ok=True)
        
        base_name, ext = os.path.splitext(os.path.basename(video_path))
        segment_paths = []
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        frames_per_segment = int(split_duration_sec * fps)
        num_segments = math.ceil(frame_count / frames_per_segment)

        for i in range(num_segments):
            segment_filename = f"{base_name}_part_{i+1}{ext}"
            segment_path = os.path.join(output_dir, segment_filename)
            segment_paths.append(segment_path)
            
            print(f"    正在创建分段 {i+1}/{num_segments}: {segment_filename}")
            
            writer = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))
            
            for _ in range(frames_per_segment):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            
            writer.release()
        
        cap.release()
        print(f"  视频切分完成，共生成 {num_segments} 个分段。")
        return segment_paths

    except Exception as e:
        print(f"  处理视频切分时发生错误: {e}")
        return [video_path] # 如果出错，则回退到处理原始视频
