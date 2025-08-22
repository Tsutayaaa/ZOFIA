import os
import sys
import cv2
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import traceback
from datetime import datetime
from tqdm import tqdm
import numpy as np

# 导入项目模块
try:
    from fishmouth_behavior.preprocess import video_clip
    from fishmouth_behavior.mouth_detect import detector
    from fishmouth_behavior.analysis.analyzer import FishMouthAnalyzer
    from fishmouth_behavior.visualization import plotter
    from fishmouth_behavior.visualization import preview
except ImportError as e:
    print(f"FATAL: A core module could not be imported: {e}", flush=True)
    print("Please ensure the 'src' directory is in your PYTHONPATH or structured correctly.", flush=True)
    sys.exit(1)

# 自定义JSON转换器，用于处理Numpy数据类型
class NumpyEncoder(json.JSONEncoder):
    """ 
    一个特殊的JSON编码器，可以将Numpy类型转换为Python原生类型，
    从而解决 'Object of type int64 is not JSON serializable' 错误。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class HierarchicalAnalysisPipeline:
    """
    完整的鱼嘴行为分层时间分析管道。
    遵循“标准化 -> 预处理 -> 分层切片 -> 独立分析 -> 汇总报告”的流程。
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.batch_summary = []

    def _standardize_video_duration(self, input_path: str, output_path: str, target_duration_sec: int = 300) -> bool:
        """
        将输入视频标准化为固定的目标时长。
        """
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"  错误：无法打开视频 {input_path}", flush=True)
                return False
                
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            target_frames = int(target_duration_sec * fps)
            
            with tqdm(total=target_frames, desc=f"  标准化帧进度", unit="frame", leave=False) as pbar:
                frames_written = 0
                while frames_written < target_frames:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            print("  错误：无法从视频中读取帧，即使是循环播放。", flush=True)
                            break
                            
                    writer.write(frame)
                    frames_written += 1
                    pbar.update(1)
                
            cap.release()
            writer.release()
            
            print(f"  ✓ 已将视频标准化为 {target_duration_sec} 秒。", flush=True)
            return True
            
        except Exception as e:
            print(f"  错误：视频标准化过程中出错: {e}", flush=True)
            traceback.print_exc()
            return False

    def _create_time_segments(self, video_path: str, output_dir: str) -> Dict[str, str]:
        """
        将标准化的视频分割成多个预定义的时间段。
        """
        segments = {}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"  错误：无法打开视频 {video_path} 进行切片。", flush=True)
            return segments
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        time_ranges = [
            (0, 300, "full_5min"),
            (0, 60, "min_1"), (60, 120, "min_2"), (120, 180, "min_3"), (180, 240, "min_4"), (240, 300, "min_5"),
            (0, 10, "min1_seg1_10s"), (10, 20, "min1_seg2_10s"), (20, 30, "min1_seg3_10s"),
            (30, 40, "min1_seg4_10s"), (40, 50, "min1_seg5_10s"), (50, 60, "min1_seg6_10s"),
        ]
        
        with tqdm(total=len(time_ranges), desc="  创建时间片段", unit="segment", leave=False) as pbar:
            for start_sec, end_sec, segment_name in time_ranges:
                pbar.set_description(f"  创建片段 [{segment_name}]")
                output_path = os.path.join(output_dir, f"{segment_name}.mp4")
                segments[segment_name] = output_path
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * fps))
                
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                end_frame_pos = int(end_sec * fps)
                
                while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame_pos:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                    
                writer.release()
                pbar.update(1)
            
        cap.release()
        print(f"  ✓ 已成功创建 {len(time_ranges)} 个时间片段。", flush=True)
        return segments

    def _analyze_segment(self, segment_name: str, segment_path: str, output_dirs: Dict) -> Dict:
        """对单个时间片段进行完整的分析。"""
        print(f"\n    --- 开始分析时间段: {segment_name} ---", flush=True)
        
        print(f"      [1/3] 开始特征检测...", flush=True)
        detection_df = detector.process_video_and_extract_data(
            video_source_path=segment_path,
            yolo_model_path_eyes=self.config['eye_model'],
            yolo_model_path_mouth=self.config['mouth_model'],
            mouth_conf_threshold=self.config.get('mouth_conf_threshold', 0.3),
            eye_conf_threshold=self.config.get('eye_conf_threshold', 0.3)
        )
        print(f"      [1/3] 特征检测完成。", flush=True)
        
        if detection_df is None or detection_df.empty:
            print(f"      [警告] {segment_name} 的特征检测失败，跳过分析。", flush=True)
            return {'status': 'Detection Failed'}
            
        detection_csv_path = Path(output_dirs['detection_data']) / f'{segment_name}_detection.csv'
        detection_df.to_csv(detection_csv_path, index=False)
        
        print(f"      [2/3] 开始行为分析...", flush=True)
        analyzer = FishMouthAnalyzer(detection_df)
        events = analyzer.analyze(**self.config.get('analysis_params', {}))
        print(f"      [2/3] 行为分析完成。", flush=True)
        
        events_json_path = Path(output_dirs['analysis_results']) / f'{segment_name}_events.json'
        with open(events_json_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
        print(f"      [3/3] 开始生成可视化图表...", flush=True)
        plot_path = Path(output_dirs['visualizations']) / f'{segment_name}_analysis_plot.png'
        plotter.plot_analysis_results(
            df_full=analyzer.df_processed,
            events=events,
            time_col='timestamp_sec',
            area_col='normalized_area',
            output_image_path=str(plot_path)
        )
        print(f"      [3/3] 可视化图表生成完毕。", flush=True)
        
        print(f"      ✓ {segment_name} 分析完成，检测到 {len(events)} 个事件。", flush=True)
        return {
            'status': 'Success',
            'detection_data_path': str(detection_csv_path),
            'events_data_path': str(events_json_path),
            'plot_path': str(plot_path),
            'events': events,
            'event_count': len(events)
        }

    def process_single_video(self, video_path: Path) -> None:
        """处理单个视频的完整分层分析流程。"""
        video_name = video_path.stem
        print(f"\n{'='*70}", flush=True)
        print(f"🎬 开始处理视频: {video_name} [时间: {datetime.now()}]", flush=True)
        print(f"{'='*70}", flush=True)
        
        base_output_dir = Path(self.config['output_dir']) / video_name
        dirs = {
            'base': base_output_dir,
            'preprocessed': base_output_dir / '01_preprocessed',
            'segments': base_output_dir / '02_time_segments', 
            'detection_data': base_output_dir / '03_detection_data',
            'analysis_results': base_output_dir / '04_analysis_results',
            'visualizations': base_output_dir / '05_visualizations'
        }
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        try:
            print(f"\n[步骤 1/5] 开始视频时长标准化...", flush=True)
            standardized_path = dirs['preprocessed'] / 'standardized_5min.mp4'
            if not self._standardize_video_duration(str(video_path), str(standardized_path)):
                raise RuntimeError("视频标准化失败。")
            print(f"[步骤 1/5] 视频时长标准化完成。", flush=True)
                
            print(f"\n[步骤 2/5] 开始视频预处理（鱼头裁剪）...", flush=True)
            preprocessed_path = dirs['preprocessed'] / 'cropped_5min.mp4'
            video_clip.run_clipping(
                video_path=str(standardized_path),
                output_video_path=str(preprocessed_path),
                yolo_model_path=self.config['head_model'],
                confidence_thresh=self.config.get('head_conf_threshold', 0.5)
            )
            print(f"[步骤 2/5] 视频预处理完成。", flush=True)
            
            print(f"\n[步骤 3/5] 开始创建时间分段...", flush=True)
            segments = self._create_time_segments(str(preprocessed_path), str(dirs['segments']))
            print(f"[步骤 3/5] 创建时间分段完成。", flush=True)
            
            print(f"\n[步骤 4/5] 开始对所有时间片段进行独立分析...", flush=True)
            segment_analysis_results = {}
            for name, path in tqdm(segments.items(), desc="  分析时间片段", unit="segment"):
                if Path(path).exists():
                    result = self._analyze_segment(name, str(path), dirs)
                    segment_analysis_results[name] = result
                else:
                    print(f"  [警告] 时间片段文件不存在: {path}，跳过。", flush=True)
            print(f"[步骤 4/5] 所有时间片段分析完成。", flush=True)
            
            print(f"\n[步骤 5/5] 开始生成此视频的汇总报告...", flush=True)
            video_summary = self._generate_video_summary(video_name, segment_analysis_results, dirs)
            self.batch_summary.append(video_summary)
            print(f"[步骤 5/5] 汇总报告生成完成。", flush=True)
            
            print(f"\n✅ 视频 {video_name} 处理完成!", flush=True)

        except Exception as e:
            print(f"\n❌ 处理视频 {video_name} 时发生严重错误: {e}", flush=True)
            traceback.print_exc()

    def _generate_video_summary(self, video_name: str, segment_results: Dict, dirs: Dict) -> Dict:
        """为单个视频生成详细的汇总文件。"""
        summary_data_for_csv = []
        for segment_name, results in segment_results.items():
            if results.get('status') != 'Success':
                continue
            
            events = results.get('events', [])
            row = {'segment': segment_name, 'event_count': len(events)}
            
            if events:
                durations = [e['duration_s'] for e in events]
                row['avg_duration_s'] = round(sum(durations) / len(durations), 2)
                row['max_duration_s'] = round(max(durations), 2)
                row['min_duration_s'] = round(min(durations), 2)
            else:
                row.update({'avg_duration_s': 0, 'max_duration_s': 0, 'min_duration_s': 0})
            
            summary_data_for_csv.append(row)
        
        summary_df = pd.DataFrame(summary_data_for_csv)
        summary_csv_path = dirs['base'] / f'{video_name}_summary_report.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"  ✓ 已保存分段对比报告: {summary_csv_path}", flush=True)
        
        report = {
            'video_name': video_name,
            'summary_csv_path': str(summary_csv_path),
            'segments': segment_results
        }
        report_path = dirs['base'] / f'{video_name}_processing_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            for seg_name, seg_data in report['segments'].items():
                seg_data.pop('events', None)
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        return report

    def run_batch_processing(self, video_files: List[Path]):
        """
        根据提供的文件列表进行批量处理。
        """
        print(f"发现 {len(video_files)} 个视频文件待处理。", flush=True)
        
        for video_path in tqdm(video_files, desc="视频批处理进度"):
            self.process_single_video(video_path)
        
        self.generate_final_batch_summary()

    def generate_final_batch_summary(self):
        """为整个批次生成最终的汇总报告。"""
        if not self.batch_summary:
            print("警告: 没有可供生成最终报告的数据。", flush=True)
            return
            
        final_summary_path = Path(self.config['output_dir']) / 'batch_summary.json'
        
        for video_report in self.batch_summary:
            for seg_name, seg_data in video_report.get('segments', {}).items():
                seg_data.pop('events', None)

        batch_summary_data = {
            'run_datetime': datetime.now().isoformat(),
            'total_videos_processed': len(self.batch_summary),
            'processing_config': {k: v for k, v in self.config.items() if k != 'output_dir'},
            'video_reports': self.batch_summary
        }
        
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
        print(f"\n{'='*70}", flush=True)
        print(f"🎉 批处理完成！最终汇总报告已保存至: {final_summary_path}", flush=True)
        print(f"{'='*70}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='鱼嘴行为分层时间分析工作流',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- 核心修改：接受 --input_dir 或 --input_file ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_dir', help='包含输入视频的目录路径。')
    group.add_argument('--input_file', help='单个输入视频文件的路径。')
    
    parser.add_argument('--output_dir', required=True, help='用于保存所有结果的总输出目录路径。')
    parser.add_argument('--head_model', required=True, help='鱼头检测模型 (.pt) 的路径。')
    parser.add_argument('--eye_model', required=True, help='眼部检测模型 (.pt) 的路径。')
    parser.add_argument('--mouth_model', required=True, help='嘴部检测模型 (.pt) 的路径。')

    args = parser.parse_args()

    # --- 核心修改：根据输入参数构建待处理文件列表 ---
    video_files_to_process = []
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.is_file():
            print(f"❌ 致命错误: 指定的输入文件不存在: {args.input_file}", flush=True)
            sys.exit(1)
        video_files_to_process.append(input_path)
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.is_dir():
            print(f"❌ 致命错误: 指定的输入目录不存在: {args.input_dir}", flush=True)
            sys.exit(1)
        video_files_to_process = sorted([p for p in input_path.rglob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']])

    if not video_files_to_process:
        print(f"❌ 错误: 未找到任何可处理的视频文件。", flush=True)
        sys.exit(1)

    # 验证模型路径
    for p in [args.head_model, args.eye_model, args.mouth_model]:
        if not Path(p).exists():
            print(f"❌ 致命错误: 模型路径不存在: {p}", flush=True)
            sys.exit(1)

    # 构建配置字典
    config = {
        'output_dir': args.output_dir,
        'head_model': args.head_model,
        'eye_model': args.eye_model,
        'mouth_model': args.mouth_model,
        'head_conf_threshold': 0.5,
        'mouth_conf_threshold': 0.3,
        'eye_conf_threshold': 0.01,
        'analysis_params': {
            'peak_prominence': 0.01,
            'peak_height_min': 0.01,
            'peak_distance_samples': 10
        },
    }
    
    print("⚙️  任务配置如下:", flush=True)
    print(json.dumps({k:v for k,v in config.items() if k!='analysis_params'}, indent=2), flush=True)
    print("  \"analysis_params\":", json.dumps(config['analysis_params']), flush=True)
    
    # 启动流水线
    pipeline = HierarchicalAnalysisPipeline(config)
    pipeline.run_batch_processing(video_files_to_process)


if __name__ == "__main__":
    main()
