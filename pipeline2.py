import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from fishmouth_behavior.preprocess import video_clip
    from fishmouth_behavior.mouth_detect import detector
    from fishmouth_behavior.analysis.analyzer import FishMouthAnalyzer
    from fishmouth_behavior.visualization import plotter
except ImportError as e:
    print(f"FATAL: A core module could not be imported: {e}", flush=True)
    sys.exit(1)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DirectAnalysisPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.batch_summary = []

    def _create_preview_video(self, input_video_path: str, output_video_path: str):
        print(f"      Starting to create preview video...", flush=True)
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"      ERROR: Cannot open video file for preview creation: {input_video_path}", flush=True)
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        preview_detector = detector.FishmouthDetector(
            model_path_mouth=self.config['mouth_model'],
            model_path_eyes=self.config['eye_model'],
            mouth_conf=self.config.get('mouth_conf_threshold', 0.3),
            eye_conf=self.config.get('eye_conf_threshold', 0.3)
        )
        
        with tqdm(total=total_frames, desc="      Preview video rendering", unit="frame", leave=False) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                visual_results = preview_detector.detect_for_visualization(frame)
                annotated_frame = frame.copy()
                mask_area = 0

                if visual_results.get('eye_boxes') is not None:
                    for box in visual_results['eye_boxes'].xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if visual_results.get('mouth_masks') is not None and visual_results['mouth_masks'].data.shape[0] > 0:
                    mask_tensor = visual_results['mouth_masks'].data[0]
                    mask_np = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
                    
                    target_height, target_width = annotated_frame.shape[:2]
                    if mask_np.shape[0] != target_height or mask_np.shape[1] != target_width:
                        resized_mask_np = cv2.resize(mask_np, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                    else:
                        resized_mask_np = mask_np

                    color_mask = np.zeros_like(annotated_frame)
                    color_mask[resized_mask_np == 1] = [255, 0, 0]
                    annotated_frame = cv2.addWeighted(annotated_frame, 1.0, color_mask, 0.5, 0)
                    mask_area = int(np.sum(resized_mask_np))

                text = f"Mask Area: {mask_area} px"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = width - text_size[0] - 10
                text_y = text_size[1] + 10
                cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

                writer.write(annotated_frame)
                pbar.update(1)

        cap.release()
        writer.release()
        print(f"      ✓ Preview video saved to: {output_video_path}", flush=True)

    def process_single_video(self, video_path: Path) -> None:
        video_name = video_path.stem
        print(f"\n{'='*70}", flush=True)
        print(f"🎬 Starting to process video: {video_name} [Time: {datetime.now()}]", flush=True)
        print(f"{'='*70}", flush=True)
        
        base_output_dir = Path(self.config['output_dir']) / video_name
        dirs = {
            'base': base_output_dir,
            'preprocessed': base_output_dir / '01_preprocessed',
            'detection_data': base_output_dir / '02_detection_data',
            'analysis_results': base_output_dir / '03_analysis_results',
            'visualizations': base_output_dir / '04_visualizations'
        }
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        try:
            print(f"\n[Step 1/5] Starting video preprocessing (head cropping)...", flush=True)
            preprocessed_path = dirs['preprocessed'] / f'{video_name}_cropped.mp4'
            video_clip.run_clipping(
                video_path=str(video_path),
                output_video_path=str(preprocessed_path),
                yolo_model_path=self.config['head_model'],
                confidence_thresh=self.config.get('head_conf_threshold', 0.5)
            )
            print(f"[Step 1/5] Video preprocessing complete.", flush=True)
            
            print(f"\n[Step 2/5] Starting feature detection...", flush=True)
            detection_df = detector.process_video_and_extract_data(
                video_source_path=str(preprocessed_path),
                yolo_model_path_eyes=self.config['eye_model'],
                yolo_model_path_mouth=self.config['mouth_model'],
                mouth_conf_threshold=self.config.get('mouth_conf_threshold', 0.3),
                eye_conf_threshold=self.config.get('eye_conf_threshold', 0.3)
            )
            print(f"[Step 2/5] Feature detection complete.", flush=True)

            if detection_df is None or detection_df.empty:
                raise RuntimeError("Feature detection failed, no data was generated.")
            
            detection_csv_path = dirs['detection_data'] / f'{video_name}_detection.csv'
            detection_df.to_csv(detection_csv_path, index=False)
            
            if self.config.get('create_preview', False):
                print(f"\n[Step 3/5] Starting to generate preview video...", flush=True)
                preview_video_path = dirs['visualizations'] / f'{video_name}_preview.mp4'
                self._create_preview_video(str(preprocessed_path), str(preview_video_path))
                print(f"[Step 3/5] Preview video generation complete.", flush=True)
            
            print(f"\n[Step 4/5] Starting behavior analysis...", flush=True)
            analyzer = FishMouthAnalyzer(detection_df)
            events = analyzer.analyze(**self.config.get('analysis_params', {}))
            print(f"[Step 4/5] Behavior analysis complete. Found {len(events)} events.", flush=True)
            
            events_json_path = dirs['analysis_results'] / f'{video_name}_events.json'
            with open(events_json_path, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            print(f"\n[Step 5/5] Starting to generate plots and reports...", flush=True)
            plot_path = dirs['visualizations'] / f'{video_name}_analysis_plot.png'
            plotter.plot_analysis_results(
                df_full=analyzer.df_processed,
                events=events,
                time_col='timestamp_sec',
                area_col='normalized_area',
                output_image_path=str(plot_path)
            )
            
            summary_txt_path = dirs['base'] / f'{video_name}_summary.txt'
            with open(summary_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Summary for video: {video_name}\n")
                f.write(f"Total detected events: {len(events)}\n")
                f.write(f"Results directory: {dirs['base']}\n")

            self.batch_summary.append({
                'video_name': video_name,
                'event_count': len(events),
                'output_path': str(dirs['base'])
            })
            print(f"[Step 5/5] Plots and reports generation complete.", flush=True)
            
            print(f"\n✅ Video {video_name} processed successfully!", flush=True)

        except Exception as e:
            print(f"\n❌ A critical error occurred while processing video {video_name}: {e}", flush=True)
            traceback.print_exc()

    def run_batch_processing(self, video_files: List[Path]):
        print(f"Found {len(video_files)} video files to process.", flush=True)
        
        for video_path in tqdm(video_files, desc="Overall Progress"):
            self.process_single_video(video_path)
        
        self.generate_final_batch_summary()

    def generate_final_batch_summary(self):
        if not self.batch_summary:
            print("Warning: No data available to generate a final summary.", flush=True)
            return
            
        final_summary_path = Path(self.config['output_dir']) / 'batch_summary.json'
        
        batch_summary_data = {
            'run_datetime': datetime.now().isoformat(),
            'total_videos_processed': len(self.batch_summary),
            'processing_config': {k: v for k, v in self.config.items() if k != 'output_dir'},
            'video_reports': self.batch_summary
        }
        
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_summary_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
        print(f"\n{'='*70}", flush=True)
        print(f"🎉 Batch processing complete! Final summary saved to: {final_summary_path}", flush=True)
        print(f"{'='*70}", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description='Direct Fish Mouth Behavior Analysis Workflow (No Time Limit)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_dir', help='Path to the directory containing input videos.')
    group.add_argument('--input_file', help='Path to a single input video file.')
    
    parser.add_argument('--output_dir', required=True, help='Path to the main output directory for all results.')
    parser.add_argument('--head_model', required=True, help='Path to the fish head detection model (.pt).')
    parser.add_argument('--eye_model', required=True, help='Path to the eye detection model (.pt).')
    parser.add_argument('--mouth_model', required=True, help='Path to the mouth detection model (.pt).')
    parser.add_argument('--create_preview', action='store_true', help='If set, will create a preview video with detections for each video.')

    args = parser.parse_args()

    video_files_to_process = []
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.is_file():
            print(f"❌ FATAL ERROR: The specified input file does not exist: {args.input_file}", flush=True)
            sys.exit(1)
        video_files_to_process.append(input_path)
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.is_dir():
            print(f"❌ FATAL ERROR: The specified input directory does not exist: {args.input_dir}", flush=True)
            sys.exit(1)
        video_files_to_process = sorted([p for p in input_path.rglob('*') if p.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']])

    if not video_files_to_process:
        print(f"❌ ERROR: No video files found to process.", flush=True)
        sys.exit(1)

    for p in [args.head_model, args.eye_model, args.mouth_model]:
        if not Path(p).exists():
            print(f"❌ FATAL ERROR: Model path does not exist: {p}", flush=True)
            sys.exit(1)

    config = {
        'output_dir': args.output_dir,
        'head_model': args.head_model,
        'eye_model': args.eye_model,
        'mouth_model': args.mouth_model,
        'create_preview': args.create_preview,
        'head_conf_threshold': 0.5,
        'mouth_conf_threshold': 0.3,
        'eye_conf_threshold': 0.3,
        'analysis_params': {
            'peak_prominence': 0.01,
            'peak_height_min': 0.01,
            'peak_distance_samples': 10
        },
    }
    
    print("⚙️  Task configuration:", flush=True)
    print(json.dumps({k:v for k,v in config.items() if k!='analysis_params'}, indent=2), flush=True)
    print("  \"analysis_params\":", json.dumps(config['analysis_params']), flush=True)
    
    pipeline = DirectAnalysisPipeline(config)
    pipeline.run_batch_processing(video_files_to_process)

if __name__ == "__main__":
    main()
