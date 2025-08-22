from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class FishMouthAnalyzer:
    """
    分析鱼嘴张合行为。
    使用标准化数据进行事件检测，但报告原始数据的统计值，并额外输出眼部对角线长度。
    """
    def __init__(self,
                 detection_df: pd.DataFrame,
                 time_col: str = 'timestamp_sec',
                 frame_col: str = 'frame_number',
                 area_col: str = 'mouth_area',
                 height_col: str = 'mouth_height',
                 eye_diagonal_col: str = 'eye_box_diagonal'):
        """
        初始化分析器。
        """
        self.time_col = time_col
        self.frame_col = frame_col
        self.area_col = area_col
        self.height_col = height_col
        self.eye_diagonal_col = eye_diagonal_col
        
        self.df_processed = self._validate_and_normalize_data(detection_df)
        
        # 初始化峰值和谷值属性，供可视化模块使用
        self.peaks: Optional[np.ndarray] = None
        self.valleys: Optional[np.ndarray] = None


    def _validate_and_normalize_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """验证并使用眼部尺寸标准化数据。"""
        if df is None or df.empty:
            print("错误: 输入的DataFrame为空或None。")
            return None
            
        required_cols = [self.time_col, self.frame_col, self.area_col, self.height_col, self.eye_diagonal_col]
        for col in required_cols:
            if col not in df.columns:
                print(f"错误: 必需的列 '{col}' 在DataFrame中未找到。")
                return None
        
        df_copy = df.copy()
        epsilon = 1e-6
        
        eye_diagonal = df_copy[self.eye_diagonal_col].replace(0, np.nan).ffill().bfill()
        
        if eye_diagonal.isnull().all():
            print("警告: 眼部对角线数据全部为空。无法进行标准化。")
            df_copy['normalized_area'] = 0
            df_copy['normalized_height'] = 0
        else:
            df_copy['normalized_area'] = df_copy[self.area_col] / (eye_diagonal**2 + epsilon)
            df_copy['normalized_height'] = df_copy[self.height_col] / (eye_diagonal + epsilon)
        
        print("数据验证和标准化成功。")
        return df_copy

    def analyze(self,
                peak_prominence: float = 0.01,
                peak_height_min: float = 0.01,
                peak_distance_samples: int = 1) -> List[Dict]:
        """
        执行完整的开合事件分析。
        """
        if self.df_processed is None or self.df_processed.empty:
            print("分析中止: 数据不可用。")
            return []

        # 使用标准化数据进行峰谷检测
        normalized_area = self.df_processed['normalized_area'].fillna(0)
        
        peaks_indices, _ = find_peaks(normalized_area, height=peak_height_min, prominence=peak_prominence, distance=peak_distance_samples)
        valleys_indices, _ = find_peaks(-normalized_area, prominence=peak_prominence, distance=peak_distance_samples)
        
        self.peaks = self.df_processed.index[peaks_indices]
        self.valleys = self.df_processed.index[valleys_indices]
        
        if len(self.peaks) == 0 or len(self.valleys) == 0:
            print("在标准化数据中未找到足够的峰或谷。")
            return []

        # (极值点排序和交替清理逻辑保持不变)
        extrema = []
        for p_idx in self.peaks: extrema.append({'index': p_idx, 'type': 'peak', 'value': normalized_area[p_idx]})
        for v_idx in self.valleys: extrema.append({'index': v_idx, 'type': 'valley', 'value': normalized_area[v_idx]})
        extrema.sort(key=lambda x: x['index'])
        if not extrema: return []
        cleaned_extrema = [extrema[0]]
        for i in range(1, len(extrema)):
            if extrema[i]['type'] != cleaned_extrema[-1]['type']:
                cleaned_extrema.append(extrema[i])
            else:
                if extrema[i]['type'] == 'peak' and extrema[i]['value'] > cleaned_extrema[-1]['value']:
                    cleaned_extrema[-1] = extrema[i]
                elif extrema[i]['type'] == 'valley' and extrema[i]['value'] < cleaned_extrema[-1]['value']:
                    cleaned_extrema[-1] = extrema[i]

        events = []
        for i in range(len(cleaned_extrema) - 2):
            if cleaned_extrema[i]['type'] == 'valley' and \
               cleaned_extrema[i+1]['type'] == 'peak' and \
               cleaned_extrema[i+2]['type'] == 'valley':
                
                start_idx, end_idx = cleaned_extrema[i]['index'], cleaned_extrema[i+2]['index']
                peak_idx = cleaned_extrema[i+1]['index']
                
                # 从原始数据(df_processed)中提取事件片段
                segment_df = self.df_processed.iloc[start_idx : end_idx + 1]
                if segment_df.empty: continue
                
                # --- 关键修改：报告原始数据和新增的眼部对角线长度 ---
                area_segment = segment_df[self.area_col]
                height_segment = segment_df[self.height_col]
                
                # 获取面积达到峰值那一帧的眼部对角线长度
                eye_diagonal_at_peak = self.df_processed.loc[peak_idx, self.eye_diagonal_col]

                event_data = {
                    "event_id": len(events) + 1,
                    "start_frame": int(segment_df[self.frame_col].iloc[0]),
                    "end_frame": int(segment_df[self.frame_col].iloc[-1]),
                    "duration_s": round((segment_df[self.time_col].iloc[-1] - segment_df[self.time_col].iloc[0]), 2),
                    "eye_box_diagonal_at_peak": round(eye_diagonal_at_peak, 2), # 新增字段
                    "area": {"max": round(area_segment.max(), 2), "min": round(area_segment.min(), 2), "mean": round(area_segment.mean(), 2), "peak_frame": int(segment_df.loc[area_segment.idxmax(), self.frame_col])},
                    "height": {"max": round(height_segment.max(), 2), "min": round(height_segment.min(), 2), "mean": round(height_segment.mean(), 2), "peak_frame": int(segment_df.loc[height_segment.idxmax(), self.frame_col])},
                    "_internal": { "start_index": start_idx, "end_index": end_idx }
                }
                events.append(event_data)

        print(f"分析完成。找到 {len(events)} 个严格交替的事件。")
        return events
