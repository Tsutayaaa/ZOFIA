import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Optional, List, Dict

def plot_analysis_results(
    df_full: pd.DataFrame,
    events: List[Dict],
    time_col: str,
    area_col: str,
    frame_col: str = 'frame_number',
    output_image_path: Optional[str] = None
):
    """
    Generates a plot of the analysis results based on the event dictionary.
    Only the peak (max) and valley (min) points that constitute each valid event are marked.
    """
    if df_full is None or df_full.empty:
        print("Plotting failed: Input DataFrame is empty or None.")
        return

    fig, ax1 = plt.subplots(figsize=(18, 9))

    # Plot the main curve
    color_area = 'tab:blue'
    ax1.set_xlabel(f'Time ({time_col})')
    ax1.set_ylabel(f'Mouth Opening Signal ({area_col})', color=color_area)
    ax1.plot(df_full[time_col], df_full[area_col], 
             label=f'Signal ({area_col})', alpha=0.7, color=color_area)
    ax1.tick_params(axis='y', labelcolor=color_area)
    ax1.grid(True, linestyle=':', alpha=0.6)

    if events:
        peak_points = []
        valley_points = []

        for event in events:
            peak_frame = event['area']['peak_frame']
            peak_row = df_full[df_full[frame_col] == peak_frame]
            if not peak_row.empty:
                peak_points.append((peak_row[time_col].iloc[0], peak_row[area_col].iloc[0]))

            start_frame = event['start_frame']
            end_frame = event['end_frame']
            start_row = df_full[df_full[frame_col] == start_frame]
            end_row = df_full[df_full[frame_col] == end_frame]
            if not start_row.empty:
                valley_points.append((start_row[time_col].iloc[0], start_row[area_col].iloc[0]))
            if not end_row.empty:
                valley_points.append((end_row[time_col].iloc[0], end_row[area_col].iloc[0]))

        if peak_points:
            peak_times, peak_values = zip(*peak_points)
            ax1.scatter(peak_times, peak_values, color='red', s=60, marker='^', label='Event Peak', zorder=5)

        if valley_points:
            valley_times, valley_values = zip(*valley_points)
            ax1.scatter(valley_times, valley_values, color='green', s=60, marker='v', label='Event Valley', zorder=5)

    ax1.legend(loc='upper left')
    plt.title('Fish Mouth Opening Event Analysis')
    fig.tight_layout()
    
    if output_image_path:
        try:
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            fig.savefig(output_image_path, dpi=300, bbox_inches='tight')
            print(f"    ✓ Plot saved to: {output_image_path}")
        except Exception as e:
            print(f"    Error saving plot: {e}")
    
    try:
        # In a non-interactive environment like Slurm, this might raise a warning.
        # It's generally safe to ignore, as we are saving the file anyway.
        plt.show()
    except Exception as e:
        print(f"    Cannot display plot interactively: {e}")
    finally:
        plt.close(fig) # Ensure the figure is closed to free up memory
