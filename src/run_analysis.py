"""
Sailing Data Analysis - Configuration-Based Runner
===================================================

This script loads parameters from config.py and runs the complete pipeline.

USAGE:
    1. Edit config.py with your parameters
    2. Run: python3 run_analysis.py

This keeps all configuration in one place for easy management.
"""

import sys
import os

# Import configuration
try:
    from config import *
except ImportError:
    print("ERROR: config.py not found!")
    print("Please ensure config.py is in the same directory as this script.")
    sys.exit(1)

# Import the pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import pytz

# ============================================================================
# PREPROCESSING FUNCTIONS (from complete_sailing_pipeline.py)
# ============================================================================

def adjust_angle(angle, adjustment):
    """Adjust angle and wrap to 0-360 range"""
    adjusted = angle + adjustment
    if adjusted > 360:
        adjusted -= 360
    elif adjusted < 0:
        adjusted += 360
    return adjusted

def check_upwind(cog, course_axis):
    """Check if sailing upwind based on COG and course axis"""
    diff = abs((cog - course_axis + 180) % 360 - 180)
    return diff <= 90

def filter_by_time_window(df, start_time_str, end_time_str, date_str, timezone_str='Australia/Sydney'):
    """Filter dataframe by time window in local timezone"""
    df['timestamp_utc'] = pd.to_datetime(df['ISODateTimeUTC'])
    local_tz = pytz.timezone(timezone_str)
    df['timestamp_local'] = df['timestamp_utc'].dt.tz_convert(local_tz)
    
    start_dt = local_tz.localize(
        datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %H:%M")
    )
    end_dt = local_tz.localize(
        datetime.strptime(f"{date_str} {end_time_str}", "%Y-%m-%d %H:%M")
    )
    
    mask = (df['timestamp_local'] >= start_dt) & (df['timestamp_local'] <= end_dt)
    return df[mask].copy()

def preprocess_sailing_data(input_file, start_time, end_time, date_str,
                            course_axis, course_axis_adj,
                            upwind_min_speed, upwind_max_speed, ma_window):
    """Preprocess raw sailing data"""
    print(f"\nPreprocessing: {os.path.basename(input_file)}")
    
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} records")
    
    df_filtered = filter_by_time_window(df, start_time, end_time, date_str)
    print(f"  Filtered to {len(df_filtered)} records in time window")
    
    if len(df_filtered) == 0:
        print("  ERROR: No data in time window!")
        return None
    
    df_proc = pd.DataFrame()
    df_proc['timestamp'] = pd.to_datetime(df_filtered['ISODateTimeUTC'])
    df_proc['latitude'] = df_filtered['Lat']
    df_proc['longitude'] = df_filtered['Lon']
    df_proc['sog_raw'] = df_filtered['SOG']
    df_proc['cog_raw'] = df_filtered['COG']
    df_proc['hdg_raw'] = df_filtered['Heading']
    
    df_proc['cog_adj'] = df_proc['cog_raw'].apply(lambda x: adjust_angle(x, course_axis_adj))
    df_proc['hdg_adj'] = df_proc['hdg_raw'].apply(lambda x: adjust_angle(x, course_axis_adj))
    
    df_proc['SOG_MA'] = df_proc['sog_raw'].rolling(window=ma_window, min_periods=1).mean()
    df_proc['COG_MA'] = df_proc['cog_adj'].rolling(window=ma_window, min_periods=1).mean()
    df_proc['HDG_MA'] = df_proc['hdg_adj'].rolling(window=ma_window, min_periods=1).mean()
    
    df_proc['is_upwind'] = df_proc['COG_MA'].apply(lambda x: check_upwind(x, course_axis))
    df_proc['upwind_speed_ok'] = ((df_proc['SOG_MA'] > upwind_min_speed) & 
                                   (df_proc['SOG_MA'] < upwind_max_speed))
    
    df_proc['SOG'] = 0.0
    df_proc['COG'] = 0.0
    df_proc['HDG'] = 0.0
    
    mask = df_proc['is_upwind'] & df_proc['upwind_speed_ok']
    df_proc.loc[mask, 'SOG'] = df_proc.loc[mask, 'SOG_MA']
    df_proc.loc[mask, 'COG'] = df_proc.loc[mask, 'COG_MA']
    df_proc.loc[mask, 'HDG'] = df_proc.loc[mask, 'HDG_MA']
    
    output_df = df_proc[['timestamp', 'latitude', 'longitude', 'SOG', 'COG', 'HDG']].copy()
    output_df['latitude'] = output_df['latitude'].round(6)
    output_df['longitude'] = output_df['longitude'].round(6)
    output_df['SOG'] = output_df['SOG'].round(2)
    output_df['COG'] = output_df['COG'].round(2)
    output_df['HDG'] = output_df['HDG'].round(2)
    
    valid_count = (output_df['SOG'] > 0).sum()
    print(f"  Valid upwind records: {valid_count} ({100*valid_count/len(output_df):.1f}%)")
    
    return output_df

def process_dataset_for_analysis(df, dataset_name, lag_seconds):
    """Process preprocessed data for leeway analysis"""
    
    df_indexed = df.set_index('timestamp')
    df_1sec = df_indexed.resample('1s').mean()
    
    df_1sec['SOG'] = df_1sec['SOG'].round(2)
    df_1sec['COG'] = df_1sec['COG'].round(2)
    df_1sec['HDG'] = df_1sec['HDG'].round(2)
    
    if 'latitude' in df_1sec.columns:
        df_1sec['latitude'] = df_1sec['latitude'].round(6)
    if 'longitude' in df_1sec.columns:
        df_1sec['longitude'] = df_1sec['longitude'].round(6)
    
    df_1sec['Leeway'] = (df_1sec['HDG'] - df_1sec['COG']).abs().round(2)
    df_1sec['SOG_Change'] = df_1sec['SOG'].diff().round(3)
    df_1sec['Leeway_Lag'] = df_1sec['Leeway'].shift(lag_seconds).round(2)
    df_1sec = df_1sec.reset_index()
    
    df_1sec['is_upwind'] = (df_1sec['SOG'] > 0) & (df_1sec['COG'] > 0) & (df_1sec['HDG'] > 0)
    df_1sec['period_change'] = df_1sec['is_upwind'].astype(int).diff().fillna(0)
    df_1sec['period_id'] = (df_1sec['period_change'] != 0).cumsum()
    
    upwind_periods = df_1sec[df_1sec['is_upwind']].groupby('period_id').agg({
        'timestamp': ['first', 'last', 'count']
    })
    upwind_periods.columns = ['_'.join(col).strip() for col in upwind_periods.columns.values]
    upwind_periods = upwind_periods.reset_index()
    upwind_periods['duration_seconds'] = (
        upwind_periods['timestamp_last'] - upwind_periods['timestamp_first']
    ).dt.total_seconds()
    
    top_3_periods = upwind_periods.nlargest(3, 'duration_seconds')
    top_3_period_ids = top_3_periods['period_id'].tolist()
    
    top3_data = df_1sec[df_1sec['period_id'].isin(top_3_period_ids) & df_1sec['is_upwind']].copy()
    top3_data = top3_data.dropna(subset=['SOG_Change', 'Leeway_Lag'])
    
    def categorize_leeway(leeway):
        if leeway < 2:
            return '0-2°'
        elif leeway < 4:
            return '2-4°'
        elif leeway < 6:
            return '4-6°'
        elif leeway < 8:
            return '6-8°'
        elif leeway < 10:
            return '8-10°'
        elif leeway < 12:
            return '10-12°'
        else:
            return '12+°'
    
    top3_data['Leeway_Bucket'] = top3_data['Leeway_Lag'].apply(categorize_leeway)
    
    corr_accel = stats.pearsonr(top3_data['Leeway_Lag'], top3_data['SOG_Change'])
    corr_sog = stats.pearsonr(top3_data['Leeway_Lag'], top3_data['SOG'])
    
    return top3_data, corr_accel, corr_sog, top_3_period_ids, dataset_name

def create_analysis_plots(datasets_data, lag_seconds, output_dir):
    """Create comprehensive analysis plots"""
    
    bucket_names = ['0-2°', '2-4°', '4-6°', '6-8°', '8-10°', '10-12°', '12+°']
    bucket_colors = ['darkgreen', 'limegreen', 'yellowgreen', 'yellow', 'orange', 'orangered', 'red']
    
    num_datasets = len(datasets_data)
    
    fig = plt.figure(figsize=(24, 6 * num_datasets))
    fig.suptitle(f'Leeway Analysis ({lag_seconds} sec lag)', fontsize=20, fontweight='bold')
    
    gs = fig.add_gridspec(num_datasets, 3, hspace=0.3, wspace=0.3)
    
    for idx, (data, corr_accel, corr_sog, period_ids, name) in enumerate(datasets_data):
        
        buckets = []
        for bucket_name in bucket_names:
            bucket_data = data[data['Leeway_Bucket'] == bucket_name]
            buckets.append(bucket_data)
        
        ax1 = fig.add_subplot(gs[idx, 0])
        means_accel = [bucket['SOG_Change'].mean() if len(bucket) > 0 else 0 for bucket in buckets]
        bars = ax1.bar(bucket_names, means_accel, color=bucket_colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax1.set_ylabel('Mean Acceleration (kt/s)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Leeway Bucket', fontsize=12, fontweight='bold')
        ax1.set_title(f'{name}: Mean Acceleration by Leeway', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        for bar, mean in zip(bars, means_accel):
            if mean != 0:
                height = bar.get_height()
                y_pos = height + 0.005 if height > 0 else height - 0.01
                ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{mean:.4f}', ha='center', va='bottom' if height > 0 else 'top', 
                        fontsize=9, fontweight='bold')
        
        ax2 = fig.add_subplot(gs[idx, 1])
        means_sog = [bucket['SOG'].mean() if len(bucket) > 0 else 0 for bucket in buckets]
        bars = ax2.bar(bucket_names, means_sog, color=bucket_colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Mean SOG (knots)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Leeway Bucket', fontsize=12, fontweight='bold')
        ax2.set_title(f'{name}: Mean SOG by Leeway', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)
        for bar, mean in zip(bars, means_sog):
            if mean != 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.2f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        ax3 = fig.add_subplot(gs[idx, 2])
        colors_period = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for pidx, period_id in enumerate(period_ids):
            period_data = data[data['period_id'] == period_id]
            ax3.scatter(period_data['Leeway_Lag'], period_data['SOG'],
                      color=colors_period[pidx], alpha=0.4, s=20, label=f'Period {pidx+1}',
                      edgecolors='black', linewidth=0.3)
        
        z = np.polyfit(data['Leeway_Lag'], data['SOG'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(data['Leeway_Lag'].min(), data['Leeway_Lag'].max(), 100)
        ax3.plot(x_trend, p(x_trend), "red", linewidth=2, alpha=0.8, label=f'Trend (slope={z[0]:.3f})')
        ax3.set_xlabel(f'Leeway {lag_seconds}s Ago (degrees)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('SOG (knots)', fontsize=12, fontweight='bold')
        ax3.set_title(f'{name}: SOG vs Leeway\nr = {corr_sog[0]:.4f}, p = {corr_sog[1]:.6f}', 
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/leeway_analysis_lag{lag_seconds}s.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: leeway_analysis_lag{lag_seconds}s.png")
    
    plt.close()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_complete_pipeline():
    """Execute complete analysis pipeline"""
    
    print("=" * 80)
    print("SAILING DATA ANALYSIS - Configuration-Based Run")
    print("=" * 80)
    
    print("\nConfiguration Summary:")
    print(f"  Date: {DATE_STR}")
    print(f"  Time Window: {START_TIME_AEDT} - {END_TIME_AEDT} AEDT")
    print(f"  Course Axis: {COURSE_AXIS}°")
    print(f"  Speed Range: {UPWIND_MIN_SPEED}-{UPWIND_MAX_SPEED} knots")
    print(f"  MA Window: {MA_WINDOW} samples")
    print(f"  Lag: {LAG_SECONDS} seconds")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80)
    
    processed_datasets = []
    dataset_names = []
    
    if RAW_DATA_FILE_1 and os.path.exists(RAW_DATA_FILE_1):
        df1 = preprocess_sailing_data(
            RAW_DATA_FILE_1, START_TIME_AEDT, END_TIME_AEDT, DATE_STR,
            COURSE_AXIS, COURSE_AXIS_ADJ,
            UPWIND_MIN_SPEED, UPWIND_MAX_SPEED, MA_WINDOW
        )
        if df1 is not None:
            processed_datasets.append(df1)
            dataset_names.append(os.path.splitext(os.path.basename(RAW_DATA_FILE_1))[0])
            
            output_file = f"{OUTPUT_DIR}/{dataset_names[-1]}_processed.csv"
            df1.to_csv(output_file, index=False)
            print(f"  Saved preprocessed data: {output_file}")
    
    if RAW_DATA_FILE_2 and os.path.exists(RAW_DATA_FILE_2):
        df2 = preprocess_sailing_data(
            RAW_DATA_FILE_2, START_TIME_AEDT, END_TIME_AEDT, DATE_STR,
            COURSE_AXIS, COURSE_AXIS_ADJ,
            UPWIND_MIN_SPEED, UPWIND_MAX_SPEED, MA_WINDOW
        )
        if df2 is not None:
            processed_datasets.append(df2)
            dataset_names.append(os.path.splitext(os.path.basename(RAW_DATA_FILE_2))[0])
            
            output_file = f"{OUTPUT_DIR}/{dataset_names[-1]}_processed.csv"
            df2.to_csv(output_file, index=False)
            print(f"  Saved preprocessed data: {output_file}")
    
    if len(processed_datasets) == 0:
        print("\nERROR: No datasets to process!")
        return
    
    print("\n" + "=" * 80)
    print("STEP 2: LEEWAY ANALYSIS")
    print("=" * 80)
    
    analyzed_datasets = []
    for df, name in zip(processed_datasets, dataset_names):
        print(f"\nAnalyzing: {name}")
        result = process_dataset_for_analysis(df, name, LAG_SECONDS)
        analyzed_datasets.append(result)
        
        data, corr_accel, corr_sog, _, _ = result
        print(f"  Records in analysis: {len(data)}")
        print(f"  Leeway vs Acceleration: r = {corr_accel[0]:.4f}, p = {corr_accel[1]:.6f}")
        print(f"  Leeway vs SOG: r = {corr_sog[0]:.4f}, p = {corr_sog[1]:.6f}")
    
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    create_analysis_plots(analyzed_datasets, LAG_SECONDS, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    for data, corr_accel, corr_sog, _, name in analyzed_datasets:
        print(f"\n{name.upper()}:")
        print(f"  Records: {len(data)}")
        print(f"  SOG: Mean={data['SOG'].mean():.2f} kt, Std={data['SOG'].std():.2f} kt")
        print(f"  Leeway: Mean={data['Leeway'].mean():.2f}°, Std={data['Leeway'].std():.2f}°")
        print(f"  Correlations:")
        print(f"    Leeway vs Acceleration: r={corr_accel[0]:.4f}, p={corr_accel[1]:.6f}")
        print(f"    Leeway vs SOG: r={corr_sog[0]:.4f}, p={corr_sog[1]:.6f}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_complete_pipeline()
